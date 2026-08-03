# =============================================================================
# Private driver helpers
# =============================================================================

"""
	_initialise_waveform!(W, R, master_modes, master_eigenvalues,
						  master_modes_derivatives, unit_offset, model)

Initialise the linear-monomial coefficients of `W` and `R` from spectral data:

- `W[:, 1, eᵣ] = master_modes[:, r]` and, for `ORD > 1`,
  `W[:, k, eᵣ] = master_modes_derivatives[:, k-1, r]`.
- `R[r, eᵣ] = master_eigenvalues[r]`.

Also embeds the external-system linear dynamics into the external rows of `R` via
`_embed_external_dynamics!` when `model.external_system !== nothing`.
"""
function _initialise_waveform!(
        W::Parametrisation, R::ReducedDynamics,
        master_modes, master_eigenvalues,
        master_modes_derivatives, unit_offset::Int, model
)
    ROM = length(master_eigenvalues)
    ORD_W = size(W.poly.coefficients, 2)
    for r in 1:ROM
        idx_er = r + unit_offset
        W.poly.coefficients[:, 1, idx_er] .= view(master_modes, :, r)
        if master_modes_derivatives !== nothing
            for k in 2:ORD_W
                W.poly.coefficients[:, k, idx_er] .= view(
                    master_modes_derivatives, :, k -
                                                 1, r)
            end
        end
        R.poly.coefficients[r, idx_er] = master_eigenvalues[r]
    end
    if model.external_system !== nothing
        _embed_external_dynamics!(R, model.external_system.first_order_dynamics, multiindex_set(W))
    end
    return nothing
end

"""
	_solve_external_directions!(W, R, partial_ctx_for, model, ml_cache, sym, N_EXT, ROM, unit_offset)

Solve the `N_EXT` external forcing directions in increasing variable order.

`partial_ctx_for(e)` returns the *partial* context for external direction `e`: one in which
the external generalised eigenvectors `Φ_ext` are known only for the directions already
solved, and zero from `e` onwards.  Solving in order is what makes that possible — column
`e` of the external column-polynomial recurrences reads only `Φ_ext,j` with `Λ_ext[j, e] ≠ 0`,
hence only `j ≤ e` for the upper-triangular `Λ_ext` that the solver requires.

When conjugate-symmetry is active, secondary external monomials are filled from their
primaries via `fill_conjugate_monomial!` in the same pass rather than afterwards, so that
`Φ_ext,k` is populated for every `k < e` regardless of how it was obtained.  This is safe
because `primary_pairs` guarantees a secondary's source has the smaller index.

All external linear monomials are marked in `sym.skip_bits` after this call so
the main `solve_cohomological_equations!` loop does not overwrite them.
"""
function _solve_external_directions!(
        W, R, partial_ctx_for, model, ml_cache,
        sym::ConjugateSymmetryData{NoConjugatePermutation}, N_EXT::Int, ROM::Int, unit_offset::Int
)
    for e in 1:N_EXT
        idx = ROM + e + unit_offset
        solve_single_monomial!(W, R, idx, partial_ctx_for(e), model, ml_cache)
        @inbounds sym.skip_bits[idx] = true
    end
    return nothing
end

function _solve_external_directions!(
        W, R, partial_ctx_for, model, ml_cache,
        sym::ConjugateSymmetryData{<:SVector}, N_EXT::Int, ROM::Int, unit_offset::Int
)
    N_EXT == 0 && return nothing
    for e in 1:N_EXT
        idx = ROM + e + unit_offset
        if @inbounds sym.skip_bits[idx]
            # Secondary: its source is a pre-marked primary with a smaller index, so it is
            # already available — `_build_conjugate_symmetry` only ever marks the larger
            # index of a pair, and `monomial_map` is symmetric.
            fill_conjugate_monomial!(W, R, idx, sym.monomial_map[idx], sym)
        else
            solve_single_monomial!(W, R, idx, partial_ctx_for(e), model, ml_cache)
        end
    end
    for e in 1:N_EXT
        @inbounds sym.skip_bits[ROM + e + unit_offset] = true
    end
    return nothing
end

"""
	_make_sparse_solver(MT, linear_terms, FOM, ROM) -> Union{SparseLinearSolverState, Nothing}

Dispatch helper: returns a `SparseLinearSolverState` when `MT <: SparseMatrixCSC`
(sparse path), or `nothing` for all other matrix types (dense path).
"""
_make_sparse_solver(::Type{<:AbstractMatrix}, _, ::Int, ::Int) = nothing
function _make_sparse_solver(
        ::Type{<:SparseMatrixCSC}, linear_terms, FOM::Int, ROM::Int
)
    L_template, L_mappings = precompute_sparse_L_template(linear_terms)
    T = eltype(L_template)
    return SparseLinearSolverState{T}(L_template, L_mappings, FOM, ROM)
end

"""
	_build_context(linear_terms, generalised_eigenmodes, lambda_diag,
				   inv_ops, orth_ops, resonance_set, linear_skip_set,
				   lower_order, buffers, sparse_solver) -> CohomologicalContext

Construct a `CohomologicalContext` from pre-assembled operator data and shared
resources.  All type parameters are inferred from the arguments.
"""
function _build_context(
        linear_terms::NTuple{ORDP1, MT},
        generalised_eigenmodes::Matrix{T},
        lambda_diag::Vector{T},
        inv_ops::InvarianceOperators{T},
        orth_ops::OrthogonalityOperators{T},
        resonance_set::ResonanceSet,
        linear_skip_set::Set{Int},
        lower_order::LowerOrderResources{NVAR, T},
        buffers::CohomologicalBuffers{T},
        sparse_solver
) where {ORDP1, MT, T, NVAR}
    FOM = size(generalised_eigenmodes, 1)
    ORD = ORDP1 - 1
    LT = eltype(MT)
    return CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT}(
        linear_terms, generalised_eigenmodes, lambda_diag,
        inv_ops, orth_ops,
        resonance_set, linear_skip_set,
        lower_order, buffers, sparse_solver
    )
end

# =============================================================================
# High-level driver
# =============================================================================

"""
	solve_cohomological_problem(
		model, mset, master_eigenvalues,
		master_modes, left_eigenmodes, resonance_set;
		initial_W = nothing, initial_R = nothing,
		master_modes_derivatives = nothing,
		conjugate_permutation = nothing
	) -> (W, R)

High-level driver that assembles a [`CohomologicalContext`](@ref) from raw
spectral data and solves the full set of cohomological equations.

The external dynamics enter through `model.external_system.first_order_dynamics`, which
`_embed_external_dynamics!` copies into the external rows of `R`; the superharmonics `s`
are then contracted against `diag(Λ)` read back from `R`, so the external part of `s` is
the diagonal of the external linear matrix.  That matrix must be upper triangular — the
requirement is enforced by the `ExternalSystem` constructors and explained in the
`ExternalSystems` module docstring.  The linear-operator tuple is read from
`model.linear_terms`.

## Steps

1. Create (or reuse) [`Parametrisation`](@ref) `W` and [`ReducedDynamics`](@ref) `R`
   and initialise master-mode linear monomials.
2. Build shared resources (buffers, lower-order coupling data, sparse solver state).
3. Solve the linear cohomological equations for each external forcing direction via
   a partial context in which the not-yet-solved external columns of
   `generalised_right_eigenmodes` are set to zero.
4. Build the full `generalised_right_eigenmodes` from the solved external directions.
5. Assemble the full context and call [`solve_cohomological_equations!`](@ref).

## Arguments

- `model :: NDOrderModel` — full-order model; `model.linear_terms` provides `(B₀,…,B_ORD)`.
- `mset :: MultiindexSet{NVAR}` — multiindex set over all `NVAR = ROM + N_EXT` variables.
- `master_eigenvalues :: SVector{ROM, ComplexF64}`.
- `master_modes :: Matrix{ComplexF64}` — size `FOM × ROM`.
- `left_eigenmodes :: AbstractMatrix{ComplexF64}` — size `FOM × ROM`.
- `resonance_set :: ResonanceSet`.
- `initial_W`, `initial_R` — optionally supply already-initialised objects.
- `master_modes_derivatives` — `FOM × (ORD-1) × ROM`; required when `ORD > 1`.
- `left_modes_derivatives` — `FOM × (ORD-1) × ROM`; the lower-order left
  eigenvector blocks `left_modes_derivatives[:, j, r] = φ_{r,j}` (as returned by
  `solve_left`); required when `ORD > 1`. The orthogonality row operators are
  read directly off these blocks — no eigenvalue folding.
- `conjugate_permutation` — optional `AbstractVector{Int}` of length NVAR encoding the
  conjugate permutation on modes.  `perm[i] = j` means mode `j` is the complex conjugate
  of mode `i`.  When `nothing` (default), no conjugate-symmetry exploitation is performed.
  The caller must verify that the eigenvectors satisfy the conjugate symmetry before passing
  this argument — two eigenvalues forming a conjugate pair is necessary but not sufficient
  (the eigenspace must be one-dimensional, or the eigenvectors must be chosen conjugately).
- `validate_mset` — check `mset` against the contract the solve assumes (variable count,
  minimum degree, unit multiindices, downward closure, and closure under
  `conjugate_permutation`) via [`validate_multiindex_set`](@ref), throwing an
  `ArgumentError` naming the offending exponent.  Default `true`: every path into the
  solve passes through here, so this is where the contract is enforced.  `parametrise`
  checks first and passes `false` to avoid walking the set twice.
- `show_progress` — print a progress line to `stderr` while solving (default: `true`).
  Suppressed automatically when `stderr` is not a TTY.

## Returns

`(W, R)` — the solved [`Parametrisation`](@ref) and [`ReducedDynamics`](@ref).
"""
function solve_cohomological_problem(
        model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, LT, MT},
        mset::MultiindexSet{NVAR},
        master_eigenvalues::SVector{ROM, ComplexF64},
        master_modes::Matrix{ComplexF64},
        left_eigenmodes::AbstractMatrix{ComplexF64},
        resonance_set::ResonanceSet;
        initial_W::Union{Nothing, Parametrisation} = nothing,
        initial_R::Union{Nothing, ReducedDynamics} = nothing,
        master_modes_derivatives::Union{Nothing, AbstractArray{ComplexF64, 3}} = nothing,
        left_modes_derivatives::Union{Nothing, AbstractArray{ComplexF64, 3}} = nothing,
        conjugate_permutation::Union{Nothing, AbstractVector{Int}} = nothing,
        validate_mset::Bool = true,
        show_progress::Bool = true,
        benchmark_dir::Union{Nothing, AbstractString} = nothing
) where {ORD, ORDP1, N_NL, N_EXT, LT, MT, NVAR, ROM}
    @assert NVAR == ROM + N_EXT "Multiindex set has $NVAR variables but ROM + N_EXT = $(ROM + N_EXT)"
    # Every path into the solve lands here, so this is where the mset contract is
    # enforced. `parametrise` checks first and passes validate_mset = false.
    validate_mset && validate_multiindex_set(mset, NVAR, ROM;
        conjugate_permutation = conjugate_permutation)
    FOM = size(master_modes, 1)
    @assert size(master_modes, 2) == ROM "master_modes must have $ROM columns"
    T = ComplexF64

    linear_terms = model.linear_terms

    zero_vec = SVector{NVAR, Int}(ntuple(_ -> 0, Val(NVAR)))
    has_zero = length(mset) >= 1 && mset.exponents[1] == zero_vec
    unit_offset = has_zero ? 1 : 0

    # ── 1. Parametrisation and reduced-dynamics objects ───────────────────────
    if initial_W !== nothing && initial_R !== nothing
        W = initial_W
        R = initial_R
    else
        @assert ORD == 1 || master_modes_derivatives !== nothing """
          master_modes_derivatives must be provided for ORD > 1 systems.
          Supply a FOM × (ORD-1) × ROM array whose slice [:, k, r] = W^(k+1)[e_r].
          """
        W, R = create_parametrisation_method_objects(mset, ORD, FOM, ROM, N_EXT, T)
        _initialise_waveform!(W, R, master_modes, master_eigenvalues,
            master_modes_derivatives, unit_offset, model)
    end

    Λ = view(R.poly.coefficients, 1:NVAR, (unit_offset + 1):(unit_offset + NVAR))
    lambda_diag = [R.poly.coefficients[i, i + unit_offset] for i in 1:NVAR]

    # ── 2. Shared resources: sym is built before ml_cache so skip_bits are available ──
    # Only master linear monomials are pre-initialised from eigenvectors and never solved;
    # external linear monomials go through the same conjugate-symmetry logic as all other
    # non-master monomials and are pre-solved by _solve_external_directions!.
    linear_skip_set = Set(_linear_monomial_indices(mset)[1:ROM])
    lower_order = LowerOrderResources{NVAR, T}(mset, ORD, FOM)
    sparse_solver = _make_sparse_solver(MT, linear_terms, FOM, ROM)

    _conj_perm = conjugate_permutation !== nothing ?
                 SVector{NVAR, Int}(conjugate_permutation) : NoConjugatePermutation()

    if !(_conj_perm isa NoConjugatePermutation)
        @info """
          conjugate_permutation is active — the following assumptions must hold:
            1. Real-valued FOM: all matrices in model.linear_terms and all nonlinear/force \
          	 terms must have real-valued entries (eltype <: Real or purely-real complex).
            2. Each mode either comes in a complex conjugate pair with another mode, or is \
          	 self-paired  meaning it has a real eigenvalue and a real-valued mode shape.
            3. Eigenvalue conjugacy is necessary but NOT sufficient for paired modes; \
          	 the eigenvectors must satisfy master_modes[:, perm[r]] = conj(master_modes[:, r]).
            4. If external modes are present (N_EXT > 0): the same pairing rules apply \
          	 to the external eigenvalues, encoded in the NVAR-length permutation.
          Passing an incorrect permutation silently corrupts the parametrisation and reduced-dynamics.
          """
    end

    sym = if _conj_perm isa NoConjugatePermutation
        _build_conjugate_symmetry(_conj_perm, linear_skip_set, length(mset))
    else
        _build_conjugate_symmetry(_conj_perm, linear_skip_set, mset,
            lower_order.multiindex_dict)
    end

    ml_cache = build_multilinear_terms_cache(model, W, sym.skip_bits)
    buffers = CohomologicalBuffers(T, MT, FOM, ROM)

    # ── 3. Φ_ext-independent operators ───────────────────────────────────────
    orthogonality_J_coeffs = precompute_orthogonality_operator_coefficients(
        linear_terms, left_eigenmodes, left_modes_derivatives
    )
    # Right master-mode order-blocks: the linear master monomials of W hold all
    # ORD derivative blocks (filled from the eigenvectors at initialisation).
    right_master_blocks = view(W.poly.coefficients, :, :,
        (unit_offset + 1):(unit_offset + ROM))
    Λ_master = view(R.poly.coefficients, 1:ROM, (unit_offset + 1):(unit_offset + ROM))
    invariance_C_coeffs, D_master_steps = precompute_master_column_polynomials(
        linear_terms, master_modes, Λ_master
    )

    # ── 4. Solve external linear monomials via partial contexts ──────────────
    if initial_W === nothing || initial_R === nothing
        # Φ_ext is unknown while the external directions are themselves being solved, so the
        # partial context carries zeros in its place.  For a non-diagonal (upper-triangular)
        # external block the directions couple, and the ones already solved are *known* data
        # that must be fed back — that is what `external_directions` carries.
        #
        # `blank` is the direction currently being solved.  Its own external column must be
        # forced to zero: the recurrences propagate Φ_ext,j into column e through Λ_ext[j, e]
        # for j < e, but that coupling is *already* delivered to the right-hand side by the
        # e_dyn[j]·E_j(s) terms, so leaving it in column e would count it twice.  The
        # remaining E_j, j > e, are weighted by e_dyn[j] = Λ_ext[j, e], which sits strictly
        # below the diagonal and therefore vanishes — that is exactly upper-triangularity.
        known_directions = zeros(T, FOM, N_EXT)
        build_partial_ctx = function (external_directions, blank::Int)
            partial_E_coeffs = precompute_external_column_polynomials(
                linear_terms, external_directions, Λ, D_master_steps
            )
            partial_orth_C_coeffs,
            partial_orth_E_coeffs = precompute_orthogonality_column_polynomials(
                orthogonality_J_coeffs, right_master_blocks, external_directions, Λ
            )
            if 1 <= blank <= N_EXT
                fill!(partial_E_coeffs[blank], zero(T))
                for r in 1:ROM
                    partial_orth_E_coeffs[r][:, blank] .= zero(T)
                end
            end
            return _build_context(
                linear_terms, hcat(master_modes, external_directions), lambda_diag,
                InvarianceOperators{T}(invariance_C_coeffs, partial_E_coeffs),
                OrthogonalityOperators{T}(orthogonality_J_coeffs,
                    partial_orth_C_coeffs, partial_orth_E_coeffs),
                resonance_set, linear_skip_set,
                lower_order, buffers, sparse_solver
            )
        end

        # Harmonic forcing (±iΩ) and every other diagonal external block need no feedback:
        # one Φ_ext = 0 context serves all directions, exactly as before.
        coupled_external = !isdiag(view(Λ, (ROM + 1):NVAR, (ROM + 1):NVAR))
        shared_partial_ctx = coupled_external ? nothing :
                             build_partial_ctx(known_directions, 0)
        partial_ctx_for = function (e)
            coupled_external || return shared_partial_ctx
            for k in 1:(e - 1)
                known_directions[:, k] .= view(
                    W.poly.coefficients, :, 1, ROM + k + unit_offset)
            end
            return build_partial_ctx(known_directions, e)
        end

        _solve_external_directions!(
            W, R, partial_ctx_for, model, ml_cache, sym, N_EXT, ROM, unit_offset)
    else
        # initial values provided: external directions are already in W; mark them done
        # so the main loop does not overwrite them.
        for e in 1:N_EXT
            @inbounds sym.skip_bits[ROM + e + unit_offset] = true
        end
    end

    # ── 5. Full eigenmodes and operators ──────────────────────────────────────
    external_directions = zeros(T, FOM, N_EXT)
    for e in 1:N_EXT
        external_directions[:, e] .= W.poly.coefficients[:, 1, ROM + e + unit_offset]
    end
    generalised_eigenmodes = hcat(master_modes, external_directions)

    invariance_E_coeffs = precompute_external_column_polynomials(
        linear_terms, external_directions, Λ, D_master_steps
    )
    orthogonality_C_coeffs,
    orthogonality_E_coeffs = precompute_orthogonality_column_polynomials(
        orthogonality_J_coeffs, right_master_blocks, external_directions, Λ
    )

    # ── 6. Full context and main solve ────────────────────────────────────────
    ctx = _build_context(
        linear_terms, generalised_eigenmodes, lambda_diag,
        InvarianceOperators{T}(invariance_C_coeffs, invariance_E_coeffs),
        OrthogonalityOperators{T}(orthogonality_J_coeffs,
            orthogonality_C_coeffs, orthogonality_E_coeffs),
        resonance_set, linear_skip_set,
        lower_order, buffers, sparse_solver
    )

    # ── 7. Main solve ─────────────────────────────────────────────────────────
    if benchmark_dir !== nothing
        solve_cohomological_equations_benchmarked!(W, R, ctx, sym, model, ml_cache;
            benchmark_dir, show_progress)
    else
        solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache; show_progress)
    end

    return W, R
end
