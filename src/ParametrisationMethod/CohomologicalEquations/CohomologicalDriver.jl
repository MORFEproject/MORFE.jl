# =============================================================================
# Private driver helpers
# =============================================================================

# Initialise W and R linear monomials from spectral data and embed external dynamics.
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
                W.poly.coefficients[:, k, idx_er] .= view(master_modes_derivatives, :, k - 1, r)
            end
        end
        R.poly.coefficients[r, idx_er] = master_eigenvalues[r]
    end
    if model.external_system !== nothing
        _embed_external_dynamics!(R, model.external_system.first_order_dynamics, multiindex_set(W))
    end
    return nothing
end

# Solve the external linear monomials using a partial context (Φ_ext = 0).
function _solve_external_directions!(
        W, R, partial_ctx, model, ml_cache, N_EXT::Int, ROM::Int, unit_offset::Int
)
    for e in 1:N_EXT
        solve_single_monomial!(W, R, ROM + e + unit_offset, partial_ctx, model, ml_cache)
    end
    return nothing
end

# Dispatch helper: sparse path creates SparseLinearSolverState, dense path returns nothing.
_make_sparse_solver(::Type{<:AbstractMatrix}, _, ::Int, ::Int) = nothing
function _make_sparse_solver(
        ::Type{<:SparseMatrixCSC}, linear_terms, FOM::Int, ROM::Int
)
    L_template, L_mappings = precompute_sparse_L_template(linear_terms)
    return SparseLinearSolverState{ComplexF64}(L_template, L_mappings, FOM, ROM)
end

# Assemble a CohomologicalContext from operator data and shared resources.
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
        master_modes_derivatives = nothing
    ) -> (W, R)

High-level driver that assembles a [`CohomologicalContext`](@ref) from raw
spectral data and solves the full set of cohomological equations.

External eigenvalues are read directly from `model.external_system.eigenvalues`
(or treated as absent when `model.external_system === nothing`).  The
linear-operator tuple is read from `model.linear_terms`.

## Steps

1. Create (or reuse) [`Parametrisation`](@ref) `W` and [`ReducedDynamics`](@ref) `R`
   and initialise master-mode linear monomials.
2. Build shared resources (buffers, lower-order coupling data, sparse solver state).
3. Solve the linear cohomological equations for each external forcing direction via
   a partial context in which the external columns of `generalised_right_eigenmodes`
   are set to zero.
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
        master_modes_derivatives::Union{Nothing, AbstractArray{ComplexF64, 3}} = nothing
) where {ORD, ORDP1, N_NL, N_EXT, LT, MT, NVAR, ROM}

    @assert NVAR == ROM + N_EXT "Multiindex set has $NVAR variables but ROM + N_EXT = $(ROM + N_EXT)"
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

    # ── 2. Shared resources (allocated once, reused in both contexts) ─────────
    ml_cache = build_multilinear_terms_cache(model, W)
    linear_skip_set = Set(_linear_monomial_indices(mset))
    lower_order = LowerOrderResources{NVAR, T}(mset, ORD, FOM)
    buffers = CohomologicalBuffers{T}(FOM, ROM)
    sparse_solver = _make_sparse_solver(MT, linear_terms, FOM, ROM)

    # ── 3. Φ_ext-independent operators ───────────────────────────────────────
    orthogonality_J_coeffs = precompute_orthogonality_operator_coefficients(
        linear_terms, left_eigenmodes, master_eigenvalues
    )
    Λ_master = view(R.poly.coefficients, 1:ROM, (unit_offset + 1):(unit_offset + ROM))
    invariance_C_coeffs, D_master_steps = precompute_master_column_polynomials(
        linear_terms, master_modes, Λ_master
    )

    # ── 4. Solve external linear monomials via partial context (Φ_ext = 0) ───
    if initial_W === nothing || initial_R === nothing
        partial_E_coeffs = precompute_external_column_polynomials(
            linear_terms, zeros(T, FOM, N_EXT), Λ, D_master_steps
        )
        partial_eigenmodes = hcat(master_modes, zeros(T, FOM, N_EXT))
        partial_orth_C_coeffs, partial_orth_E_coeffs = precompute_orthogonality_column_polynomials(
            orthogonality_J_coeffs, partial_eigenmodes, Λ
        )
        partial_ctx = _build_context(
            linear_terms, partial_eigenmodes, lambda_diag,
            InvarianceOperators{T}(invariance_C_coeffs, partial_E_coeffs),
            OrthogonalityOperators{T}(orthogonality_J_coeffs,
                                      partial_orth_C_coeffs, partial_orth_E_coeffs),
            resonance_set, linear_skip_set,
            lower_order, buffers, sparse_solver
        )
        _solve_external_directions!(W, R, partial_ctx, model, ml_cache, N_EXT, ROM, unit_offset)
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
    orthogonality_C_coeffs, orthogonality_E_coeffs = precompute_orthogonality_column_polynomials(
        orthogonality_J_coeffs, generalised_eigenmodes, Λ
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

    solve_cohomological_equations!(W, R, ctx, model, ml_cache)

    return W, R
end
