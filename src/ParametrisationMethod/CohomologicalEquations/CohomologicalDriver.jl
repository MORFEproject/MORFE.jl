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
	master_modes_derivatives, unit_offset::Int, model,
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

"""
	_solve_external_directions!(W, R, partial_ctx, model, ml_cache, sym, N_EXT, ROM, unit_offset)

Solve the `N_EXT` external forcing directions using a *partial* context in which
the external generalised eigenvectors `Φ_ext` are set to zero (they are unknown at
this stage).  When conjugate-symmetry is active, secondary external monomials are
skipped and filled from their primaries via `fill_conjugate_monomial!`.

All external linear monomials are marked in `sym.skip_bits` after this call so
the main `solve_cohomological_equations!` loop does not overwrite them.
"""
function _solve_external_directions!(
	W, R, partial_ctx, model, ml_cache,
	sym::ConjugateSymmetryData{NoConjugatePermutation}, N_EXT::Int, ROM::Int, unit_offset::Int,
)
	for e in 1:N_EXT
		idx = ROM + e + unit_offset
		solve_single_monomial!(W, R, idx, partial_ctx, model, ml_cache)
		@inbounds sym.skip_bits[idx] = true
	end
	return nothing
end

function _solve_external_directions!(
	W, R, partial_ctx, model, ml_cache,
	sym::ConjugateSymmetryData{<:SVector}, N_EXT::Int, ROM::Int, unit_offset::Int,
)
	N_EXT == 0 && return nothing
	ext_idx_set = Set(ROM + e + unit_offset for e in 1:N_EXT)
	for e in 1:N_EXT
		idx = ROM + e + unit_offset
		@inbounds sym.skip_bits[idx] && continue   # secondary: filled below
		solve_single_monomial!(W, R, idx, partial_ctx, model, ml_cache)
	end
	for (src, dst) in sym.primary_pairs
		src ∈ ext_idx_set || continue
		fill_conjugate_monomial!(W, R, dst, src, sym)
	end
	for e in 1:N_EXT
		@inbounds sym.skip_bits[ROM+e+unit_offset] = true
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
	::Type{<:SparseMatrixCSC}, linear_terms, FOM::Int, ROM::Int,
)
	L_template, L_mappings = precompute_sparse_L_template(linear_terms)
	return SparseLinearSolverState{ComplexF64}(L_template, L_mappings, FOM, ROM)
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
	sparse_solver,
) where {ORDP1, MT, T, NVAR}
	FOM = size(generalised_eigenmodes, 1)
	ORD = ORDP1 - 1
	LT = eltype(MT)
	return CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT}(
		linear_terms, generalised_eigenmodes, lambda_diag,
		inv_ops, orth_ops,
		resonance_set, linear_skip_set,
		lower_order, buffers, sparse_solver,
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
	show_progress::Bool = true,
	benchmark_dir::Union{Nothing, AbstractString} = nothing,
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

	Λ = view(R.poly.coefficients, 1:NVAR, (unit_offset+1):(unit_offset+NVAR))
	lambda_diag = [R.poly.coefficients[i, i+unit_offset] for i in 1:NVAR]

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
		linear_terms, left_eigenmodes, left_modes_derivatives,
	)
	# Right master-mode order-blocks: the linear master monomials of W hold all
	# ORD derivative blocks (filled from the eigenvectors at initialisation).
	right_master_blocks = view(W.poly.coefficients, :, :,
		(unit_offset+1):(unit_offset+ROM))
	Λ_master = view(R.poly.coefficients, 1:ROM, (unit_offset+1):(unit_offset+ROM))
	invariance_C_coeffs, D_master_steps = precompute_master_column_polynomials(
		linear_terms, master_modes, Λ_master,
	)

	# ── 4. Solve external linear monomials via partial context (Φ_ext = 0) ───
	if initial_W === nothing || initial_R === nothing
		partial_E_coeffs = precompute_external_column_polynomials(
			linear_terms, zeros(T, FOM, N_EXT), Λ, D_master_steps,
		)
		partial_eigenmodes = hcat(master_modes, zeros(T, FOM, N_EXT))
		partial_orth_C_coeffs, partial_orth_E_coeffs = precompute_orthogonality_column_polynomials(
			orthogonality_J_coeffs, right_master_blocks, zeros(T, FOM, N_EXT), Λ,
		)
		partial_ctx = _build_context(
			linear_terms, partial_eigenmodes, lambda_diag,
			InvarianceOperators{T}(invariance_C_coeffs, partial_E_coeffs),
			OrthogonalityOperators{T}(orthogonality_J_coeffs,
				partial_orth_C_coeffs, partial_orth_E_coeffs),
			resonance_set, linear_skip_set,
			lower_order, buffers, sparse_solver,
		)
		_solve_external_directions!(W, R, partial_ctx, model, ml_cache, sym, N_EXT, ROM, unit_offset)
	else
		# initial values provided: external directions are already in W; mark them done
		# so the main loop does not overwrite them.
		for e in 1:N_EXT
			@inbounds sym.skip_bits[ROM+e+unit_offset] = true
		end
	end

	# ── 5. Full eigenmodes and operators ──────────────────────────────────────
	external_directions = zeros(T, FOM, N_EXT)
	for e in 1:N_EXT
		external_directions[:, e] .= W.poly.coefficients[:, 1, ROM+e+unit_offset]
	end
	generalised_eigenmodes = hcat(master_modes, external_directions)

	invariance_E_coeffs = precompute_external_column_polynomials(
		linear_terms, external_directions, Λ, D_master_steps,
	)
	orthogonality_C_coeffs, orthogonality_E_coeffs = precompute_orthogonality_column_polynomials(
		orthogonality_J_coeffs, right_master_blocks, external_directions, Λ,
	)

	# ── 6. Full context and main solve ────────────────────────────────────────
	ctx = _build_context(
		linear_terms, generalised_eigenmodes, lambda_diag,
		InvarianceOperators{T}(invariance_C_coeffs, invariance_E_coeffs),
		OrthogonalityOperators{T}(orthogonality_J_coeffs,
			orthogonality_C_coeffs, orthogonality_E_coeffs),
		resonance_set, linear_skip_set,
		lower_order, buffers, sparse_solver,
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
