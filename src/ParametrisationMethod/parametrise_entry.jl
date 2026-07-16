# High-level `parametrise(model, order, eigenproblem; …)` entry point.
#
# The generic function `parametrise` is owned by `ParametrisationMethod`
# (declared there). This method body lives in a separate file because it calls
# `solve_cohomological_problem` from `CohomologicalEquations`, which is included
# *after* `ParametrisationMethod` in `src/MORFE.jl`. This file is therefore
# included at MORFE top-level scope after the `using .CohomologicalEquations`
# re-export block, so the bare names below resolve.

using StaticArrays: SVector

"""
	parametrise(model, order, eigenproblem; resonance, resonance_tol, conjugacy_map) -> (W, R)

Convenience entry point for the parametrisation method. Assembles all required inputs
from a solved `Eigenproblem` and calls `solve_cohomological_problem` internally.

## Arguments

- `model::NDOrderModel`: full-order model providing linear and nonlinear terms.
- `order::Int`: polynomial order of the parametrisation (must be > 0).
- `eigenproblem::Eigenproblem`: solved eigenproblem with master modes selected via one
  of the `select_master_modes_*` functions.

## Keyword Arguments

- `resonance::Union{Symbol, ResonanceSet} = :graph`: resonance style. Either a
  pre-built `ResonanceSet` (passed through unchanged) or one of the symbols:
  - `:graph` — graph style; every monomial of degree ≥ 2 is resonant with all master
	modes.
  - `:complex_normal_form` — inner resonances determined by eigenvalue proximity.
  - `:real_normal_form` — like `:complex_normal_form` but conjugate pairs share the
	resonance flag. Requires `conjugacy_map` to be set.
- `resonance_tol::Float64 = 1e-2`: tolerance used in eigenvalue-proximity resonance
  checks. Only used when `resonance` is a `Symbol`.
- `conjugacy_map::Union{Nothing, Vector{Int}} = nothing`: local conjugacy map of length
  `ROM + n_outer`; required when `resonance = :real_normal_form`, ignored otherwise.
- `mset::Union{Nothing, MultiindexSet} = nothing`: custom multiindex set (e.g. an
  anisotropic z-total × θ-box set for parametric ROMs). Must have `NVAR = ROM + N_EXT`
  variables, minimum total degree ≥ 1, contain every unit multiindex, and be
  **downward closed** (every divisor of a member is a member) as well as closed
  under the conjugate permutation when one is used — the graded solve relies on
  both. `nothing` → `all_multiindices_up_to(NVAR, order; min_degree = 1)`.
- `conjugate_permutation::Union{Nothing, Vector{Int}} = nothing`: NVAR-length
  permutation pairing conjugate coordinates (self-paired entries for real modes);
  passed through to the cohomological solve to enforce conjugate symmetry.
- `external_eigenvalues::Union{Nothing, Vector{ComplexF64}} = nothing`: override for
  the external eigenvalues used in resonance detection (default: taken from
  `model.external_system`).
- `master_modes_derivatives = nothing`, `left_modes_derivatives = nothing`:
  explicit `(FOM, ORD-1, ROM)` derivative/order blocks. Needed when the
  `Eigenproblem` was solved on a *lower-order* operator than `model` (e.g. an
  augmented `(K, C, M, 0)` ORD-3 model with a second-order structural
  eigenproblem): the internal slices then don't match `ORD`, and callers supply
  blocks built from the eigenpairs (right: `λ^{k-1}·Y[:, 2, r]`; left: via
  `left_eigenmode_orders_from_slice(model.linear_terms, …)`). Default `nothing`
  → sliced from the `Eigenproblem` storage as before.

## Returns

`(W, R)` — the solved [`Parametrisation`](@ref) and [`ReducedDynamics`](@ref).

"""
function ParametrisationMethod.parametrise(
	model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, LT, MT},
	order::Int,
	eigenproblem::Eigenproblem;
	resonance::Union{Symbol, ResonanceSet} = :graph,
	resonance_tol::Float64 = 1e-2,
	conjugacy_map = nothing,
	mset::Union{Nothing, MultiindexSet} = nothing,
	conjugate_permutation::Union{Nothing, Vector{Int}} = nothing,
	external_eigenvalues::Union{Nothing, Vector{ComplexF64}} = nothing,
	master_modes_derivatives = nothing,
	left_modes_derivatives = nothing) where {ORD, ORDP1, N_NL, N_EXT, LT, MT}

	# Extract eigenproblem
	master_mask = eigenproblem.master_modes
	@assert !isnothing(master_mask) "master_modes not set on Eigenproblem"
	ROM = sum(master_mask)
	@assert ROM > 0 "No modes chosen as master modes in Eigenproblem"
	NVAR = ROM + N_EXT

	# Copies, not views: a Bool-mask view is not strided, so downstream
	# BLAS/sparse products would fall back to slow generic matmul.
	# eigenproblem.eigenmodes is FOM × ORD × n_eigs; physical slice is [:, 1, :]
	master_modes = eigenproblem.eigenmodes[:, 1, master_mask]   # FOM × ROM
	left_eigenmodes = eigenproblem.left_eigenmodes[:, master_mask] # FOM × ROM

	# SVector is required by the type signature
	master_eigs_vec = eigenproblem.eigenvalues[master_mask]
	ROM = length(master_eigs_vec)
	master_eigenvalues = SVector{ROM, ComplexF64}(master_eigs_vec)

	# For ORD > 1: derivatives live in higher slices [:, 2:end, master_mask].
	# An explicit kwarg overrides the slicing (lower-order Eigenproblem storage
	# feeding a higher-order model — see docstring).
	if master_modes_derivatives === nothing
		master_modes_derivatives = ORD > 1 ?
								   @view(eigenproblem.eigenmodes[:, 2:end, master_mask]) :   # FOM × (ORD-1) × ROM
								   nothing
	end

	# For ORD > 1: lower-order left eigenvector blocks φ_1 … φ_{ORD-1} feed the
	# orthogonality row operators directly (no eigenvalue folding).
	if left_modes_derivatives === nothing
		left_modes_derivatives = if ORD > 1
			@assert eigenproblem.left_eigenmodes_orders !== nothing """
			Eigenproblem stores only the physical-space left eigenmode slice, but
			ORD > 1 orthogonality solves need the full left eigenvector order-blocks.
			Use an eigensolver path that supplies them (solve_left returns FOM × ORD × n).
			"""
			@view(eigenproblem.left_eigenmodes_orders[:, 1:(ORD-1), master_mask])   # FOM × (ORD-1) × ROM
		else
			nothing
		end
	end

	# Multiindex set: default graded-total, or a validated custom set.
	@assert order > 0 "order must be an integer bigger than zero"
	if mset === nothing
		mset = all_multiindices_up_to(NVAR, order; min_degree = 1)
	else
		mset isa MultiindexSet{NVAR} || throw(ArgumentError(
			"custom mset has $(length(first(mset.exponents))) variables, " *
			"but the model requires NVAR = ROM + N_EXT = $NVAR"))
		sum(first(mset.exponents)) ≥ 1 || throw(ArgumentError(
			"custom mset must not contain the zero multiindex (min total degree ≥ 1)"))
		for i in 1:NVAR
			unit = [j == i ? 1 : 0 for j in 1:NVAR]
			find_in_set(mset, unit) === nothing && throw(ArgumentError(
				"custom mset is missing the unit multiindex e_$i; the linear " *
				"initialisation of the parametrisation requires all unit multiindices"))
		end
	end

	# Generate ResonanceSet
	resonance_set = resonance isa ResonanceSet ? resonance :
					build_resonance_set(model, resonance, mset,
		eigenproblem, resonance_tol, conjugacy_map;
		external_eigenvalues = external_eigenvalues)

	# Solve cohomological equation
	W,
	R = solve_cohomological_problem(
		model, mset,
		master_eigenvalues,
		master_modes, left_eigenmodes,
		resonance_set;
		master_modes_derivatives = master_modes_derivatives,
		left_modes_derivatives = left_modes_derivatives,
		conjugate_permutation = conjugate_permutation,
	)

	return W, R
end
