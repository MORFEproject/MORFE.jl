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

## Returns

`(W, R)` — the solved [`Parametrisation`](@ref) and [`ReducedDynamics`](@ref).

"""
function ParametrisationMethod.parametrise(
	model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, LT, MT},
	order::Int,
	eigenproblem::Eigenproblem;
	resonance::Union{Symbol, ResonanceSet} = :graph,
	resonance_tol::Float64 = 1e-2,
	conjugacy_map = nothing) where {ORD, ORDP1, N_NL, N_EXT, LT, MT}

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

	# For ORD > 1: derivatives live in higher slices [:, 2:end, master_mask]
	master_modes_derivatives = ORD > 1 ?
							   @view(eigenproblem.eigenmodes[:, 2:end, master_mask]) :   # FOM × (ORD-1) × ROM
							   nothing

	# For ORD > 1: lower-order left eigenvector blocks φ_1 … φ_{ORD-1} feed the
	# orthogonality row operators directly (no eigenvalue folding).
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

	# Generate MultiindexSet
	@assert order > 0 "order must be an integer bigger than zero"
	mset = all_multiindices_up_to(NVAR, order; min_degree = 1)

	# Generate ResonanceSet
	resonance_set = resonance isa ResonanceSet ? resonance :
					build_resonance_set(model, resonance, mset,
		eigenproblem, resonance_tol, conjugacy_map)

	# Solve cohomological equation
	W,
	R = solve_cohomological_problem(
		model, mset,
		master_eigenvalues,
		master_modes, left_eigenmodes,
		resonance_set;
		master_modes_derivatives = master_modes_derivatives,
		left_modes_derivatives = left_modes_derivatives,
	)

	return W, R
end
