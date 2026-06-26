module ConvenienceMethods
using ..Resonance
using ..FullOrderModel
using ..Eigenproblems
using ..Multiindices

export parametrize
"""
Script for the definitions of high level entry methods for a simple usabilty of the MORFE.jl package
"""

"""
    parametrize(model, order, eigenproblem; resonance, resonance_tol, conjugacy_map) -> (W, R)

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
function parametrize(
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

    # Views into the eigenproblem arrays — no allocation for the matrices
    # eigenproblem.eigenmodes is FOM × ORD × n_eigs; physical slice is [:, 1, :]
    master_modes = @view eigenproblem.eigenmodes[:, 1, master_mask]   # FOM × ROM
    left_eigenmodes = @view eigenproblem.left_eigenmodes[:, master_mask] # FOM × ROM

    # SVector is required by the type signature
    master_eigs_vec = eigenproblem.eigenvalues[master_mask]
    ROM = length(master_eigs_vec)
    master_eigenvalues = SVector{ROM, ComplexF64}(master_eigs_vec)

    # For ORD > 1: derivatives live in higher slices [:, 2:end, master_mask]
    master_modes_derivatives = ORD > 1 ?
                               @view(eigenproblem.eigenmodes[:, 2:end, master_mask]) :   # FOM × (ORD-1) × ROM
                               nothing

    # Generate MultiindexSet
    @assert order > 0 "order must be an integer bigger than zero"
    mset = all_multiindices_up_to(NVAR, order; min_degree = 1)

    # Generate ResonanceSet
    resonance_set = resonance isa ResonanceSet ? resonance :
                    _build_resonance_set(model, resonance, mset,
        eigenproblem, resonance_tol, conjugacy_map)

    # Solve cohomological equation
    W,
    R = solve_cohomological_problem(
        model, mset,
        master_eigenvalues,
        master_modes, left_eigenmodes,
        resonance_set;
        master_modes_derivatives = master_modes_derivatives
    )

    return W, R
end

"""
    _build_resonance_set

Helper function that defines the resonance_set dependent on the parametrization style.
Accepted styles:
- `:graph`
- `:complex_normal_form`
- `:real_normal_form`
"""
function _build_resonance_set(
        model::NDOrderModel,
        style::Symbol,
        mset::MultiindexSet,
        ep::Eigenproblem,
        tol::Float64;
        conjugacy_map = nothing
)
    master_mask = eigenproblem.master_modes
    outer_mask = .!eigenproblem.master_modes
    master_eigenvalues = eigenproblem.eigenvalues[master_mask]
    outer_eigenvalues = eigenproblem.eigenvalues[outer_mask]
    external_eigenvalues = model.external_system === nothing ? ComplexF64[] :
                           model.external_system.eigenvalues

    if style === :graph
        return resonance_set_from_graph_style(
            mset, master_eigenvalues, external_eigenvalues, outer_eigenvalues, tol)

    elseif style === :complex_normal_form
        return resonance_set_from_complex_normal_form_style(
            mset, master_eigenvalues, tol;
            external_eigenvalues, outer_eigenvalues)

    elseif style === :real_normal_form
        @assert !isnothing(conjugacy_map) ":real_normal_form requires conjugacy_map to be set"
        return resonance_set_from_real_normal_form_style(
            mset, master_eigenvalues, conjugacy_map, tol;
            external_eigenvalues, outer_eigenvalues)
    else
        throw(ArgumentError("Unknown resonance_style :$style. Choose :graph or :complex_normal_form"))
    end
    #TODO resonance_set_from_condition_number_estimate
end

end