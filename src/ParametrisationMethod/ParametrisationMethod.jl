"""
Module `ParametrisationMethod` — the user-facing entry point to the DPIM
parametrisation.

Owns [`parametrise`](@ref), the high-level driver that turns a full-order model plus
spectral data into a solved invariant manifold `(W, R)`, and the individual pipeline
steps it is built from — each a separate function so that a new policy is a new method
rather than a new branch:

| Step | Function | Dispatches on |
|------|----------|---------------|
| build the monomial set | [`build_multiindex_set`](@ref) | the expansion order |
| build the resonance set | `build_resonance_set` | the resonance style |
| solve | `solve_cohomological_problem` | dense vs sparse model |

The coefficient containers themselves (`Parametrisation`, `ReducedDynamics`, …) live in
[`ParametrisationObjects`](@ref) and are re-exported here, so this module remains the
single namespace users need.

## Load order

This module is included **after** `CohomologicalEquations`, because `parametrise` calls
`solve_cohomological_problem`.  That is why the containers live in a separate module:
`CohomologicalEquations` needs *them*, and they must therefore load first.  Previously
this ordering was worked around by defining `parametrise`'s method in a bare,
module-less `parametrise_entry.jl` included at `MORFE` top level; that file is gone.
"""
module ParametrisationMethod

using StaticArrays: SVector

using ..Multiindices: MultiindexSet, all_multiindices_up_to
using ..FullOrderModel: NDOrderModel
using ..Eigenproblems: Eigenproblem
using ..Resonance: ResonanceSet, build_resonance_set
using ..CohomologicalEquations: solve_cohomological_problem

# Re-exported wholesale so `ParametrisationMethod` stays the one namespace users (and
# `ext/MORFEBifurcationKitExt.jl`, which reaches for
# `MORFE.ParametrisationMethod: ReducedDynamics`) need to know about.
using ..ParametrisationObjects: Parametrisation, ReducedDynamics,
                                create_parametrisation_method_objects,
                                compute_higher_derivative_coefficients!,
                                restrict_ReducedDynamics_to_degree,
                                restrict_Parametrisation_to_degree,
                                validate_multiindex_set,
# Not exported (they never were), but reachable as
# `MORFE.ParametrisationMethod.coefficients` — which
# the test suite relies on. Keep them bound here.
                                coefficients, multiindex_set

export Parametrisation, ReducedDynamics, create_parametrisation_method_objects,
       compute_higher_derivative_coefficients!,
       restrict_ReducedDynamics_to_degree, restrict_Parametrisation_to_degree,
       validate_multiindex_set,
       parametrise, build_multiindex_set

# ==================== Expansion order → multiindex set ====================

"""
	build_multiindex_set(expansion_order, nvar) -> MultiindexSet

Turn an expansion order into the monomial set the solve will run over.  This is the
dispatch seam for expansion policies: [`parametrise`](@ref) leaves its third argument
untyped and delegates here, so a new policy is one new method and `parametrise` never
changes.

Two policies ship today:

- `expansion_order::Integer` — total-degree truncation,
  `all_multiindices_up_to(nvar, order; min_degree = 1)`.
- `expansion_order::MultiindexSet` — a set the caller built (e.g. the anisotropic
  z-total × θ-box sets used by parametric ROMs), used exactly as given.

Anything else raises an `ArgumentError` naming what is accepted, rather than a bare
`MethodError`.

This function **builds only; it does not validate**.  `parametrise` runs
[`validate_multiindex_set`](@ref) once on the result, whatever its source, then tells
the solve to skip its own check so the set is walked exactly once.  Validation needs
`ROM` and the conjugate permutation, neither of which an expansion policy has any
business knowing.
"""
function build_multiindex_set(expansion_order::Integer, nvar::Int)
    @assert expansion_order>0 "expansion_order must be an integer bigger than zero"
    return all_multiindices_up_to(nvar, Int(expansion_order); min_degree = 1)
end

build_multiindex_set(mset::MultiindexSet, ::Int) = mset

function build_multiindex_set(x, ::Int)
    throw(ArgumentError(
        "expansion_order must be an Integer (total-degree truncation) or a MultiindexSet; " *
        "got $(typeof(x))"))
end

# ==================== High-level entry point ====================

"""
	parametrise(model, expansion_order, eigenproblem; resonance, resonance_tol, conjugacy_map) -> (W, R)

Convenience entry point for the parametrisation method. Assembles all required inputs
from a solved `Eigenproblem` and calls `solve_cohomological_problem` internally.

## Arguments

- `model::NDOrderModel`: full-order model providing linear and nonlinear terms.
- `expansion_order`: either an `Integer` (total-degree truncation, must be > 0) or a
  `MultiindexSet` used as given — see [`build_multiindex_set`](@ref).
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
- `mset::Union{Nothing, MultiindexSet} = nothing`: **deprecated spelling** of a custom
  monomial set — pass the `MultiindexSet` as `expansion_order` instead. When given it
  overrides `expansion_order`. Must have `NVAR = ROM + N_EXT` variables, minimum total
  degree ≥ 1, contain every unit multiindex, and be **downward closed** (every divisor
  of a member is a member) as well as closed under the conjugate permutation when one is
  used — the graded solve relies on both. All of this is **enforced** by
  [`validate_multiindex_set`](@ref), which throws an `ArgumentError` naming the offending
  exponent.
- `validate_mset::Bool = true`: check the monomial set against the contract above before
  solving. Set `false` only when the set has already been validated, or when the check
  itself is too costly on a very large set; an invalid set then produces a silently
  wrong parametrisation rather than an error.
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
function parametrise(
        model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, LT, MT},
        expansion_order,
        eigenproblem::Eigenproblem;
        resonance::Union{Symbol, ResonanceSet} = :graph,
        resonance_tol::Float64 = 1e-2,
        conjugacy_map = nothing,
        mset::Union{Nothing, MultiindexSet} = nothing,
        conjugate_permutation::Union{Nothing, Vector{Int}} = nothing,
        validate_mset::Bool = true,
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
            @view(eigenproblem.left_eigenmodes_orders[:, 1:(ORD - 1), master_mask])   # FOM × (ORD-1) × ROM
        else
            nothing
        end
    end

    # Monomial set: from `expansion_order` via the dispatch seam, unless the
    # deprecated `mset` keyword overrides it.
    monomials = mset === nothing ? build_multiindex_set(expansion_order, NVAR) : mset

    # Check here rather than only in the solve, so a malformed set is rejected
    # before the eigenproblem slicing and resonance-set construction.
    if validate_mset
        validate_multiindex_set(monomials, NVAR, ROM;
            conjugate_permutation = conjugate_permutation)
    end

    # Generate ResonanceSet
    resonance_set = resonance isa ResonanceSet ? resonance :
                    build_resonance_set(model, resonance, monomials,
        eigenproblem, resonance_tol, conjugacy_map;
        external_eigenvalues = external_eigenvalues)

    # Solve cohomological equation
    W,
    R = solve_cohomological_problem(
        model, monomials,
        master_eigenvalues,
        master_modes, left_eigenmodes,
        resonance_set;
        master_modes_derivatives = master_modes_derivatives,
        left_modes_derivatives = left_modes_derivatives,
        conjugate_permutation = conjugate_permutation,
        validate_mset = false   # already checked above; don't walk the set twice
    )

    return W, R
end

end # module
