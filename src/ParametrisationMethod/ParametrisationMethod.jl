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

using Printf: @printf, @sprintf
using SparseArrays: SparseMatrixCSC
using StaticArrays: SVector

using ..Multiindices: MultiindexSet, all_multiindices_up_to
using ..FullOrderModel: NthOrderModel, _term_label
using ..SpectralDecomposition: SpectralData, master_eigenvalues,
                               master_conjugate_permutation
using ..Resonance: ResonanceSet, ResonanceConfig, build_resonance_set
using ..CohomologicalEquations: solve_cohomological_problem,
                                CohomologicalSolverConfig,
                                CohomologicalCheckpoint

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

export Parametrisation,
       ReducedDynamics, create_parametrisation_method_objects,
       compute_higher_derivative_coefficients!,
       restrict_ReducedDynamics_to_degree, restrict_Parametrisation_to_degree,
       validate_multiindex_set,
       parametrise, build_multiindex_set, print_setup,
       CohomologicalSolverConfig, CohomologicalCheckpoint

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

# ==================== Setup banner ====================

# λ as `a ± bi` when it is half of a conjugate pair, `a + bi` on its own otherwise.
function _eigenvalue_entries(λ, perm)
    seen = falses(length(λ))
    parts = String[]
    for r in eachindex(λ)
        seen[r] && continue
        seen[r] = true
        partner = perm === nothing ? r : perm[r]
        if partner != r
            seen[partner] = true
            push!(parts, @sprintf("%.3e ± %.3ei", real(λ[r]), abs(imag(λ[r]))))
        else
            push!(parts,
                @sprintf("%.3e %s %.3ei", real(λ[r]), imag(λ[r]) < 0 ? "-" : "+",
                    abs(imag(λ[r]))))
        end
    end
    return parts
end

function _resonance_summary(c::ResonanceConfig)
    s = string(c.style)
    c.tol === nothing || (s *= ",  tol = $(c.tol)")
    c.tol_relative === nothing || (s *= ",  tol_relative = $(c.tol_relative)")
    c.outer_targets && (s *= ",  outer targets on")
    s
end
_resonance_summary(rs::ResonanceSet) = "supplied — " * sprint(show, rs)

# Rows after the first of a multi-line block line up under the key column.
function _print_block(io::IO, key::AbstractString, entries)
    for (i, entry) in enumerate(entries)
        println(io, i == 1 ? key : " "^length(key), entry)
    end
end

"""
	print_setup(io, model, spectral, mset, resonance)

Print a summary of what [`parametrise`](@ref) is about to solve: model size and layout,
nonlinear terms, external system, reduced dimensions, master eigenvalues, the monomial
count, the resonance policy and the conjugate involution.

A normal function, so a caller who wants a different banner defines a method rather than
editing `parametrise`. It prints **only what the arguments genuinely carry**: `parametrise`
receives an `NthOrderModel`, and every backend's model is that same type, so there is
nothing backend-specific to dispatch on here. Backends with richer information print their
own summary before calling — they already have `show` methods for it.

Called by `parametrise` only when the output is going somewhere interactive; see the
`verbose` / `setup_io` keywords there.
"""
function print_setup(io::IO,
        model::NthOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT},
        spectral::SpectralData{ORD, ROM},
        mset::MultiindexSet{NVAR},
        resonance) where {ORD, ORDP1, N_NL, N_EXT, T, MT, NVAR, ROM}
    λ = master_eigenvalues(spectral)
    perm = master_conjugate_permutation(spectral)
    layout = MT <: SparseMatrixCSC ? "sparse" : "dense"

    println(io, repeat("=", 70))
    println(io, "MORFE parametrisation")
    println(io, repeat("-", 70))
    @printf(io, "  model       : FOM = %d,  ORD = %d,  %s (%s)\n",
        model.n_fom, ORD, layout, MT)
    if N_NL > 0
        _print_block(io, "  nonlinear   : ",
            ("$(_term_label(t))  (deg $(t.deg))" for t in model.nonlinear_terms))
    end
    if model.external_system !== nothing
        ext = _eigenvalue_entries(model.external_system.eigenvalues, nothing)
        @printf(io, "  external    : N_EXT = %d   (λ = %s)\n", N_EXT, join(ext, ", "))
    end
    @printf(io, "  reduced     : ROM = %d,  NVAR = %d\n", ROM, NVAR)
    _print_block(io, "  masters     : ", _eigenvalue_entries(λ, perm))
    # GrLex order puts the highest total degree last, so this is a lookup, not a scan.
    @printf(io, "  expansion   : total degree ≤ %d   →   %d monomials\n",
        sum(mset.exponents[end]), length(mset))
    println(io, "  resonance   : ", _resonance_summary(resonance))
    if perm !== nothing
        println(io, "  conjugate   : master ", perm,
            N_EXT > 0 ? "  (+ external, derived)" : "")
    end
    println(io, repeat("=", 70))
    return nothing
end

# ==================== The unified entry point ====================

"""
	parametrise(model, spectral::SpectralData, expansion_order; resonance, …) -> (W, R)

Compute the invariant manifold and its reduced dynamics. **This is the entry point.**

Two positional arguments carry everything the reduction needs — the full-order model and
the spectral data — and the third says how far to expand:

```julia
(; model, spectral) = build_model(case)    # or SpectralData(model, spectrum; master = …)
W, R = parametrise(model, spectral, 5)     # total degree ≤ 5
W, R = parametrise(model, spectral, mset)  # a monomial set you built
```

## Arguments

- `model::NthOrderModel` — full-order model; supplies the linear operators, the nonlinear
  terms, and the external system.
- `spectral::SpectralData` — master eigenvalues and their right/left blocks, the outer
  eigenvalues resonance detection reads, and the master-block conjugate involution. Build
  it with `SpectralData(model, spectrum; master = …)`.
- `expansion_order` — an `Integer` (total-degree truncation) or a `MultiindexSet` used as
  given. Dispatch happens in [`build_multiindex_set`](@ref), so a new expansion policy is
  a new method there and never a change here.

## Keyword arguments

- `resonance::Union{ResonanceConfig, ResonanceSet} = ResonanceConfig()` — either the
  policy ([`ResonanceConfig`](@ref) gathers style, tolerances and the outer-target
  settings) or a `ResonanceSet` you built yourself, used verbatim.
- `conjugate_permutation = :from_spectral` — by default the master-block involution from
  `spectral`, extended over the external variables using the model's external system.
  Pass an explicit `NVAR`-length vector to override, or `nothing` to disable conjugate
  symmetry for this solve.
- `validate_mset::Bool = true` — check the monomial set against the five-clause contract
  before solving. The solve is then told to skip its own check, so the set is walked once.
- `show_progress::Bool = true`.
- `verbose::Bool = true`, `setup_io::IO = stderr` — print the [`print_setup`](@ref) banner.
  On the default `stderr` it is gated on `stderr isa Base.TTY`, exactly as the progress
  reporter is, so redirected output and test logs stay as they were. An explicitly passed
  `setup_io` is always written to — a caller who names a destination asked for the output.

## Returns

`(W, R)` — the solved [`Parametrisation`](@ref) and [`ReducedDynamics`](@ref).
"""
function parametrise(
        model::NthOrderModel{ORD, ORDP1, N_NL, N_EXT, LT, MT},
        spectral::SpectralData{ORD, ROM},
        expansion_order;
        resonance::Union{ResonanceConfig, ResonanceSet} = ResonanceConfig(),
        conjugate_permutation = :from_spectral,
        validate_mset::Bool = true,
        show_progress::Bool = true,
        verbose::Bool = true,
        setup_io::IO = stderr,
        solver_config::CohomologicalSolverConfig = CohomologicalSolverConfig(),
        checkpoint::Union{Nothing, CohomologicalCheckpoint} = nothing
) where {ORD, ORDP1, N_NL, N_EXT, LT, MT, ROM}
    NVAR = ROM + N_EXT

    # Each step is a separate, independently dispatchable function.
    mset = build_multiindex_set(expansion_order, NVAR)

    # Validated here rather than only in the solve, so a malformed set is rejected before
    # the resonance-set construction. The permutation is needed for the closure clause,
    # so it is resolved first.
    if validate_mset
        perm_for_check = conjugate_permutation === :from_spectral ?
                         master_conjugate_permutation(spectral) : conjugate_permutation
        validate_multiindex_set(mset, NVAR, ROM;
            conjugate_permutation = perm_for_check === nothing ? nothing :
                                    _pad_permutation(perm_for_check, NVAR))
    end

    # Decided once, before anything is formatted: the whole banner — every `@sprintf`, every
    # term label — is built inside this branch, because interpolation boxes what it captures
    # even when the result is thrown away. Printed before the resonance set is built so the
    # banner precedes any tolerance `@info` it emits.
    _setup_output_enabled(verbose, setup_io) &&
        print_setup(setup_io, model, spectral, mset, resonance)

    resonance_set = resonance isa ResonanceSet ? resonance :
                    build_resonance_set(model, mset, spectral, resonance)

    return solve_cohomological_problem(model, mset, spectral, resonance_set;
        conjugate_permutation = conjugate_permutation,
        validate_mset = false,   # already checked above; don't walk the set twice
        show_progress = show_progress,
        solver_config, checkpoint)
end

# Same policy as `_make_progress`: on the default `stderr`, print only when it is a TTY, so
# CI logs and redirected runs are unchanged. An explicitly supplied destination is always
# written to — naming one is itself the request for output.
function _setup_output_enabled(verbose::Bool, io::IO)
    verbose && (io !== stderr || stderr isa Base.TTY)
end

# The mset contract's conjugate-closure clause wants an NVAR-length permutation, while
# `SpectralData` stores the ROM-length master block. Extending it with the identity on the
# external variables is enough for the closure check: it is a permutation of 1:NVAR that
# agrees with the real one on the master block, and the external block is checked
# separately by the solve against the external system itself.
function _pad_permutation(perm::AbstractVector{Int}, nvar::Int)
    length(perm) == nvar && return collect(Int, perm)
    return vcat(collect(Int, perm), collect((length(perm) + 1):nvar))
end

end # module
