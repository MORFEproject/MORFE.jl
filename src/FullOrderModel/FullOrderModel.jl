"""
Module `FullOrderModel` — representation of high-dimensional nonlinear ODEs.

The central type is `NthOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT}`, which encodes
an `ORD`-th order system

	B_ORD ẋ^(ORD) + … + B₁ ẋ + B₀ x = F(x, ẋ, …, r)

where `r` satisfies an autonomous external system.  Linear terms are stored as an
`NTuple{ORDP1, MT}` of matrices; nonlinear terms as an `NTuple{N_NL, AbstractMultilinearMap}`.

Key functions: `linear_first_order_matrices` (produces the companion-form `(A, B)` pair
used by eigensolvers), `evaluate_nonlinear_terms!`.
"""
module FullOrderModel

using LinearAlgebra
using SparseArrays
using StaticArrays: SVector

using ..Polynomials: DensePolynomial
using ..MultilinearMaps: AbstractMultilinearMap, FEMMultilinearMap, MultilinearMap,
                         evaluate_term!, fem_elements, _definition_site
using ..ExternalSystems: ExternalSystem, external_basis

export NthOrderModel, linear_first_order_matrices,
       evaluate_nonlinear_terms!

abstract type AbstractFullOrderModel end

function _term_label(t)
    if t isa MultilinearMap
        # `_definition_site` rather than `only(methods(...))`: `f!` may legitimately carry
        # several methods, and this runs inside an `@info` that must never throw.
        return "MultilinearMap" * _definition_site(t.f!)
    else
        T_name = string(nameof(typeof(t)))
        try
            m = which(fem_elements, Tuple{typeof(t)})
            return "$T_name @ $(basename(String(m.file))):$(m.line)"
        catch
        end
        return T_name
    end
end

function _info_implicit_symmetry(nonlinear_terms)
    hits = [(i, t)
            for (i, t) in enumerate(nonlinear_terms)
            if any(x -> x > 1, t.multiindex) && t.fully_asymmetric === nothing]
    isempty(hits) && return
    lines = join(
        ["  · term $i: $(_term_label(t)), with multiindex=$(t.multiindex) and deg=$(t.deg)"
         for (i, t) in hits], "\n")
    @info "NthOrderModel: the following terms did not set `fully_asymmetric`\n" *
          "(f! is assumed symmetric within each derivative-order argument group):\n" *
          lines * "\n  Pass `fully_asymmetric=true` to any term where this does not hold."
end

"""
	_check_external_terms(nonlinear_terms)

Reject terms that read the external state in a model built without one.

Without this the mismatch survives construction and only surfaces mid-solve, as
`"Term expects external arguments but no external state provided"` from `evaluate_term!`.
"""
function _check_external_terms(nonlinear_terms)
    for (i, t) in enumerate(nonlinear_terms)
        t.multiplicity_external == 0 && continue
        throw(
            ArgumentError(
            "NthOrderModel: term $i ($(_term_label(t))) takes " *
            "$(t.multiplicity_external) external factor(s), but this " *
            "model was built without an external system.\nPass an " *
            "`ExternalSystem` (or the external dynamics polynomial) as " *
            "the third argument, or set multiplicity_external = 0.",
        ),
        )
    end
end

"""

	NthOrderModel{ORD, ORDP1, N_NL, T, MT} <: AbstractFullOrderModel

Representation of an ORD-th (ORDP1=ORD+1) order dynamical system of the form

	B_ORD x^(ORD) + ... + B_1 x^(1) + B_0 x = F(x^(ORD-1), …, x^(1), x, r, …, r)

where:
- x^(n) is the n-th derivative of x (x^(n) = d_t^n x)
- x^(0) = x is the state vector
- B_i are the coefficient matrices
- F is a multilinear polynomial function of the derivatives and the external state vector r
- The external state r satisfies its own first‑order dynamics r' = g(r)

# Generic type parameters

- `ORD` defines the order of the ODE.
- `ORDP1` is the number of linear terms (from 0 through ORD). It must satisfy ORDP1 == ORD+1.
- `N_NL` is the number of nonlinear terms in the tuple nonlinear_terms.
- `N_EXT` is the size of the external system.
- `T` is the numeric type.
- `MT` is the matrix type that forms the ORDP1-tuple of linear_terms.

# Fields

- `n_fom::Int` — dimension of the full‑order state vector `x`.
- `linear_terms::NTuple{ORDP1, MT}` — the linear coefficient matrices
  `(B_0, …, B_ORD)`, all of identical size.
- `nonlinear_terms::NTuple{N_NL, AbstractMultilinearMap{ORD}}` — the nonlinear
  contributions, each a [`MultilinearMap`](@ref) or [`FEMMultilinearMap`](@ref).
- `external_system::Union{Nothing, ExternalSystem{N_EXT}}` — the external dynamics,
  or `nothing` for an unforced model.
- `max_nl_degree::Int` — the largest combined degree over `nonlinear_terms`, cached
  at construction.  It bounds the polynomial order at which anything nonlinear can
  still contribute, and drives the progress indicator's work estimate.

# Representation

- Linear terms are stored as a tuple `(B_0, …, B_ORD)`
- Nonlinear terms are represented as a collection of `MultilinearMap`s

Each `MultilinearMap` defines:
- which derivatives are involved (via `multiindex`)
- how many times the external state appears as an argument (via `multiplicity_external`)
- the combined degree `deg = sum(multiindex) + multiplicity_external`

# Notes
- All matrices must have identical size.
- The nonlinear structure is stored in sparse form (only active terms).
"""
struct NthOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT <: AbstractMatrix{T}} <:
       AbstractFullOrderModel
    n_fom::Int
    linear_terms::NTuple{ORDP1, MT}
    nonlinear_terms::NTuple{N_NL, AbstractMultilinearMap{ORD}}
    external_system::Union{Nothing, ExternalSystem{N_EXT}}
    max_nl_degree::Int

    """
    			NthOrderModel(linear_terms, nonlinear_terms, external_system::ExternalSystem)

    			Construct an `NthOrderModel` from an `ExternalSystem` object directly.

    			# Checks performed
    			- Correct relationship between `ORD` and `ORDP1`.
    			- All matrices in `linear_terms` must be adequately sized.

    			The two-argument form, which builds a model *without* an external system, additionally
    			rejects any term with `multiplicity_external > 0` — such a term would otherwise construct
    			fine and fail only at evaluation time, inside `evaluate_term!`.
    			"""
    function NthOrderModel(
            linear_terms::NTuple{ORDP1, MT},
            nonlinear_terms::NTuple{N_NL, AbstractMultilinearMap{ORD}},
            external_system::ExternalSystem{N_EXT}
    ) where {ORD, ORDP1, N_NL, N_EXT, MT <: AbstractMatrix}
        T = eltype(MT)
        @assert ORDP1 == ORD + 1
        n_fom = size(linear_terms[1], 1)
        @assert all(size(B) == (n_fom, n_fom) for B in linear_terms)
        max_nl_deg = N_NL > 0 ?
                     maximum(t.deg-t.multiplicity_external for t in nonlinear_terms) : 1
        _info_implicit_symmetry(nonlinear_terms)
        new{ORD, ORDP1, N_NL, N_EXT, T, MT}(
            n_fom, linear_terms, nonlinear_terms, external_system, max_nl_deg)
    end

    # Constructor accepting the external dynamics polynomial directly
    function NthOrderModel(
            linear_terms::NTuple{ORDP1, MT},
            nonlinear_terms::NTuple{N_NL, AbstractMultilinearMap{ORD}},
            external_dynamics::DensePolynomial{TE, N_EXT}
    ) where {ORD, ORDP1, N_NL, N_EXT, TE, MT <: AbstractMatrix}
        T = eltype(MT)
        @assert ORDP1 == ORD + 1
        n_fom = size(linear_terms[1], 1)
        @assert all(size(B) == (n_fom, n_fom) for B in linear_terms)
        max_nl_deg = N_NL > 0 ? maximum(t.deg for t in nonlinear_terms) : 1
        _info_implicit_symmetry(nonlinear_terms)
        new{ORD, ORDP1, N_NL, N_EXT, T, MT}(
            n_fom, linear_terms, nonlinear_terms, ExternalSystem(external_dynamics), max_nl_deg)
    end

    # Constructor without external system
    function NthOrderModel(
            linear_terms::NTuple{ORDP1, MT},
            nonlinear_terms::NTuple{N_NL, AbstractMultilinearMap{ORD}}
    ) where {ORD, ORDP1, N_NL, MT <: AbstractMatrix}
        T = eltype(MT)
        @assert ORDP1 == ORD + 1
        n_fom = size(linear_terms[1], 1)
        @assert all(size(B) == (n_fom, n_fom) for B in linear_terms)
        max_nl_deg = N_NL > 0 ? maximum(t.deg for t in nonlinear_terms) : 1
        _check_external_terms(nonlinear_terms)
        _info_implicit_symmetry(nonlinear_terms)
        new{ORD, ORDP1, N_NL, 0, T, MT}(
            n_fom, linear_terms, nonlinear_terms, nothing, max_nl_deg)
    end

    # Constructor without external system and nonlinear_terms#
    function NthOrderModel(linear_terms::NTuple{ORDP1, MT}) where {ORDP1, MT}
        @warn ("Definition of NthOrderModel without nonlinear terms.")
        n_fom = size(linear_terms[1], 1)
        T = eltype(linear_terms[1])
        new{ORDP1 - 1, ORDP1, 0, 0, T, MT}(n_fom, linear_terms, tuple(), nothing, 1)
    end
end

"""
	Base.show(io::IO, ::MIME"text/plain", m::NthOrderModel)

Print a summary of the model.

Without this, showing a model dumps every field — which for a FEM backend means the entire
`DofHandler` behind each nonlinear term. Everything printed here is already on the type;
nothing is computed.

The one-line method exists for the same reason: a model nested inside another container
would otherwise expand its whole field tree there too.
"""
function Base.show(io::IO,
        m::NthOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT}) where {
        ORD, ORDP1, N_NL, N_EXT, T, MT}
    print(io, "NthOrderModel{ORD=", ORD, ", N_EXT=", N_EXT, "} FOM=", m.n_fom, ", ",
        MT <: SparseMatrixCSC ? "sparse" : "dense", ", ", N_NL,
        N_NL == 1 ? " nonlinear term" : " nonlinear terms",
        ", max degree ", m.max_nl_degree)
end

function Base.show(io::IO, ::MIME"text/plain",
        m::NthOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT}) where {
        ORD, ORDP1, N_NL, N_EXT, T, MT}
    layout = MT <: SparseMatrixCSC ? "sparse" : "dense"
    println(io, "NthOrderModel{ORD = ", ORD, ", N_EXT = ", N_EXT, "}")
    println(io, "  size      : FOM = ", m.n_fom, ",  ", layout, " (", MT, ")")
    println(io, "  linear    : ", ORDP1, " operators (B_0 … B_", ORD, ")")
    if N_NL == 0
        println(io, "  nonlinear : none")
    else
        # A FEM model carries one term per polynomial degree, but a hand-written one can
        # carry many; cap the list so a `show` never floods the REPL.
        shown = min(N_NL, 6)
        for i in 1:shown
            t = m.nonlinear_terms[i]
            key = i == 1 ? "  nonlinear : " : "              "
            println(io, key, _term_label(t), "  (deg ", t.deg, ")")
        end
        shown < N_NL && println(io, "              … and ", N_NL - shown, " more")
        # Not `maximum(t.deg)`: with an external system the external factors are subtracted,
        # and this is the value the solve's progress exponent actually uses.
        println(io, "  max degree: ", m.max_nl_degree)
    end
    print(io, "  external  : ")
    if m.external_system === nothing
        print(io, "none")
    else
        print(io, "N_EXT = ", N_EXT, ",  λ = ", Vector(m.external_system.eigenvalues))
        external_basis(m.external_system) === nothing || print(io, "  (re-based)")
    end
    return nothing
end

"""
	evaluate_nonlinear_terms!(res, model, order, state_vectors, r = nothing)

Evaluate all nonlinear terms of a given polynomial degree for an `NthOrderModel`.

# Arguments
- `res`: output vector (modified in-place)
- `model`: the `NthOrderModel`
- `order`: degree of the nonlinear terms to evaluate
- `state_vectors`: tuple `(x, x^(1), …, x^(ORD-1))` of state derivatives
- `r`: external state vector (default `nothing`). Must be provided if any term uses external variables.

!!! note "`r` is the *physical* external state"
	Terms are defined in the external coordinates the model was written in.  When the
	external system was re-based (its linear matrix was not upper triangular, so
	`ExternalSystem` chose a new basis `Q`), the solver's *reduced* external coordinates
	`r′` are related to these by `r = Q r′`, and it is the caller's job to convert:
	`ExternalSystems.to_physical_external(model.external_system, r′)` does it, and is the
	identity for every system that was not re-based.  Passing `r′` where `r` is expected
	silently evaluates the terms at the wrong point.
"""
function evaluate_nonlinear_terms!(res, model::NthOrderModel{ORD, ORDP1, N_NL},
        order, state_vectors, r = nothing) where {ORD, ORDP1, N_NL}
    order <= 0 && return res
    @assert length(res)==model.n_fom "Result vector length does not match full‑order state dimension"

    for term in model.nonlinear_terms
        if term.deg == order
            evaluate_term!(res, term, state_vectors, r)
        end
    end
end

"""
	linear_first_order_matrices(model::NthOrderModel)

Construct the matrices A and B of the equivalent linear first-order system:

	B Ẋ = A X

obtained from the ORD-th order model

	B_ORD x^(ORD) + ... + B_1 x^(1) + B_0 x = F(...)

by introducing the augmented state vector

	X = [x, x^(1), ..., x^(ORD-1)].

and the (ORD*n_fom x ORD*n_fom)-block matrices

	B = [ I   0   0   ⋯   0
		  0   I   0   ⋯   0
		  ⋮       ⋱
		  0   0   0   ⋯  B_ORD ]

and

	A = [ 0   I   0   ⋯   0
		  0   0   I   ⋯   0
		  ⋮       ⋱
		 -B₀ -B₁ -B₂ ⋯ -B_{ORD-1} ]

where `I` is the `n_fom × n_fom` identity matrix.
"""
function linear_first_order_matrices(model::NthOrderModel{ORD, ORDP1, N_NL, N_EXT, T,
        MT}
) where {ORD, ORDP1, N_NL, N_EXT, T, MT <: SparseMatrixCSC{T}}
    n = model.n_fom
    #T = eltype(model.linear_terms[1])
    total = ORD * n

    B = spzeros(T, total, total)
    A = spzeros(T, total, total)
    Id = sparse(one(T) * I, n, n)

    # --- B matrix ---
    for i in 1:(ORD - 1)
        rows = ((i - 1) * n + 1):(i * n)
        B[rows, rows] .= Id
    end

    # last block
    rows = ((ORD - 1) * n + 1):(ORD * n)
    B[rows, rows] .= model.linear_terms[end]   # B_ORD

    # --- A matrix ---

    # shift identities
    for i in 1:(ORD - 1)
        rows = ((i - 1) * n + 1):(i * n)
        cols = (i * n + 1):((i + 1) * n)
        A[rows, cols] .= Id
    end

    # last row: -B0 ... -B_{ORD-1}
    lastrow = ((ORD - 1) * n + 1):(ORD * n)

    for i in 1:ORD
        cols = ((i - 1) * n + 1):(i * n)
        A[lastrow, cols] .= -model.linear_terms[i]
    end

    return A, B
end

function linear_first_order_matrices(model::NthOrderModel{ORD, ORDP1, N_NL, N_EXT, T,
        MT}
) where {ORD, ORDP1, N_NL, N_EXT, T, MT <: AbstractMatrix{T}}
    n = model.n_fom
    # T = eltype(model.linear_terms[1])
    total = ORD * n

    B = zeros(T, total, total)
    A = zeros(T, total, total)
    Id = Matrix{T}(I, n, n)

    # --- B matrix ---
    for i in 1:(ORD - 1)
        rows = ((i - 1) * n + 1):(i * n)
        B[rows, rows] .= Id
    end

    # last block
    rows = ((ORD - 1) * n + 1):(ORD * n)
    B[rows, rows] .= model.linear_terms[end]   # B_ORD

    # --- A matrix ---

    # shift identities
    for i in 1:(ORD - 1)
        rows = ((i - 1) * n + 1):(i * n)
        cols = (i * n + 1):((i + 1) * n)
        A[rows, cols] .= Id
    end

    # last row: -B0 ... -B_{ORD-1}
    lastrow = ((ORD - 1) * n + 1):(ORD * n)

    for i in 1:ORD
        cols = ((i - 1) * n + 1):(i * n)
        A[lastrow, cols] .= -model.linear_terms[i]
    end

    return A, B
end

end # module
