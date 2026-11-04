"""
Module `ParametrisationObjects` — the coefficient containers of the DPIM
parametrisation, and the contract their multiindex set must satisfy.

Defines the two objects that together represent the invariant manifold and the
reduced dynamics:

- `Parametrisation{ORD, NVAR, T}` — the map `W : ℂᴺᵛᵃʳ → ℂᶠᵒᵐ` expanded as a
  `DensePolynomial` with a `(FOM × ORD × L)` coefficient tensor; the `ORD` axis
  stores the time-derivative orders required by higher-order ODEs.
- `ReducedDynamics{ROM, NVAR, T}` — the reduced ODE `ż = R(z)` expanded as a
  `DensePolynomial` with a `(NVAR × L)` coefficient matrix.

Also provides `create_parametrisation_method_objects` (allocates both objects for a
given `MultiindexSet`), `compute_higher_derivative_coefficients!` (fills the
derivative-order slices of `W` from the solved first-order slice and the reduced
dynamics), and `validate_multiindex_set` (the five-clause `mset` contract).

## Why this is a module of its own

`CohomologicalEquations` needs exactly these definitions, while the user-facing
`ParametrisationMethod` — which owns `parametrise` — needs *`CohomologicalEquations`*.
Splitting the containers out breaks what would otherwise be a circular module
dependency, and lets `ParametrisationMethod` load last and simply call the solver.
Everything here is re-exported by `ParametrisationMethod`, so
`MORFE.ParametrisationMethod.ReducedDynamics` and friends keep resolving.
"""
module ParametrisationObjects

using LinearAlgebra: mul!
using StaticArrays: SVector
using ..Multiindices: MultiindexSet, find_in_set, is_downward_closed, is_conjugate_closed
using ..Polynomials: DensePolynomial, restrict_polynomial_to_degree

export Parametrisation, ReducedDynamics, create_parametrisation_method_objects,
       compute_higher_derivative_coefficients!,
       restrict_ReducedDynamics_to_degree, restrict_Parametrisation_to_degree,
       validate_multiindex_set

# ==================== Multiindex-set contract ====================

"""
	validate_multiindex_set(mset, nvar, rom; conjugate_permutation = nothing)

Check a custom multiindex set against everything the cohomological solve assumes, and
throw an `ArgumentError` naming the offending exponent on the first violation.

The clauses, and why each one matters:

1. **`nvar` variables**, matching `ROM + N_EXT`.
2. **Minimum total degree ≥ 1** — the expansion is centred on the fixed point, so the
   constant monomial has no coefficient to solve for.
3. **Every unit multiindex `eᵢ`** — the linear part of the parametrisation is
   initialised from the eigenvectors, one column per unit multiindex.
4. **Downward closed** — the graded solve reads `W[α - β + eᵢ]` while working on `α`
   and factorises `α = β₁ + … + β_d` over members of `mset`. A missing divisor is not an
   error at run time: it is read as zero, silently corrupting the right-hand side.
5. With a `conjugate_permutation`: that it is an involutive permutation of `1:nvar`
   mapping `1:rom` into itself, and that `mset` is **closed under it**. A member whose
   partner is absent is solved directly rather than filled by conjugation, so the result
   loses the conjugate structure the permutation asserts.

Returns `nothing`.  Cost is `O(nvar · |mset| · log|mset|)` — negligible beside a solve,
but `parametrise` and `solve_parametrisation` both take a `validate_mset = false`
escape hatch for callers that have already checked.

See [`is_downward_closed`](@ref) and [`is_conjugate_closed`](@ref) for the two closure
predicates on their own.
"""
function validate_multiindex_set(mset::MultiindexSet{N}, nvar::Int, rom::Int;
        conjugate_permutation::Union{Nothing, AbstractVector{Int}} = nothing) where {N}
    N == nvar || throw(ArgumentError(
        "custom mset has $N variables, but the model requires NVAR = ROM + N_EXT = $nvar"))
    isempty(mset.exponents) && throw(ArgumentError("custom mset is empty"))

    # Grlex puts the lowest degree first, so one look at the head settles clause 2.
    sum(first(mset.exponents)) ≥ 1 || throw(ArgumentError(
        "custom mset must not contain the zero multiindex (min total degree ≥ 1)"))

    for i in 1:N
        unit = [j == i ? 1 : 0 for j in 1:N]
        find_in_set(mset, unit) === nothing && throw(ArgumentError(
            "custom mset is missing the unit multiindex e_$i; the linear " *
            "initialisation of the parametrisation requires all unit multiindices"))
    end

    if !is_downward_closed(mset)
        α, β = _first_missing_divisor(mset)
        throw(ArgumentError(
            "custom mset is not downward closed: $(Vector(α)) is a member but its " *
            "divisor $(Vector(β)) is not. The graded solve reads W[α - β + eᵢ] and " *
            "factorises α over members of mset, so a missing divisor is read as zero " *
            "and silently corrupts the right-hand side."))
    end

    conjugate_permutation === nothing && return nothing
    perm = conjugate_permutation
    length(perm) == N || throw(ArgumentError(
        "conjugate_permutation has $(length(perm)) entries, but NVAR = $N"))
    sort(collect(perm)) == collect(1:N) || throw(ArgumentError(
        "conjugate_permutation must be a permutation of 1:$N, got $(collect(perm))"))
    all(i -> perm[perm[i]] == i, 1:N) || throw(ArgumentError(
        "conjugate_permutation must be an involution (perm[perm[i]] == i), " *
        "got $(collect(perm))"))
    # The reduced rows are filled as R[r, conj] = conj(R[perm[r], src]) for r in 1:ROM
    # only, so the master block must be closed under the permutation.
    all(r -> perm[r] ≤ rom, 1:rom) || throw(ArgumentError(
        "conjugate_permutation must map the master block 1:$rom into itself, " *
        "got $(collect(perm)); pairing a master coordinate with an external one " *
        "would fill the reduced dynamics from the wrong row"))

    if !is_conjugate_closed(mset, perm)
        α, pα = _first_missing_conjugate(mset, perm)
        throw(ArgumentError(
            "custom mset is not closed under conjugate_permutation $(collect(perm)): " *
            "$(Vector(α)) is a member but its conjugate $(Vector(pα)) is not. The " *
            "solve fills conjugate monomials from their partners, so the result would " *
            "lack the conjugate structure the permutation asserts."))
    end
    return nothing
end

# Locate one concrete (member, absent divisor) pair, for the error message only.
function _first_missing_divisor(mset::MultiindexSet{N}) where {N}
    for α in mset.exponents
        sum(α) > 1 || continue          # degree 1 divides only the exempt constant
        for i in 1:N
            α[i] == 0 && continue
            β = α - SVector{N, Int}(ntuple(j -> j == i ? 1 : 0, Val(N)))
            find_in_set(mset, β) === nothing && return (α, β)
        end
    end
    error("mset reported not downward closed but no missing divisor was found")
end

# Locate one concrete (member, absent conjugate) pair, for the error message only.
function _first_missing_conjugate(mset::MultiindexSet{N}, perm) where {N}
    for α in mset.exponents
        pα = SVector{N, Int}(ntuple(k -> α[perm[k]], Val(N)))
        pα == α && continue
        find_in_set(mset, pα) === nothing && return (α, pα)
    end
    error("mset reported not conjugate closed but no missing partner was found")
end

"""
	Parametrisation{ORD, NVAR, T}

A dense polynomial with a contiguous `(FOM, ORD, L)` coefficient array.
Represents a parametrisation mapping from reduced coordinates and forcing variables to the full state.

- `ORD`: native order of the full ODE (1 for first‑order, 2 for second‑order).
- `NVAR`: total number of variables = reduced coordinates + forcing variables.
- `T`: numeric element type (e.g., `ComplexF64`).

Layout: `coefficients[:, ord, l]` is the full‑state vector (length FOM) for the
`ord`-th time derivative of the `l`-th monomial coefficient.

# Fields

- `poly::DensePolynomial{T, NVAR, 3, Array{T, 3}}` — the coefficient array in the
  layout above, together with the multiindex set it is aligned to.
- `external_system_size::Int` — how many of the `NVAR` variables are external
  forcing amplitudes rather than reduced coordinates.  The reduced dimension is the
  remainder, so this is what separates the master block from the forcing block when
  slicing the coefficients.
"""
struct Parametrisation{ORD, NVAR, T}
    poly::DensePolynomial{T, NVAR, 3, Array{T, 3}}
    external_system_size::Int

    function Parametrisation(poly::DensePolynomial{T, NVAR, 3, Array{T, 3}},
            external_system_size::Int) where {T, NVAR}
        ORD = size(poly.coefficients, 2)
        @assert external_system_size >= 0 "external_system_size must be non‑negative"
        new{ORD, NVAR, T}(poly, external_system_size)
    end
end

Base.size(W::Parametrisation) = size(W.poly.coefficients, 1) # FOM: full‑order state dimension
# `multiindex_set` and `coefficients` here are generic functions OWNED BY THIS MODULE.
# `Polynomials` defines its own functions of the same names for `DensePolynomial`
# (Polynomials.jl:216-217).  They are deliberately NOT the same generic: do not `import`
# the Polynomials versions to "unify" them, or every consumer's method table changes.
multiindex_set(W::Parametrisation) = W.poly.multiindex_set
coefficients(W::Parametrisation) = W.poly.coefficients

"""
	ReducedDynamics{ROM, NVAR, T}

A dense polynomial whose coefficients are `SVector{ROM, T}`.
Represents the reduced dynamics on a manifold of dimension `ROM`.

- `ROM`: dimension of the reduced state (first‑order system).
- `NVAR`: total number of variables = ROM + external_system_size.
- `T`: numeric type.

The dynamics are: ż = R(z, r), where r are the forcing variables.

# Fields

- `poly::DensePolynomial{T, NVAR, 2, Matrix{T}}` — coefficients as a `NVAR × L`
  matrix aligned to the multiindex set.  Rows `1:ROM` are solved for; the trailing
  `external_system_size` rows hold the known forcing amplitudes.
- `external_system_size::Int` — number of forcing variables, fixing where the
  master rows end and `ROM = NVAR - external_system_size`.
"""
struct ReducedDynamics{ROM, NVAR, T}
    poly::DensePolynomial{T, NVAR, 2, Matrix{T}}
    external_system_size::Int

    function ReducedDynamics(poly::DensePolynomial{T, NVAR, 2, Matrix{T}},
            external_system_size::Int) where {T, NVAR}
        @assert external_system_size >= 0 "external_system_size must be non‑negative"
        ROM = NVAR - external_system_size
        @assert ROM > 0 "ROM = NVAR - external_system_size must be positive; got $(ROM)"
        new{ROM, NVAR, T}(poly, external_system_size)
    end
end

Base.size(::ReducedDynamics{ROM}) where {ROM} = ROM
multiindex_set(R::ReducedDynamics) = R.poly.multiindex_set
coefficients(R::ReducedDynamics) = R.poly.coefficients

"""
	create_parametrisation_method_objects(mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int, ROM::Int, external_system_size::Int, ::Type{T}=Complex)

Create a consistent pair of polynomials:
- `W`: a `Parametrisation{ORD, NVAR, T}` with zero coefficients,
- `R`: a `ReducedDynamics{ROM, NVAR, T}` with zero coefficients.

Both polynomials share the same multiindex set `mset` and element type `T`.
The total number of variables `NVAR` must satisfy `NVAR == ROM + external_system_size`.
`FOM` is the full‑order dimension (size of the state vector). It is not stored but used
to initialise the coefficient vectors correctly.

# Arguments
- `mset`: multiindex set for `NVAR` variables.
- `ORD`: native order of the full ODE (1 or 2).
- `FOM`: dimension of the full‑order state in its native order.
- `ROM`: dimension of the reduced state.
- `external_system_size`: number of forcing variables (default 0).
- `T`: element type.
"""
function create_parametrisation_method_objects(
        mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int, ROM::Int, external_system_size::Int,
        ::Type{T} = Complex) where {T <: Number, NVAR}
    # Validate variable count
    @assert NVAR == ROM + external_system_size "Multiindex set has $NVAR variables, but ROM + external_system_size = $(ROM + external_system_size)"

    # Parametrisation coefficients: (FOM, ORD, L) 3-D array
    W_poly = DensePolynomial(zeros(T, FOM, ORD, length(mset)), mset)
    W = Parametrisation(W_poly, external_system_size)

    # Reduced dynamics coefficients: (ROM, L) matrix
    R_poly = DensePolynomial(zeros(T, NVAR, length(mset)), mset)
    # THE REDUCED DYNAMICS POLYNOMIAL HAS NVAR VARIABLES, NOT ROM, BECAUSE IT DEPENDS ON ALL REDUCED + EXTERNAL VARS
    # ADDITIONALLY, THE COEFFICIENTS ARE NVAR-VECTORS, NOT SCALARS, SO THE COEFFICIENT ARRAY HAS SHAPE (NVAR, L)
    # THE LAST ROWS OF THE COEFFICIENTS CORRESPOND TO THE EXTERNAL SYSTEM TERMS, WHICH ARE DIRECTLY COPIED FROM THE FULL ORDER MODEL
    R = ReducedDynamics(R_poly, external_system_size)

    return (W, R)
end

# For the special case where ROM = NVAR (no forcing, and reduced dimension equals variable count)
function create_parametrisation_method_objects(
        mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int,
        ::Type{T} = Complex) where {T <: Number, NVAR}
    return create_parametrisation_method_objects(mset, ORD, FOM, NVAR, 0, T)
end

"""
	compute_higher_derivative_coefficients!(
		param_coeff, red_coeff, external_dynamics, superharmonic, global_index,
		lower_order_couplings; self_variable = 0
	) -> nothing

Fill the higher time-derivative levels `W⁽ʲ⁺¹⁾[γ]`, `j = 1 … ORD-1`, of the monomial `γ` at
`global_index` from its solved first level, by differentiating `W⁽ʲ⁾` along `ż = R(z)`:

```
W⁽ʲ⁺¹⁾[γ] = s · W⁽ʲ⁾[γ]  +  Φ⁽ʲ⁾ R[γ]  +  ξ[j]
```

with `s = ⟨λ, γ⟩` the superharmonic, `ξ` the lower-order couplings and `Φ⁽ʲ⁾` the **linear
part of `W⁽ʲ⁾`**: its coefficients at the unit monomials, which a valid multiindex set holds
first and in variable order (masters `1:ROM`, then the external directions). `Φ⁽ʲ⁾` is read
in place at the same level `j`. That matters from `ORD = 3` on: at linear order
`Φ⁽ʲ⁺¹⁾ = Φ⁽ʲ⁾ Λ`, so a master mode's velocity column is `λφ` and its acceleration column
`λ²φ`, and level 1 at every level would put `φ` where `λφ` belongs. The master columns are
written from the spectral data before the solve, the external directions by this function
as they are solved.

At a unit monomial `γ = eₘ` (an external direction) the term `i = m` of `Φ⁽ʲ⁾ R[γ]` is the
same contribution as `s · W⁽ʲ⁾[γ]`. `self_variable = m` counts it once, so that the formula
reproduces `Φ⁽ʲ⁺¹⁾ = Φ⁽ʲ⁾ Λ`; `0` excludes nothing.

## Arguments

- `param_coeff :: AbstractArray{T, 3}` — the `FOM × ORD × L` coefficients of `W`.
- `red_coeff :: AbstractMatrix{T}` — the `ROM × L` master rows of `R`.
- `external_dynamics :: AbstractVector{T}` — the `N_EXT` external rows of `R[γ]`.
- `superharmonic :: T` — `s = ⟨λ, γ⟩`.
- `global_index :: Int` — the position of `γ` in the multiindex set.
- `lower_order_couplings :: AbstractVector{<:AbstractVector{T}}` — `ξ[j]` for each level,
  from [`LowerOrderCouplings.compute_lower_order_couplings`](@ref).
"""
function compute_higher_derivative_coefficients!(
        param_coeff::AbstractArray{T, 3},
        red_coeff::AbstractMatrix{T},
        external_dynamics::AbstractVector{T},
        superharmonic::T,
        global_index::Int,
        lower_order_couplings::AbstractVector{<:AbstractVector{T}};
        self_variable::Int = 0
) where {T}
    R_master = view(red_coeff, :, global_index)
    self_variable == 0 &&
        return _add_derivative_levels!(param_coeff, R_master, external_dynamics,
            superharmonic, global_index, lower_order_couplings)

    # γ = eₘ: zero the coefficient of its own column (in a copy, at most N_EXT times per
    # solve), so that column adds an exact zero — the s-term already accounts for it.
    R_master, R_ext = collect(R_master), collect(external_dynamics)
    ROM = length(R_master)
    if self_variable <= ROM
        R_master[self_variable] = zero(T)
    else
        R_ext[self_variable - ROM] = zero(T)
    end
    return _add_derivative_levels!(param_coeff, R_master, R_ext,
        superharmonic, global_index, lower_order_couplings)
end

"""
	_add_derivative_levels!(param_coeff, R_master, R_ext, superharmonic, global_index,
		lower_order_couplings) -> nothing

The loop of [`compute_higher_derivative_coefficients!`](@ref), behind a function barrier so
that its allocation-free path (views of `R`) and its unit-monomial path (copies) each
compile to type-stable code.
"""
function _add_derivative_levels!(
        param_coeff::AbstractArray{T, 3}, R_master::AbstractVector, R_ext::AbstractVector,
        superharmonic, global_index::Int, lower_order_couplings
) where {T}
    ROM, N_EXT = length(R_master), length(R_ext)
    for j in 1:(size(param_coeff, 2) - 1)
        W_j = view(param_coeff, :, j, global_index)
        W_next = view(param_coeff, :, j + 1, global_index)
        @. W_next = superharmonic * W_j + lower_order_couplings[j]
        mul!(W_next, view(param_coeff, :, j, 1:ROM), R_master, one(T), one(T))
        N_EXT > 0 && mul!(W_next, view(param_coeff, :, j, (ROM + 1):(ROM + N_EXT)),
            R_ext, one(T), one(T))
    end
    return nothing
end

"""
	function restrict_Parametrisation_to_degree(W::Parametrisation, max_degree::Int) -> Parametrisation

Returns a new Parametrisation that contains all the monomials of `poly` that are of degree lower or equal than `max_degree`. 
"""
function restrict_Parametrisation_to_degree(W::Parametrisation, max_degree::Int)
    new_poly = restrict_polynomial_to_degree(W.poly, max_degree)
    return Parametrisation(new_poly, W.external_system_size)
end

"""
	function restrict_ReducedDynamics_to_degree(R::ReducedDynamics, max_degree::Int) -> ReducedDynamics

Returns a new ReducedDynamics that contains all the monomials of `poly` that are of degree lower or equal than `max_degree`. 
"""
function restrict_ReducedDynamics_to_degree(R::ReducedDynamics, max_degree::Int)
    new_poly = restrict_polynomial_to_degree(R.poly, max_degree)
    return ReducedDynamics(new_poly, R.external_system_size)
end

end # module
