# =============================================================================
# Fused Horner pass: parametrisation operator L(s) and lower-order RHS
# =============================================================================

"""
	evaluate_system_matrix_and_lower_order_rhs!(parametrisation_operator,
												lower_order_rhs,
												s, lower_order_couplings,
												linear_terms)
	-> parametrisation_operator

Evaluate the parametrisation operator `L(s)` **and** accumulate the lower-order
right-hand-side contributions in a **single Horner pass**, reusing the transient
intermediate matrices that are available only during the polynomial evaluation.

## Mathematical context

At step `j` of the Horner recurrence (before the scalar multiply by `s`),
the intermediate matrix

```
L[j](s) = Σ_{k=j+1}^{ORD+1} B[k] · s^{k-(j+1)}
```

is available. Multiplying by the pre-computed coupling vector
`ξ[j] = lower_order_couplings[j]` gives the contribution of lower-order solution
terms at derivative order `j` to the right-hand side:

```
contribution[j] = -L[j](s) · ξ[j]
```

The negative sign arises because these terms originate from the left-hand side of
the cohomological equation and are transposed to the right-hand side.

Summed over `j = 1, …, ORD`, the full lower-order RHS is

```
lower_order_rhs = -Σ_{j=1}^{ORD} L[j](s) · ξ[j]
				= -Σ_{j=1}^{ORD} ( Σ_{k=j+1}^{ORD+1} B[k] · s^{k-(j+1)} ) · ξ[j]
```

This computation **must** share the Horner loop with `L(s)`: the `L[j]`
intermediates are transient, and recomputing them would double the
`O(ORD · FOM²)` work.

The coupling vectors are obtained from
`MORFE.LowerOrderCouplings.compute_lower_order_couplings` applied to the
lower-order multi-indices associated with each Horner step.

## Arguments

- `parametrisation_operator :: AbstractMatrix{T}` – output buffer (`FOM × FOM`),
  overwritten with `L(s) = Σ_{k=1}^{ORD+1} B[k] · s^{k-1}`.
- `lower_order_rhs :: AbstractVector{T}` – accumulator (length `FOM`), updated
  in-place. Must be initialised to zero (or the desired starting value) by the
  caller.
- `s :: T` – evaluation superharmonic.
- `lower_order_couplings :: SVector{ORD, <:AbstractVector{T}}` – coupling vectors
  `ξ[j]` for `j = 1,…,ORD`; each element is an `AbstractVector{T}` of length
  `FOM`.
- `linear_terms :: NTuple{ORD+1, <:AbstractMatrix{T}}` – `linear_terms[k] = B[k]`.

## Complexity

`O(ORD · FOM²)`, shared with the `L(s)` evaluation.
"""
function evaluate_system_matrix_and_lower_order_rhs!(
        parametrisation_operator::AbstractMatrix,
        lower_order_rhs::AbstractVector,
        s::Number,
        lower_order_couplings::AbstractVector{<:AbstractVector},
        linear_terms::NTuple{ORDP1, <:AbstractMatrix}
) where {ORDP1}
    T = eltype(parametrisation_operator)  # output type set by the caller's buffer
    ORD = ORDP1 - 1
    @assert length(lower_order_couplings) == ORD "length(lower_order_couplings) must equal ORD = length(linear_terms) - 1."

    copyto!(parametrisation_operator, linear_terms[ORDP1]) # L ← B[ORD+1]

    for j in ORD:-1:1
        # Here: parametrisation_operator = L[j](s) = Σ_{k=j+1}^{ORD+1} B[k] · s^{k-(j+1)}.
        # Accumulate: lower_order_rhs -= L[j](s) · ξ[j].
        # mul!(y, M, x, -1, 1) computes y = y - M·x without allocation.
        mul!(lower_order_rhs, parametrisation_operator,
            lower_order_couplings[j], -one(T), one(T))

        rmul!(parametrisation_operator, s)             # L ← L · s
        parametrisation_operator .+= linear_terms[j]   # L ← L + B[j]
        # Here: parametrisation_operator = L_{j-1}(s) = Σ_{k=j}^{ORD+1} B[k] · s^{k-j}.
    end
    # On exit: parametrisation_operator = L_0(s) = Σ_{k=1}^{ORD+1} B[k] · s^{k-1} = L(s).
    return parametrisation_operator
end

# =============================================================================
# Sparse Horner evaluation of L(s) with simultaneous RHS accumulation
# =============================================================================

"""
	precompute_sparse_L_template(linear_terms) -> (L_template, mappings)

Pre-allocate a `SparseMatrixCSC{ComplexF64}` with the **union** sparsity pattern of
all `linear_terms`, and compute index mappings so that each entry of `linear_terms[k]`
can be accumulated into the correct position of the template's `nzval` array.

Returns `(L_template, mappings)` where `mappings[k][i]` is the index into
`L_template.nzval` that corresponds to position `i` of `linear_terms[k].nzval`.

Used once at context construction; the template is reused across all monomials by
the in-place `build_sparse_L_and_rhs!` overload, eliminating per-monomial sparse
arithmetic allocations even when the input matrices do not share a common pattern
(e.g. when the damping matrix is `C = α*M + β*K` and Julia's sparse addition drops
entries that cancel to exactly zero).
"""
function precompute_sparse_L_template(
        linear_terms::NTuple{ORDP1, <:SparseMatrixCSC},
) where {ORDP1}
    n = size(linear_terms[1], 1)

    # Build union pattern by summing all-ones matrices — positive values cannot
    # cancel, so every structural nonzero from every Bk is preserved.
    L_ones = spzeros(Int, n, n)
    for Bk in linear_terms
        L_ones = L_ones + SparseMatrixCSC(
            Bk.m, Bk.n, copy(Bk.colptr), copy(Bk.rowval), ones(Int, nnz(Bk))
        )
    end

    # Allocate template: union pattern, zero ComplexF64 nzval (overwritten per monomial).
    L_template = SparseMatrixCSC(
        n, n, copy(L_ones.colptr), copy(L_ones.rowval), zeros(ComplexF64, nnz(L_ones))
    )

    # For each term Bk, map its nzval positions into L_template.nzval positions.
    mappings = Vector{Vector{Int}}(undef, ORDP1)
    for k in 1:ORDP1
        Bk = linear_terms[k]
        mapping_k = Vector{Int}(undef, nnz(Bk))
        for col in 1:n
            for pos_k in Bk.colptr[col]:(Bk.colptr[col + 1] - 1)
                row = Bk.rowval[pos_k]
                lo = L_template.colptr[col]
                hi = L_template.colptr[col + 1] - 1
                pos_L = searchsortedfirst(
                    L_template.rowval, row, lo, hi, Base.Order.Forward)
                mapping_k[pos_k] = pos_L
            end
        end
        mappings[k] = mapping_k
    end

    return L_template, mappings
end

"""
	build_sparse_L_and_rhs!(rhs, L_template, mappings, linear_terms, s, lower_order_couplings)
	-> L_template

In-place Horner evaluation of the parametrisation operator using a pre-allocated
union-pattern template and index mappings from [`precompute_sparse_L_template`](@ref).

Avoids ALL per-monomial sparse arithmetic allocations regardless of whether the
`linear_terms` share a common sparsity pattern.  The template's `nzval` is overwritten
on each call; its `colptr`/`rowval` are never modified.
"""
function build_sparse_L_and_rhs!(
        rhs::AbstractVector,
        L_template::SparseMatrixCSC,
        mappings::Vector{Vector{Int}},
        linear_terms::NTuple{ORDP1, <:SparseMatrixCSC},
        s,
        lower_order_couplings::AbstractVector{<:AbstractVector}
) where {ORDP1}
    T = eltype(rhs)
    T_L = eltype(L_template)
    ORD = ORDP1 - 1

    # Horner init: L_template ← linear_terms[ORDP1], mapped into union positions.
    fill!(L_template.nzval, zero(T_L))
    nzval_last = linear_terms[ORDP1].nzval
    map_last = mappings[ORDP1]
    @inbounds for i in eachindex(nzval_last)
        L_template.nzval[map_last[i]] = T_L(nzval_last[i])
    end

    @inbounds for j in ORD:-1:1
        mul!(rhs, L_template, lower_order_couplings[j], -one(T), one(T))  # rhs -= L · ξ[j]
        L_template.nzval .*= s                                              # L ← L · s
        nzval_j = linear_terms[j].nzval
        map_j = mappings[j]
        for i in eachindex(nzval_j)
            L_template.nzval[map_j[i]] += T_L(nzval_j[i])                  # L ← L + B[j]
        end
    end

    return L_template
end
