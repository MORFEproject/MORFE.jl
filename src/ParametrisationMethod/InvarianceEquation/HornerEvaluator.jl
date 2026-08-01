# =============================================================================
# Fused Horner pass: parametrisation operator L(s) and lower-order RHS
# =============================================================================

"""
	evaluate_system_matrix_and_lower_order_rhs!(parametrisation_operator, lower_order_rhs, s, lower_order_couplings, linear_terms)
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
lower_order_rhs = -Σ_{j=1}^{ORD} L[j](s) · ξ[j] = -Σ_{j=1}^{ORD} ( Σ_{k=j+1}^{ORD+1} B[k] · s^{k-(j+1)} ) · ξ[j]
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
	linear_terms::NTuple{ORDP1, <:AbstractMatrix},
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
			Bk.m, Bk.n, copy(Bk.colptr), copy(Bk.rowval), ones(Int, nnz(Bk)),
		)
	end

	# Allocate template: union pattern, zero ComplexF64 nzval (overwritten per monomial).
	L_template = SparseMatrixCSC(
		n, n, copy(L_ones.colptr), copy(L_ones.rowval), zeros(ComplexF64, nnz(L_ones)),
	)

	# For each term Bk, map its nzval positions into L_template.nzval positions.
	mappings = Vector{Vector{Int}}(undef, ORDP1)
	for k in 1:ORDP1
		Bk = linear_terms[k]
		mapping_k = Vector{Int}(undef, nnz(Bk))
		for col in 1:n
			for pos_k in Bk.colptr[col]:(Bk.colptr[col+1]-1)
				row = Bk.rowval[pos_k]
				lo = L_template.colptr[col]
				hi = L_template.colptr[col+1] - 1
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
	precompute_sparse_bordered_template(L_template, ROM) -> (M, border_row_base)

Allocate the constant-size `(FOM+ROM) × (FOM+ROM)` **bordered** cohomological matrix

```
	┌                              ┐
	│  L(s)      C(s) P            │   FOM rows  (invariance)
	│  P Ĵ(s)    P Ĉ(s) P + τ Q    │   ROM rows  (orthogonality / R_α = 0)
	└                              ┘
```

whose sparsity pattern depends **only** on the union pattern of `L_template` and on
`ROM` — never on the resonance mask `P = diag(ρ)` of the monomial being solved.
Non-resonant border entries are carried as *numeric* zeros in structural positions,
which is exactly what allows one symbolic factorisation to be reused for every
monomial. The system being represented is documented in the `CohomologicalEquations`
module docstring; what follows is this function's own contract — where each block
lands in the CSC arrays, which the assembly in `_solve_monomial!` writes to directly.

Block layout, in CSC order:

| block          | rows        | cols        | pattern                        |
|:---------------|:------------|:------------|:-------------------------------|
| `L`            | `1:FOM`     | `1:FOM`     | union pattern of `L_template`  |
| `C P`          | `1:FOM`     | `FOM+1:end` | dense `FOM × ROM`              |
| `P Ĵ`          | `FOM+1:end` | `1:FOM`     | dense `ROM × FOM`              |
| `P Ĉ P + τ Q`  | `FOM+1:end` | `FOM+1:end` | dense `ROM × ROM`              |

so `nnz(M) = nnz(L_template) + 2·FOM·ROM + ROM²`. Appending the border rows to each
of the first `FOM` columns preserves sorted `rowval` because every union row index
is `≤ FOM`.

## Returns

- `M :: SparseMatrixCSC` — the template; only `nzval` is ever written afterwards.
- `border_row_base :: Vector{Int}` — length `FOM`; entry `M[FOM+r, c]` for `c ≤ FOM`
  lives at `M.nzval[border_row_base[c] + r - 1]`.

No `L → M` index table is returned or needed.  Because the `L` entries of column `c`
are laid down as a contiguous *prefix* of that column, the map is affine within each
column,

```
	L_template.nzval[p]  ↦  M.nzval[M.colptr[c] + (p - L_template.colptr[c])]
```

so [`scatter_L_into_bordered!`](@ref) is a per-column block copy rather than an
indirect gather — which also saves an `nnz(L)`-length index vector.

Border *column* positions need no table either: column `FOM+q` starts at
`bq = M.colptr[FOM+q]`, so `C_q(s)` occupies the contiguous run
`M.nzval[bq : bq+FOM-1]` and the corner entry `M[FOM+m, FOM+q]` sits at
`bq + FOM + m - 1`.
"""
function precompute_sparse_bordered_template(
	L_template::SparseMatrixCSC{Tv, Ti},
	ROM::Int,
) where {Tv, Ti}
	FOM = size(L_template, 1)
	N = FOM + ROM
	nnz_M = nnz(L_template) + 2 * FOM * ROM + ROM^2

	colptr = Vector{Ti}(undef, N + 1)
	rowval = Vector{Ti}(undef, nnz_M)
	border_row_base = Vector{Int}(undef, FOM)

	colptr[1] = 1
	for c in 1:FOM
		lo = L_template.colptr[c]
		hi = L_template.colptr[c+1] - 1
		base = colptr[c]
		# L entries first, as a contiguous prefix (all row indices ≤ FOM, already
		# sorted) — this prefix layout is what makes the L → M map affine per column.
		for (k, p) in enumerate(lo:hi)
			rowval[base+k-1] = L_template.rowval[p]
		end
		# … then the ROM border rows, which keeps the column sorted.
		b = base + (hi - lo + 1)
		border_row_base[c] = b
		for r in 1:ROM
			rowval[b+r-1] = FOM + r
		end
		colptr[c+1] = b + ROM
	end
	for q in 1:ROM
		base = colptr[FOM+q]
		for i in 1:N
			rowval[base+i-1] = i
		end
		colptr[FOM+q+1] = base + N
	end

	M = SparseMatrixCSC(N, N, colptr, rowval, zeros(Tv, nnz_M))
	return M, border_row_base
end

"""
	scatter_L_into_bordered!(M, L_template) -> M

Copy the freshly evaluated `L(s)` from the standalone Horner workspace into the
`(1,1)` block of the bordered template.

`L`'s entries for column `c` sit contiguously in both matrices — at
`L_template.colptr[c] …` and at `M.colptr[c] …` respectively (see
[`precompute_sparse_bordered_template`](@ref)) — so each column is a single
`copyto!` block move rather than an element-by-element gather through an index
table.  Keeping this a separate step is what leaves [`build_sparse_L_and_rhs!`](@ref)
untouched: it needs its own square workspace for the transient Horner intermediates
`L[j](s)` that accumulate the lower-order RHS.
"""
function scatter_L_into_bordered!(M::SparseMatrixCSC, L_template::SparseMatrixCSC)
	Mnz, Mcp = M.nzval, M.colptr
	Lnz, Lcp = L_template.nzval, L_template.colptr
	FOM = size(L_template, 1)
	ROM = size(M, 1) - FOM
	# Every column of M has exactly ROM more slots than the matching column of L, so a
	# mismatched pair would still write *in bounds* — just into the wrong positions,
	# silently. Check the identity that pins the layout instead of relying on bounds.
	@assert ROM ≥ 0 && size(M, 2) == size(M, 1) "bordered matrix must be square and at \
least as large as L_template; got $(size(M)) against L $(size(L_template))"
	@assert nnz(M) == nnz(L_template) + 2 * FOM * ROM + ROM^2 "bordered matrix was not \
	  built from this L_template: nnz(M) = $(nnz(M)) ≠ $(nnz(L_template) + 2 * FOM * ROM + ROM^2)"
	@inbounds for c in 1:FOM
		lo = Lcp[c]
		n = Lcp[c+1] - lo
		n == 0 && continue
		copyto!(Mnz, Mcp[c], Lnz, lo, n)
	end
	return M
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
	lower_order_couplings::AbstractVector{<:AbstractVector},
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
