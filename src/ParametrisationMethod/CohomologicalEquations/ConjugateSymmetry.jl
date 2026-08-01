# =============================================================================
# Conjugate-symmetry exploitation for real-valued FOMs
# =============================================================================
# For a real-valued FOM the parametrisation and reduced dynamics satisfy
#   W_{P·γ} = conj(W_γ)           [W-symmetry]
#   f_{P·β}[r] = conj(f_β[perm[r]])  [f-symmetry]
# where P is an involutory permutation on mode indices that swaps conjugate pairs.
# This module encapsulates the bookkeeping and fill logic so that
# CohomologicalContext and CohomologicalBuffers are completely unchanged.

# =============================================================================
# Sentinel: inactive path
# =============================================================================

"""
	NoConjugatePermutation

Sentinel type indicating that conjugate-symmetry exploitation is disabled.
Dispatch on this type eliminates all symmetry bookkeeping at compile time.
"""
struct NoConjugatePermutation end

# =============================================================================
# ConjugateSymmetryData{CP}
# =============================================================================

"""
	ConjugateSymmetryData{CP}

Self-contained optimisation layer for exploiting complex-conjugate symmetry in
the cohomological solve.

# Type parameters

| Parameter | Meaning |
|:----------|:--------|
| `CP` | `NoConjugatePermutation` (inactive) or `SVector{NVAR, Int}` (active) |

Compile-time dispatch on `CP` eliminates secondary-monomial bookkeeping when
inactive.  When active, secondary monomials (those whose conjugate partner has
a lower multiindex-set index) are marked in `skip_bits` and skipped in the
outer solve loop; their coefficients are filled by `fill_conjugate_monomial!`.
"""
struct ConjugateSymmetryData{CP}
    permutation::CP
    monomial_map::Vector{Int}
    skip_bits::BitVector           # length L; true = skip this monomial (linears + secondaries)
    primary_pairs::Vector{NTuple{2, Int}}  # (source_idx, conj_idx) where conj_idx > source_idx
end

# =============================================================================
# _build_monomial_map
# =============================================================================

"""
	_build_monomial_map(mset, perm, mdict) -> Vector{Int}

Build the monomial conjugate map induced by the mode permutation `perm`.
`monomial_map[i]` is the index in `mset` of the monomial `P·γ` where `γ = mset[i]`
and `(P·γ)[k] = γ[perm[k]]`.  Returns `0` when `P·γ` is not in `mset`.
"""
function _build_monomial_map(
        mset::MultiindexSet{NVAR},
        perm::SVector{NVAR, Int},
        mdict::Dict{SVector{NVAR, Int}, Int}
) where {NVAR}
    L = length(mset)
    monomial_map = Vector{Int}(undef, L)
    for i in 1:L
        γ = mset[i]
        Pγ = SVector{NVAR, Int}(ntuple(k -> γ[perm[k]], Val(NVAR)))
        monomial_map[i] = Pγ == γ ? i : get(mdict, Pγ, 0)
    end
    return monomial_map
end

# =============================================================================
# _build_conjugate_symmetry — factory
# =============================================================================

"""
	_build_conjugate_symmetry(perm_or_sentinel, linear_skip_set, ...) -> ConjugateSymmetryData

Factory for `ConjugateSymmetryData`.

- **Inactive** (first argument is `NoConjugatePermutation()`): wraps `linear_skip_set`
  as `skip_bits` and returns a `ConjugateSymmetryData{NoConjugatePermutation}` with
  empty `primary_pairs`.
- **Active** (first argument is an `SVector{NVAR,Int}` involution): builds the monomial
  conjugate map, identifies primary/secondary pairs, populates `skip_bits` for both
  linear monomials and secondary conjugate monomials, and returns a
  `ConjugateSymmetryData{SVector{NVAR,Int}}`.
"""
function _build_conjugate_symmetry(::NoConjugatePermutation, linear_skip_set::Set{Int}, L::Int)
    skip_bits = falses(L)
    for i in linear_skip_set
        skip_bits[i] = true
    end
    return ConjugateSymmetryData{NoConjugatePermutation}(
        NoConjugatePermutation(), Int[], skip_bits, NTuple{2, Int}[])
end

# Active path: perm must be a proper involution with no zero entries.
function _build_conjugate_symmetry(
        perm::SVector{NVAR, Int},
        linear_skip_set::Set{Int},
        mset::MultiindexSet{NVAR},
        mdict::Dict{SVector{NVAR, Int}, Int}
) where {NVAR}
    @assert all(i -> perm[i] > 0 && perm[perm[i]] == i, 1:NVAR) """
     conjugate_permutation must be an involution (perm[perm[i]] == i) with no zero entries.
     """

    monomial_map = _build_monomial_map(mset, perm, mdict)

    skip_bits = falses(length(mset))
    primary_pairs = NTuple{2, Int}[]
    for i in linear_skip_set
        skip_bits[i] = true
    end
    for i in eachindex(monomial_map)
        skip_bits[i] && continue          # linear or already-marked secondary — skip
        j = monomial_map[i]
        if j > i
            skip_bits[j] = true
            push!(primary_pairs, (i, j))
        end
        j ∈ (0, i) && continue           # no partner (j=0) or self-symmetric (j=i)
        @assert monomial_map[j] == i "conjugate map must be symmetric at i=$i"
        @assert !(j in linear_skip_set) "conjugate of a non-linear must not be linear"
    end

    return ConjugateSymmetryData{SVector{NVAR, Int}}(perm, monomial_map, skip_bits, primary_pairs)
end

# =============================================================================
# fill_conjugate_monomial!
# =============================================================================

"""
	fill_conjugate_monomial!(W, R, conj_idx, source_idx, sym)

Fill the conjugate monomial at `conj_idx` from the already-solved `source_idx`
using the W- and f-symmetry relations for a real-valued FOM:

	W_{P·γ} = conj(W_γ)
	f_{P·β}[r] = conj(f_β[perm[r]])   (master-mode rows only)

External rows of R at `conj_idx` are left untouched; they are already correct
either because they are zero (mixed monomials) or because `_embed_external_dynamics!`
has set them from the conjugate-symmetric external dynamics polynomial.

**Precondition**: all ORD time-derivative orders of W and master-mode rows (1:ROM)
of R at `source_idx` must be finalised before this is called.
"""
function fill_conjugate_monomial!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        conj_idx::Int,
        source_idx::Int,
        sym::ConjugateSymmetryData{SVector{NVAR, Int}}
) where {ORD, NVAR, T, ROM}
    Wc = W.poly.coefficients   # FOM × ORD × L
    Rc = R.poly.coefficients   # NVAR × L

    @inbounds @views Wc[:, :, conj_idx] .= conj.(Wc[:, :, source_idx])

    perm = sym.permutation
    @inbounds for r in 1:ROM
        Rc[r, conj_idx] = conj(Rc[perm[r], source_idx])
    end
    return nothing
end

# =============================================================================
# detect_conjugate_permutation — standalone utility
# =============================================================================

"""
	detect_conjugate_permutation(lambda; atol = 1e-8) -> Union{Vector{Int}, Nothing}

Attempt to construct a conjugate-permutation vector from the eigenvalue vector
`lambda` (length `NVAR`).  Returns a `Vector{Int}` `perm` such that

	lambda[perm[i]] ≈ conj(lambda[i])   for all i

and `perm[perm[i]] == i` (involution), or `nothing` if no such perfect pairing
exists (e.g. an eigenvalue has no conjugate partner within `atol`).

**Warning — necessary but not sufficient.**  Two eigenvalues forming a conjugate
pair does *not* guarantee that the corresponding eigenvectors satisfy

	master_modes[:, perm[r]] ≈ conj(master_modes[:, r]).

This condition can fail when:
- the eigenvalue is degenerate (eigenspace has dimension > 1),
- the solver returned a non-conjugate basis for a repeated eigenvalue,
- eigenvectors were post-processed with different phases or normalisation.

**Always verify eigenvector conjugacy** (e.g. check
`norm(master_modes[:, perm[r]] - conj(master_modes[:, r]))`) before passing the
returned vector to `solve_cohomological_problem` as `conjugate_permutation`.
Passing an incorrect permutation silently corrupts W and R.

## Arguments
- `lambda` — eigenvalue vector of length `NVAR` (master + external eigenvalues).
- `atol`   — absolute tolerance for the conjugate-match test
			  `|lambda[j] - conj(lambda[i])| < atol`.

## Returns
`Vector{Int}` (involution, 1-based) if a perfect pairing is found;
`nothing` otherwise.
"""
function detect_conjugate_permutation(lambda::AbstractVector; atol::Real = 1e-8)
    NVAR = length(lambda)
    perm = zeros(Int, NVAR)
    used = falses(NVAR)
    for i in 1:NVAR
        used[i] && continue
        λi = lambda[i]
        best_j = 0
        for j in i:NVAR
            used[j] && continue
            if abs(lambda[j] - conj(λi)) < atol
                best_j = j
                break
            end
        end
        best_j == 0 && return nothing
        perm[i] = best_j
        perm[best_j] = i
        used[i] = true
        used[best_j] = true
    end
    return perm
end
