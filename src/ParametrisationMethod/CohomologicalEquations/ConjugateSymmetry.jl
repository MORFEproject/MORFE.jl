# =============================================================================
# Conjugate-symmetry exploitation for real-valued FOMs
# =============================================================================
# For a real-valued FOM the parametrisation and reduced dynamics satisfy
#   W_{P·γ} = conj(W_γ)           [W-symmetry]
#   f_{P·β}[r] = conj(f_β[perm[r]])  [f-symmetry]
# where P is the involutory permutation on mode indices that swaps conjugate pairs.
# This module encapsulates the detection, bookkeeping, and fill logic so that
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
# RealArithmeticBuffers — Float64 scratch for self-conjugate monomials
# =============================================================================

"""
    RealArithmeticBuffers

Pre-allocated Float64 buffers for the real-arithmetic dense solve activated on
self-conjugate monomials when the FOM matrices are real-valued.  Solving in
Float64 instead of ComplexF64 gives ≈4× cheaper LU factorisation.
"""
struct RealArithmeticBuffers
    system::Matrix{Float64}
    rhs::Vector{Float64}
end

RealArithmeticBuffers(FOM::Int, ROM::Int) =
    RealArithmeticBuffers(Matrix{Float64}(undef, FOM + ROM, FOM + ROM),
                          Vector{Float64}(undef, FOM + ROM))


# =============================================================================
# ConjugateSymmetryData{CP, RB}
# =============================================================================

"""
    ConjugateSymmetryData{CP, RB}

Self-contained optimisation layer for exploiting complex-conjugate symmetry in
the cohomological solve.

# Type parameters

| Parameter | Meaning |
|:----------|:--------|
| `CP` | `NoConjugatePermutation` (inactive) or `SVector{NVAR, Int}` (active) |
| `RB` | `Nothing` (complex FOM or inactive) or `RealArithmeticBuffers` (real FOM + active) |

Compile-time dispatch on `CP` eliminates secondary-monomial bookkeeping when
inactive.  Compile-time dispatch on `RB` eliminates the real-arithmetic branch
when not applicable.  The only remaining runtime check is `monomial_map[idx] == idx`
(single integer comparison, < 1 ns), active only when `RB = RealArithmeticBuffers`.
"""
struct ConjugateSymmetryData{CP, RB}
    permutation::CP
    monomial_map::Vector{Int}
    skip_bits::BitVector    # length L; true = skip this monomial (linears + secondaries)
    real_buffers::RB
end

# =============================================================================
# _build_monomial_map
# =============================================================================

function _build_monomial_map(
        mset::MultiindexSet{NVAR},
        perm::SVector{NVAR, Int},
        mdict::Dict{SVector{NVAR, Int}, Int}
) where {NVAR}
    L = length(mset)
    monomial_map = Vector{Int}(undef, L)

    for i in 1:L
        γ = mset[i]

        # Any active component on an unpaired mode → no conjugate symmetry here
        has_unpaired = any(k -> γ[k] > 0 && perm[k] == 0, 1:NVAR)
        if has_unpaired
            monomial_map[i] = 0
            continue
        end

        # perm[k] == 0 means mode k is unpaired; γ[k] must be 0 here (checked above),
        # so we treat P as the identity on that component: Pγ[k] = γ[k].
        Pγ = SVector{NVAR, Int}(ntuple(k -> perm[k] == 0 ? γ[k] : γ[perm[k]], Val(NVAR)))

        monomial_map[i] = Pγ == γ ? i : get(mdict, Pγ, 0)
    end
    return monomial_map
end

# =============================================================================
# _make_real_buffers — dispatch on FOM element type, matrix type, and solver
# =============================================================================

# Default: no real-arithmetic buffers.
_make_real_buffers(::Type{LT}, ::Type{MT}, _, FOM::Int, ROM::Int, ORD::Int) where {LT, MT} = nothing
# Dense real path: real LU buffers (sparse_solver is nothing on the dense path).
_make_real_buffers(::Type{<:Real}, ::Type{<:Matrix}, ::Nothing, FOM::Int, ROM::Int, ORD::Int) =
    RealArithmeticBuffers(FOM, ROM)
# Sparse real path: disabled. A separate KLU symbolic factorisation for L_real would not
# reuse the complex klu_cache symbolic, and accessing K_cplx.p/.q triggers klu_z_sort
# (O(nnz log nnz)) -- more expensive than the saved ComplexF64 arithmetic. The conjugate
# speedup on the sparse path comes entirely from skipping secondary monomials; self-conjugate
# monomials remain on the complex KLU path (numeric refactorisation reuses the cached symbolic).
_make_real_buffers(::Type{<:Real}, ::Type{<:SparseMatrixCSC}, ::SparseLinearSolverState,
        FOM::Int, ROM::Int, ORD::Int) = nothing

# =============================================================================
# _build_conjugate_symmetry — factory
# =============================================================================

# Inactive path: wrap the linear skip set as a BitVector, allocate nothing.
function _build_conjugate_symmetry(::NoConjugatePermutation, linear_skip_set::Set{Int}, L::Int)
    skip_bits = falses(L)
    for i in linear_skip_set
        skip_bits[i] = true
    end
    return ConjugateSymmetryData{NoConjugatePermutation, Nothing}(
        NoConjugatePermutation(), Int[], skip_bits, nothing
    )
end

# Active path: build monomial map, augment skip bits, conditionally allocate real buffers.
function _build_conjugate_symmetry(
        perm::SVector{NVAR, Int},
        linear_skip_set::Set{Int},
        mset::MultiindexSet{NVAR},
        mdict::Dict{SVector{NVAR, Int}, Int},
        FOM::Int,
        ROM::Int,
        ORD::Int,
        ::Type{LT},
        ::Type{MT},
        sparse_solver
) where {NVAR, LT, MT}
    monomial_map = _build_monomial_map(mset, perm, mdict)

    skip_bits = falses(length(mset))
    for i in linear_skip_set
        skip_bits[i] = true
    end
    for i in eachindex(monomial_map)
        j = monomial_map[i]
        j > i && (skip_bits[j] = true)
        if j ∉ (0, i) && !(i in linear_skip_set)
            @assert monomial_map[j] == i "conjugate map must be symmetric at i=$i"
            @assert !(j in linear_skip_set) "conjugate of a non-linear must not be linear"
        end
    end

    real_buffers = _make_real_buffers(LT, MT, sparse_solver, FOM, ROM, ORD)
    RB = typeof(real_buffers)
    return ConjugateSymmetryData{SVector{NVAR, Int}, RB}(
        perm, monomial_map, skip_bits, real_buffers
    )
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
        sym::ConjugateSymmetryData{SVector{NVAR, Int}, RB}
) where {ORD, NVAR, T, ROM, RB}
    Wc = W.poly.coefficients   # FOM × ORD × L
    Rc = R.poly.coefficients   # NVAR × L

    for j in 1:ORD
        @views Wc[:, j, conj_idx] .= conj.(Wc[:, j, source_idx])
    end

    perm = sym.permutation
    for r in 1:ROM
        pr = perm[r]
        Rc[r, conj_idx] = conj(Rc[pr, source_idx])
    end
    return nothing
end
