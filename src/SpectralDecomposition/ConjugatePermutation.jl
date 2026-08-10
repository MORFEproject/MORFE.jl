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

# =============================================================================
# External conjugate structure
# =============================================================================

"""
	external_conjugate_permutation(sys; atol = 1e-8) -> Union{Vector{Int}, Nothing}

The conjugate involution `σ` on the `N_EXT` external variables, or `nothing` when the
external system has no conjugate structure to offer.

`σ` pairs external variable `k` with the one carrying `conj(λ_k)`, and — when the system
was re-based — additionally guarantees `Q[:, σ(k)] == conj(Q[:, k])`.  Both conditions are
needed for the reduction's conjugate symmetry: `Realification.realify` applies one
`conj_map` across all `NVAR` variables, external ones included, and
[`fill_conjugate_monomial!`](@ref) implements `W_{P·γ} = conj(W_γ)`.  With a real forcing
`F` the external columns satisfy `Φ[:, k] = L(λ_k)⁻¹ F Q[:, k]`, so
`conj(Φ[:, k]) = Φ[:, σ(k)]` follows exactly from the two conditions together.

Eigenvalue pairing alone is *not* sufficient — see [`detect_conjugate_permutation`](@ref)'s
own warning — which is why the basis columns are verified rather than assumed.  A system
re-based onto a Schur basis generally fails that check and correctly returns `nothing`: its
external variables simply are not conjugate pairs.

Use [`full_conjugate_permutation`](@ref) to assemble the full `NVAR` permutation.
"""
function external_conjugate_permutation(sys::ExternalSystem; atol::Real = 1e-8)
    σ = detect_conjugate_permutation(collect(sys.eigenvalues); atol = atol)
    σ === nothing && return nothing

    Q = external_basis(sys)
    Q === nothing && return σ           # untouched coordinates: the spectrum is all there is

    # Verify the columns, do not trust the spectrum.  The eigen route satisfies this
    # bit-exactly; a Schur basis generally does not.
    for k in eachindex(σ)
        isapprox(Q[:, σ[k]], conj(Q[:, k]); atol = atol) || return nothing
    end
    return σ
end

external_conjugate_permutation(::Nothing; atol::Real = 1e-8) = Int[]

"""
	full_conjugate_permutation(master_perm, sys) -> Vector{Int}

Assemble the full `NVAR`-length `conjugate_permutation` from the master block and the
external system, appending `ROM .+ σ` where `σ` is the external involution.

Callers otherwise hand-write the whole vector (`[2, 1, 3]`, `[2, 1, 4, 3]`, …), which bakes
in a pairing that a change of external coordinates can invalidate, and which has to special
-case an odd number of external variables. Deriving the external block instead keeps it
correct in both situations.

Throws when the external system has no conjugate structure — see
[`external_conjugate_permutation`](@ref).
"""
function full_conjugate_permutation(
        master_perm::AbstractVector{Int}, sys::Union{Nothing, ExternalSystem})
    ROM = length(master_perm)
    σ = external_conjugate_permutation(sys)
    σ === nothing && throw(ArgumentError("""
       The external system has no conjugate structure, so no conjugate permutation covers \
       its variables: either its spectrum is not closed under conjugation, or it was \
       re-based onto a basis whose columns are not conjugate pairs (the Schur route).
       Solve without `conjugate_permutation`, or supply an external system whose linear \
       matrix is real (which takes the conjugate-preserving eigenvector route).
       """))
    return vcat(collect(master_perm), ROM .+ σ)
end
