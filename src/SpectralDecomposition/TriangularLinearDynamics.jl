# =============================================================================
# Upper-triangular master linear dynamics — the right blocks that follow it, the
# conjugate frame the solve's complex symmetry requires, and how SpectralData takes it
# =============================================================================

"""
	master_right_blocks(modes, Λ, ORD) -> Array{ComplexF64, 3}

The right order-blocks, `FOM × ORD × ROM`, of master coordinates whose physical columns are
`modes` (`FOM × ROM`) and whose reduced linear dynamics is the upper-triangular `Λ`
(`ROM × ROM`). Block 1 is `modes`, and each further block is the previous one times `Λ`:

	Φ⁽ᵏ⁺¹⁾ = Φ⁽ᵏ⁾ Λ,    i.e.  Φ⁽ᵏ⁺¹⁾[:, r] = Λ[r, r] Φ⁽ᵏ⁾[:, r] + Σ_{i<r} Λ[i, r] Φ⁽ᵏ⁾[:, i].

This is the time derivative of `Φ z` along `ż = Λ z`, and the identity the solve assumes of
the stored levels of `W`. For a diagonal `Λ` it reduces to the eigenvector convention
`ψ_{k+1} = λ ψ_k`, which is how `SpectralData` extends the blocks of a lower-order
eigenproblem.

Use it to build Schur- or Jordan-coordinate inputs: the physical columns and `Λ` are the
caller's choice, and the derivative blocks follow.
"""
function master_right_blocks(modes::AbstractMatrix, Λ::AbstractMatrix, ORD::Integer)
    FOM, ROM = size(modes)
    size(Λ) == (ROM, ROM) || throw(ArgumentError(
        "Λ is $(size(Λ, 1)) × $(size(Λ, 2)) but there are $ROM master columns"))
    istriu(Λ) || throw(ArgumentError(
        "Λ must be upper triangular; the solve reads only its upper triangle"))
    ORD >= 1 || throw(ArgumentError("ORD must be at least 1, got $ORD"))
    blocks = Array{ComplexF64, 3}(undef, FOM, ORD, ROM)
    blocks[:, 1, :] .= modes
    return _extend_right_blocks!(blocks, Λ, 1)
end

"""
	_extend_right_blocks!(blocks, Λ, from) -> blocks

Overwrite blocks `from + 1 : ORD` of `blocks` with `Φ⁽ᵏ⁺¹⁾ = Φ⁽ᵏ⁾ Λ`, reading only the
upper triangle of `Λ`. Zero couplings are skipped, so a diagonal `Λ` performs exactly the
broadcast `λ .* block` and nothing else.
"""
function _extend_right_blocks!(blocks::AbstractArray{<:Number, 3}, Λ::AbstractMatrix,
        from::Int)
    for k in (from + 1):size(blocks, 2), r in axes(blocks, 3)

        @views blocks[:, k, r] .= Λ[r, r] .* blocks[:, k - 1, r]
        for i in 1:(r - 1)
            coupling = Λ[i, r]
            iszero(coupling) && continue
            @views blocks[:, k, r] .+= coupling .* blocks[:, k - 1, i]
        end
    end
    return blocks
end

"""
	conjugate_master_frame(blocks, Λ, σ; rtol = _RTOL_EXACT) -> (blocks, Λ, T)

Re-base master coordinates onto the **conjugate frame** the solve's complex symmetry needs,
keeping `Λ` upper triangular with the same diagonal.

`blocks` are the right order-blocks (`FOM × ORD × ROM`), `Λ` is the upper-triangular master
block of the reduced linear dynamics, and `σ` is the master conjugate involution: `r < σ[r]`
is a primary, `σ[r]` its secondary, and `σ[r] = r` a real mode.

## Why

Conjugate filling copies `W[σγ] = conj(W[γ])`. That is exact only when the master columns
come in conjugate pairs, `Φ[:, σ] = conj(Φ)` on every block, and then `Λ[σ, σ] = conj(Λ)`
for a real system. In such a frame a pair cannot couple to itself: an entry `Λ[r, σr]` would
force `conj(Λ[r, σr])` at `Λ[σr, r]`, below the diagonal. Columns from a complex Schur
decomposition are not conjugate pairs, and that is exactly when a pair couples to itself.

## What it does

Each secondary column is replaced by the conjugate of its primary, `Φ̂ = Φ T`, so `T`
differs from the identity only in the secondary columns, and

	Λ̂ = T⁻¹ Λ T

is again upper triangular, with the same diagonal (the eigenvalues), conjugate-consistent,
and free of couplings inside a pair. The secondary columns are computed by least squares
on the stacked blocks, then accepted only if

- the conjugates lie in the span of `Φ` (the master subspace is closed under conjugation);
- `T` is upper triangular (each leading span of the coordinates is closed under
  conjugation, which holds whenever the eigenvalues are distinct);
- `T` is well conditioned (a pair close to the real axis has `φᵣ ∥ conj(φᵣ)`; there the
  eigenvalues are real and the two modes should be real modes, `σ[r] = r`).

Returns the re-based blocks, `Λ̂` and `T`. When the frame is already conjugate, the inputs
are returned **unchanged** (the same objects) with `T = I`, so no rounding is introduced.
Real modes must have real columns; there is nothing to re-base them onto.

Map a result back to the given coordinates with `z = T ẑ`: `W ∘ T⁻¹` and `T · (R ∘ T⁻¹)`,
e.g. via `compose_linear` with `T⁻¹` extended by the identity on the external variables.
"""
function conjugate_master_frame(blocks::AbstractArray{<:Number, 3}, Λ::AbstractMatrix,
        σ::AbstractVector{<:Integer}; rtol::Real = _RTOL_EXACT)
    FOM, ORD, ROM = size(blocks)
    size(Λ) == (ROM, ROM) || throw(ArgumentError(
        "Λ is $(size(Λ, 1)) × $(size(Λ, 2)) but there are $ROM master columns"))
    istriu(Λ) || throw(ArgumentError("Λ must be upper triangular"))
    length(σ) == ROM && all(r -> 1 <= σ[r] <= ROM && σ[σ[r]] == r, 1:ROM) ||
        throw(ArgumentError("σ must be an involution of 1:$ROM, got $(collect(σ))"))

    Φ = reshape(blocks, FOM * ORD, ROM)
    for r in 1:ROM
        σ[r] == r || continue
        norm(imag.(view(Φ, :, r))) <= rtol * norm(view(Φ, :, r)) || throw(ArgumentError(
            "master column $r is a real mode (σ[$r] = $r) but its blocks are not real; " *
            "pass a real basis for real eigenvalues"))
    end
    secondaries = [s for s in 1:ROM if σ[s] < s]
    conjugate(s) = conj.(view(Φ, :, σ[s]))
    all(s -> norm(view(Φ, :, s) - conjugate(s)) <= rtol * norm(view(Φ, :, s)), secondaries) &&
        return blocks, Λ, Matrix{ComplexF64}(I, ROM, ROM)

    target = Matrix{ComplexF64}(Φ)
    for s in secondaries
        target[:, s] .= conjugate(s)
    end
    factor = qr(Matrix{ComplexF64}(Φ))
    T = Matrix{ComplexF64}(I, ROM, ROM)
    for s in secondaries
        column = factor \ target[:, s]
        residual = norm(Φ * column - view(target, :, s))
        residual <= rtol * norm(view(target, :, s)) || throw(ArgumentError(
            "the conjugate of master column $(σ[s]) is not in the span of the master " *
            "columns (relative residual $(residual / norm(view(target, :, s)))); the " *
            "master subspace must be closed under conjugation"))
        norm(view(column, (s + 1):ROM)) <= rtol * norm(column) || throw(ArgumentError(
            "the conjugate of master column $(σ[s]) needs the later columns " *
            "$(collect((s + 1):ROM)), so no upper-triangular change of coordinates makes " *
            "the frame conjugate; order the master coordinates so that each conjugate " *
            "follows the columns it is built from"))
        T[1:s, s] .= view(column, 1:s)
    end
    condition = cond(T)
    condition <= 1e8 || throw(ArgumentError(
        "re-basing onto the conjugate frame is ill-conditioned (cond(T) = $condition): a " *
        "conjugate pair is close to the real axis. There the eigenvalues are real; make " *
        "those modes real (σ[r] = r) with a real basis"))

    Λ̂ = _conjugate_consistent(UpperTriangular(T) \ (Matrix{ComplexF64}(Λ) * T), Λ, σ)
    return reshape(target, FOM, ORD, ROM), Λ̂, T
end

"""
	_conjugate_consistent(M, Λ, σ) -> Matrix{ComplexF64}

Make the rounding-level structure of `M = T⁻¹ΛT` exact: the strict lower triangle is zero,
the diagonal is `Λ`'s (the eigenvalues, so they keep a single source), the entries inside a
conjugate pair are zero, and each entry's conjugate partner `M[σi, σj]` is the conjugate of
the one earlier in column-major order, the convention of the monomial fill. An entry that
breaks this beyond rounding means the coupling does not belong to these modes, and throws.
"""
function _conjugate_consistent(M::AbstractMatrix, Λ::AbstractMatrix,
        σ::AbstractVector{<:Integer})
    ROM = size(M, 1)
    scale = max(1.0, opnorm(M, 1))
    out = triu(Matrix{ComplexF64}(M))
    for j in 1:ROM, i in 1:(j - 1)

        partner = (σ[i], σ[j])
        if partner[1] > partner[2]          # includes a pair coupled to itself
            abs(out[i, j]) <= _RTOL_EIGENSOLVER * scale || throw(ArgumentError(
                "the master coupling is not consistent with the conjugate structure: " *
                "Λ̂[$i, $j] = $(out[i, j]) has its conjugate partner below the diagonal"))
            out[i, j] = 0
        elseif LinearIndices(out)[partner...] > LinearIndices(out)[i, j]
            out[partner...] = conj(out[i, j])
        elseif partner == (i, j)
            out[i, j] = real(out[i, j])
        end
    end
    for r in 1:ROM
        out[r, r] = Λ[r, r]
    end
    return out
end

"""
	_master_dynamics(diagonal, upper_triangular) -> Union{Diagonal, UpperTriangular}

The master block of the reduced linear dynamics from the keyword `SpectralData(; …)` was
given: `diagonal` (the eigenvalues) makes a `Diagonal`, `upper_triangular` an
`UpperTriangular` — or a `Diagonal` when it has nothing above the diagonal.
"""
_master_dynamics(diagonal::AbstractVector, ::Nothing) = _master_block(Diagonal(diagonal))
function _master_dynamics(::Nothing, upper_triangular::AbstractMatrix)
    size(upper_triangular, 1) == size(upper_triangular, 2) && istriu(upper_triangular) ||
        throw(ArgumentError("upper_triangular must be a square upper-triangular matrix"))
    return _master_block(upper_triangular)
end
function _master_dynamics(::Nothing, ::Nothing)
    throw(ArgumentError(
        "give the master block: `diagonal` (the eigenvalues) or `upper_triangular`"))
end
function _master_dynamics(::AbstractVector, ::AbstractMatrix)
    throw(ArgumentError(
        "give either `diagonal` or `upper_triangular`, not both"))
end

# The stored form of a master block: `Diagonal` exactly when nothing is above the diagonal.
function _master_block(Λ::AbstractMatrix)
    ROM = size(Λ, 1)
    isdiag(Λ) && return Diagonal(SVector{ROM, ComplexF64}(diag(Λ)))
    return UpperTriangular(Matrix{ComplexF64}(Λ))
end

"""
	_conjugate_master_columns(blocks, Λ, σ) -> (blocks, Λ)

With an upper-triangular master block `Λ` and a master involution `σ`, put the master
columns in conjugate pairs with [`conjugate_master_frame`](@ref), so that the solve's
conjugate filling is exact. Columns already in pairs come back unchanged; otherwise one
`@info` line names the replaced ones. A diagonal block, or no involution, leaves everything
as it is.
"""
_conjugate_master_columns(blocks, Λ::Diagonal, σ) = blocks, Λ
_conjugate_master_columns(blocks, Λ::UpperTriangular, ::Nothing) = blocks, Λ
function _conjugate_master_columns(blocks, Λ::UpperTriangular, σ::AbstractVector)
    frame, Λ̂, _ = conjugate_master_frame(blocks, Λ, σ)
    frame === blocks && return blocks, Λ
    @info "SpectralData: the master columns $([s for s in eachindex(σ) if σ[s] < s]) " *
          "were not the conjugates of their partners, so they were replaced by them, and " *
          "the master block re-based to T⁻¹Λ̃T (same diagonal). `conjugate_master_frame` " *
          "returns T."
    return frame, _master_block(Λ̂)
end
