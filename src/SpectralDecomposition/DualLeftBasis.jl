# =============================================================================
# The dual left basis of the master modes
# =============================================================================

"""
	master_left_basis(model, right_blocks, Λ, left_blocks = nothing;
		conjugate_permutation = nothing) -> Array{ComplexF64, 3}

The left order-blocks the orthogonality rows of the solve need, for master coordinates with
right blocks `right_blocks` (`FOM × ORD × ROM`) and upper-triangular master block `Λ`. The
result has the layout `SpectralData` stores left blocks in (`FOM × ORD × ROM`, physical
slice **last**).

In the first-order form `(𝓐, 𝓑) = linear_first_order_matrices(model)`, with `Φ` and `Ψ̃` the
stacked blocks, these are the **dual basis** of the left invariant subspace:

	Ψ̃ᴴ 𝓑 Φ = I,    Ψ̃ᴴ 𝓐 = Λ Ψ̃ᴴ 𝓑.

Any basis `Ψ` of that subspace gives it as `Ψ̃ = Ψ G⁻ᴴ` with `G = Ψᴴ 𝓑 Φ`, whatever basis it
is, so everything reduces to finding the subspace.

- **Diagonal `Λ`, `left_blocks` given.** The left eigenvectors are this basis already, up
  to a per-mode scale the homogeneous orthogonality rows never see. They are returned as
  they are, and nothing is computed.
- **Route A: dualise `left_blocks`.** Used when `Λ` is not diagonal. It needs one
  `ROM × ROM` solve, and fails when a master eigenvalue is defective: a Jordan block has a
  single left eigenvector for several coordinates, so `G` is singular. It also fails when
  the given vectors do not span the subspace.
- **Route B: compute the subspace.** Used when route A fails or no `left_blocks` are given,
  and works for any eigenvalues, Jordan blocks included.
  - The master indices are grouped by eigenvalue.
  - Each group's left generalised eigenspace comes from block inverse iteration on the
    adjoint pencil: one LU per group, at a shift just off the eigenvalue. The iteration
    converges to the invariant subspace whatever the Jordan structure.
  - Given `conjugate_permutation` (the master involution), a group's conjugate partner is
    taken as the conjugate rather than solved again.

Both routes are accepted only if `cond(G) ≤ 1e8` and the relative left-invariance residual
is at most `1e-8`, the accuracy of the eigensolver data `Φ` comes from.
"""
function master_left_basis(model::NthOrderModel, right_blocks::AbstractArray{<:Number, 3},
        Λ::AbstractMatrix, left_blocks::Union{Nothing, AbstractArray{<:Number, 3}} = nothing;
        conjugate_permutation::Union{Nothing, AbstractVector{<:Integer}} = nothing)
    left_blocks !== nothing && isdiag(Λ) && return left_blocks
    FOM, ORD, ROM = size(right_blocks)
    size(Λ) == (ROM, ROM) || throw(ArgumentError(
        "Λ is $(size(Λ, 1)) × $(size(Λ, 2)) but there are $ROM master columns"))
    istriu(Λ) || throw(ArgumentError("Λ must be upper triangular"))
    A, B = linear_first_order_matrices(model)
    Φ = Matrix{ComplexF64}(reshape(right_blocks, FOM * ORD, ROM))
    if left_blocks !== nothing
        given = Matrix{ComplexF64}(reshape(left_blocks, FOM * ORD, ROM))
        dual, failure = _dualise(A, B, Φ, Λ, given)
        failure === nothing && return reshape(dual, FOM, ORD, ROM)
        @info "master_left_basis: the given left eigenvectors do not give the dual " *
              "basis ($failure); computing the left invariant subspace instead."
    end
    basis = _left_invariant_basis(A, B, Φ, Λ, conjugate_permutation)
    dual, failure = _dualise(A, B, Φ, Λ, basis)
    failure === nothing || throw(ArgumentError(
        "master_left_basis: no dual left basis for these master modes ($failure). " *
        "Check that the right blocks and Λ satisfy 𝓐Φ = 𝓑ΦΛ."))
    return reshape(dual, FOM, ORD, ROM)
end

"""
	_dualise(A, B, Φ, Λ, Ψ) -> (Ψ̃, nothing) or (nothing, reason)

The dual basis `Ψ̃ = Ψ G⁻ᴴ`, `G = Ψᴴ𝓑Φ`, of a left basis `Ψ`, after checking that `G` is
well conditioned and that `Ψ̃` is left-invariant with `Λ`. A failure returns its reason
instead, so a caller can fall back to another route.
"""
function _dualise(A, B, Φ, Λ, Ψ)
    G = Ψ' * (B * Φ)
    condition = cond(G)
    if condition > 1e8
        return nothing,
        "cond(Ψᴴ𝓑Φ) = $condition, a defective eigenvalue or a deficient basis"
    end
    dual = Ψ / G'
    left_A = (A' * dual)'
    left_B = Λ * (B' * dual)'
    residual = norm(left_A - left_B) / (norm(left_A) + norm(left_B))
    residual <= 1e-8 || return nothing, "left-invariance residual $residual"
    return dual, nothing
end

"""
	_left_invariant_basis(A, B, Φ, Λ, σ) -> Matrix{ComplexF64}

A basis of the left invariant subspace of the master eigenvalues `diag(Λ)`, one eigenvalue
group at a time, with each group's columns in that group's positions. With an involution
`σ`, a group whose conjugate partners form another group fills that one by conjugation.
"""
function _left_invariant_basis(A, B, Φ, Λ, σ)
    λ = diag(Λ)
    groups = _eigenvalue_groups(λ)
    Ψ = Matrix{ComplexF64}(undef, size(Φ))
    filled = falses(length(λ))
    for members in groups
        filled[first(members)] && continue
        basis = _group_left_basis(A, B, Φ[:, members], λ[first(members)])
        Ψ[:, members] .= basis
        filled[members] .= true
        σ === nothing && continue
        partners = σ[members]
        any(filled[partners]) && continue                   # a real group pairs with itself
        partner_group = groups[findfirst(g -> partners[1] in g, groups)]
        sort(partners) == sort(partner_group) || continue
        Ψ[:, partners] .= conj.(basis)
        filled[partners] .= true
    end
    return Ψ
end

"""
	_eigenvalue_groups(λ) -> Vector{Vector{Int}}

Group indices whose eigenvalues agree to `1e-8·max(1, |λ|)`, the relative form of the
tolerance `detect_conjugate_permutation` pairs eigenvalues with. A group's size is the
algebraic multiplicity the left basis must span.
"""
function _eigenvalue_groups(λ::AbstractVector)
    group_of = zeros(Int, length(λ))
    groups = Vector{Vector{Int}}()
    for i in eachindex(λ)
        group_of[i] == 0 || continue
        members = [j
                   for j in i:length(λ)
                   if group_of[j] == 0 && abs(λ[j] - λ[i]) <= 1e-8 * max(1, abs(λ[i]))]
        group_of[members] .= length(groups) + 1
        push!(groups, members)
    end
    return groups
end

"""
	_group_left_basis(A, B, Φ_group, λ; max_iterations = 10) -> Matrix{ComplexF64}

An orthonormal basis of the left generalised eigenspace of `λ`, by block inverse iteration
on the adjoint pencil, `X ← (𝓐 − s𝓑)⁻ᴴ 𝓑ᴴ X` with `s = λ + 10⁻⁶·max(1, |λ|)`. The start
`𝓑Φ_group` has a non-singular component along that subspace (its Gram matrix), and the
iteration converges to the whole invariant subspace at the rate `|s − λ| / gap`, so a Jordan
block needs no chains and no rank decisions.
"""
function _group_left_basis(A, B, Φ_group, λ; max_iterations::Int = 10)
    shift = λ + 1e-6 * max(1, abs(λ))
    factor = lu(_adjoint_pencil(A, B, shift))
    X = Matrix(qr(B * Φ_group).Q)
    for _ in 1:max_iterations
        next = Matrix(qr(factor \ (B' * X)).Q)
        change = norm(next - X * (X' * next))
        X = next
        change <= 1e-12 && break
    end
    return X
end

_adjoint_pencil(A::SparseMatrixCSC, B, s) = sparse((A - s * B)')
_adjoint_pencil(A::AbstractMatrix, B, s) = Matrix((A - s * B)')
