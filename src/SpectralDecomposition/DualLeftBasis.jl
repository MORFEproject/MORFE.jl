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

In the first-order form `(𝓐, 𝓑)` of the model (never assembled: see
[`_FirstOrderForm`](@ref)), with `Φ` and `Ψ̃` the stacked blocks, these are the **dual
basis** of the left invariant subspace:

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
    adjoint pencil, at a shift just off the eigenvalue. Each step is one solve with the
    `FOM × FOM` matrix `P(s)ᴴ = (Σₖ sᵏ Bₖ)ᴴ`, factorised once per group. The iteration
    converges to the invariant subspace whatever the Jordan structure.
  - Given `conjugate_permutation` (the master involution), a group's conjugate partner is
    taken as the conjugate rather than solved again.

Both routes are accepted only if `cond(G) ≤ 1e8` and the relative left-invariance residual
is at most `_RTOL_EIGENSOLVER`, the accuracy of the eigensolver data `Φ` comes from.
"""
function master_left_basis(model::NthOrderModel, right_blocks::AbstractArray{<:Number, 3},
        Λ::AbstractMatrix, left_blocks::Union{Nothing, AbstractArray{<:Number, 3}} = nothing;
        conjugate_permutation::Union{Nothing, AbstractVector{<:Integer}} = nothing)
    left_blocks !== nothing && isdiag(Λ) && return left_blocks
    FOM, ORD, ROM = size(right_blocks)
    size(Λ) == (ROM, ROM) || throw(ArgumentError(
        "Λ is $(size(Λ, 1)) × $(size(Λ, 2)) but there are $ROM master columns"))
    istriu(Λ) || throw(ArgumentError("Λ must be upper triangular"))
    form = _FirstOrderForm(model)
    Φ = _stacked(right_blocks)
    if left_blocks !== nothing
        dual, failure = _dualise(form, Φ, Λ, _stacked(left_blocks))
        failure === nothing && return reshape(dual, FOM, ORD, ROM)
        @info "master_left_basis: the given left eigenvectors do not give the dual " *
              "basis ($failure); computing the left invariant subspace instead."
    end
    basis = _left_invariant_basis(form, Φ, Λ, conjugate_permutation)
    dual, failure = _dualise(form, Φ, Λ, basis)
    failure === nothing || throw(ArgumentError(
        "master_left_basis: no dual left basis for these master modes ($failure). " *
        "Check that the right blocks and Λ satisfy 𝓐Φ = 𝓑ΦΛ."))
    return reshape(dual, FOM, ORD, ROM)
end

# Blocks `FOM × ORD × n` as the `ORD·FOM × n` first-order vectors they stack into.
_stacked(blocks) = Matrix{ComplexF64}(reshape(blocks, :, size(blocks, 3)))

"""
	_FirstOrderForm(model)

The first-order form `(𝓐, 𝓑)` of `Σₖ Bₖ x⁽ᵏ⁾ = 0`, as products with the model's own
`(B₀, …, B_ORD)` instead of assembled `ORD·FOM × ORD·FOM` matrices
(`linear_first_order_matrices`). On stacked vectors `[x₁; …; x_ORD]`,

	𝓑 = blockdiag(I, …, I, B_ORD),    𝓐 = [shift identities; −B₀ … −B_{ORD−1}],

so every product and every shifted solve needs only `FOM × FOM` operations.
"""
struct _FirstOrderForm{LT <: Tuple}
    terms::LT                 # (B₀, …, B_ORD)
    FOM::Int
end
_FirstOrderForm(model::NthOrderModel) = _FirstOrderForm(model.linear_terms, model.n_fom)

_order(form::_FirstOrderForm) = length(form.terms) - 1
_block(form::_FirstOrderForm, X, k) = view(X, ((k - 1) * form.FOM + 1):(k * form.FOM), :)

# 𝓑X, or 𝓑ᴴX: the identity on blocks 1:ORD-1, `B_ORD` (or its adjoint) on the last one.
function _apply_B(form::_FirstOrderForm, X; adjoint::Bool = false)
    Y = Matrix{ComplexF64}(X)
    ORD, B = _order(form), last(form.terms)
    mul!(_block(form, Y, ORD), adjoint ? B' : B, _block(form, X, ORD))
    return Y
end

# 𝓐ᴴX, block by block: (𝓐ᴴX)_j = X_{j−1} − B_{j−1}ᴴ X_ORD, with X₀ = 0.
function _apply_Aᴴ(form::_FirstOrderForm, X)
    ORD = _order(form)
    Y = Matrix{ComplexF64}(undef, size(X))
    for j in 1:ORD
        Yj = _block(form, Y, j)
        mul!(Yj, form.terms[j]', _block(form, X, ORD), -1, false)
        j > 1 && (Yj .+= _block(form, X, j - 1))
    end
    return Y
end

"""
	_adjoint_shifted_solve(form, factor, s, Y) -> X

Solve `(𝓐 − s𝓑)ᴴ X = Y` with one solve by `P(s)ᴴ`, where `factor = lu(P(s))` and
`P(s) = Σₖ sᵏ Bₖ`. Block by block the system reads

	−s̄X₁ − B₀ᴴX_ORD = Y₁,    X_{j−1} − s̄X_j − B_{j−1}ᴴX_ORD = Y_j  (1 < j < ORD),
	X_{ORD−1} − (B_{ORD−1} + s B_ORD)ᴴ X_ORD = Y_ORD.

Eliminating `X₁ … X_{ORD−1}` leaves `P(s)ᴴ X_ORD = −Σᵢ s̄^{i−1} Y_i`. The other blocks then
follow downwards from the last row, `X_{ORD−1} = Y_ORD + (B_{ORD−1} + s B_ORD)ᴴ X_ORD` and
`X_{j−1} = Y_j + s̄X_j + B_{j−1}ᴴ X_ORD`. That direction never divides by `s̄`: substituting
upwards from the first row would, and near `s = 0` (a rigid-body master) the cancellation
in `Y₁ + B₀ᴴX_ORD` would be amplified by `1/|s|`.
"""
function _adjoint_shifted_solve(form::_FirstOrderForm, factor, s, Y)
    ORD = _order(form)
    s̄ = conj(s)
    rhs = -_block(form, Y, ORD)              # −Σᵢ s̄^{i−1} Y_i, by Horner
    for i in (ORD - 1):-1:1
        rhs = s̄ .* rhs .- _block(form, Y, i)
    end
    X = Matrix{ComplexF64}(undef, size(Y))
    u = _block(form, X, ORD)
    u .= factor' \ rhs
    if ORD > 1
        below = _block(form, X, ORD - 1)
        below .= _block(form, Y, ORD)
        mul!(below, form.terms[ORD]', u, 1, 1)
        mul!(below, form.terms[ORD + 1]', u, s̄, 1)
        for j in (ORD - 1):-1:2
            next = _block(form, X, j - 1)
            next .= _block(form, Y, j) .+ s̄ .* _block(form, X, j)
            mul!(next, form.terms[j]', u, 1, 1)
        end
    end
    return X
end

# `P(s) = Σₖ sᵏ Bₖ`, the matrix the bordered solver factorises at a superharmonic.
function _polynomial_matrix(form::_FirstOrderForm, s)
    sum(
        s^(k - 1) * form.terms[k] for k in eachindex(form.terms))
end

"""
	_dualise(form, Φ, Λ, Ψ) -> (Ψ̃, nothing) or (nothing, reason)

The dual basis `Ψ̃ = Ψ G⁻ᴴ`, `G = Ψᴴ𝓑Φ`, of a left basis `Ψ`, after checking that `G` is
well conditioned and that `Ψ̃` is left-invariant with `Λ`. A failure returns its reason
instead, so a caller can fall back to another route.
"""
function _dualise(form::_FirstOrderForm, Φ, Λ, Ψ)
    G = Ψ' * _apply_B(form, Φ)
    condition = cond(G)
    if condition > 1e8
        return nothing,
        "cond(Ψᴴ𝓑Φ) = $condition, a defective eigenvalue or a deficient basis"
    end
    dual = Ψ / G'
    left_A = _apply_Aᴴ(form, dual)'
    left_B = Λ * _apply_B(form, dual; adjoint = true)'
    residual = norm(left_A - left_B) / (norm(left_A) + norm(left_B))
    residual <= _RTOL_EIGENSOLVER || return nothing, "left-invariance residual $residual"
    return dual, nothing
end

"""
	_left_invariant_basis(form, Φ, Λ, σ) -> Matrix{ComplexF64}

A basis of the left invariant subspace of the master eigenvalues `diag(Λ)`, one eigenvalue
group at a time, with each group's columns in that group's positions. With an involution
`σ`, a group whose conjugate partners form another group fills that one by conjugation.
"""
function _left_invariant_basis(form::_FirstOrderForm, Φ, Λ, σ)
    λ = diag(Λ)
    groups = _eigenvalue_groups(λ)
    Ψ = Matrix{ComplexF64}(undef, size(Φ))
    filled = falses(length(λ))
    for members in groups
        filled[first(members)] && continue
        basis = _group_left_basis(form, Φ[:, members], λ[first(members)])
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
	_group_left_basis(form, Φ_group, λ; max_iterations = 10) -> Matrix{ComplexF64}

An orthonormal basis of the left generalised eigenspace of `λ`, by block inverse iteration
on the adjoint pencil, `X ← (𝓐 − s𝓑)⁻ᴴ 𝓑ᴴ X` ([`_adjoint_shifted_solve`](@ref)) with
`s = λ + 10⁻⁴·max(1, |λ|)`, kept off zero for rigid-body masters. The start `𝓑Φ_group` has a
non-singular component along that subspace (its Gram matrix), and the iteration converges
to the whole invariant subspace at the rate `|s − λ| / gap`, so a Jordan block needs no
chains and no rank decisions.
"""
function _group_left_basis(form::_FirstOrderForm, Φ_group, λ; max_iterations::Int = 10)
    shift = λ + 1e-4 * max(1, abs(λ))
    factor = lu(_polynomial_matrix(form, shift))
    X = Matrix(qr(_apply_B(form, Φ_group)).Q)
    for _ in 1:max_iterations
        next = Matrix(qr(_adjoint_shifted_solve(
            form, factor, shift, _apply_B(form, X; adjoint = true))).Q)
        change = norm(next - X * (X' * next))
        X = next
        change <= 1e-12 && break
    end
    return X
end
