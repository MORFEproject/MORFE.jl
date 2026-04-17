module PropagateEigenmodes

using LinearAlgebra
using ..FullOrderModel
using ..ParametrisationMethod: Parametrisation

export propagate_left_eigenvector_from_last, propagate_left_jordan_vector,
	propagate_right_eigenvector_from_first, propagate_right_jordan_vector

"""
	propagate_left_eigenvector_from_last(model, eigenvectors, x_last, λ, index)

Fills the complete left eigenvector from the ORD‑th component `x_last`.
Uses the recurrence (without explicit transposes) to avoid allocations.
"""
function propagate_left_eigenvector_from_last(
	model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT},
	eigenvectors::Array{T2, 3},
	x_last::Vector{T2},
	λ::Number,
	index::Int,
) where {ORD, ORDP1, N_NL, N_EXT, T, T2, MT}
	linear_terms = model.linear_terms
	fom = size(eigenvectors, 1)

	@assert ORD >= 1 "ORD needs to be at least 1"
	@assert length(x_last) == fom "Vector x_last must have length $fom"
	@assert size(eigenvectors, 2) == ORD "eigenvectors[:,:,index] must have ORD columns"

	# Set the last component
	eigenvectors[:, ORD, index] .= x_last

	# Pre‑allocate a temporary vector for the adjoint products
	tmp = similar(x_last)

	@inbounds for j in (ORD-1):-1:1
		# tmp = linear_terms[j+1]' * x_last   (adjoint * vector)
		mul!(tmp, adjoint(linear_terms[j+1]), x_last)
		# eigenvectors[:, j, index] = λ * eigenvectors[:, j+1, index] + tmp
		@views eigenvectors[:, j, index] .= λ .* eigenvectors[:, j+1, index] .+ tmp
	end
end

"""
	propagate_left_jordan_vector(model, eigenvectors, λ, index)

Fills the k‑th left Jordan vector (index) from the (k‑1)‑th vector (index‑1).
Avoids temporary transposes and fixes a missing operator.
"""
function propagate_left_jordan_vector(
	model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT},
	eigenvectors::Array{T2, 3},
	λ::Number,
	index::Int,
) where {ORD, ORDP1, N_NL, N_EXT, T, T2, MT}
	linear_terms = model.linear_terms
	fom = size(eigenvectors, 1)

	# Build the matrix polynomial P(λ) = B₁ + λ B₂ + ... + λ^ORD B_{ORD+1}
	tmp_mat = copy(linear_terms[1])
	λ_pow = one(λ)
	for i in 2:ORDP1
		λ_pow *= λ
		tmp_mat .+= λ_pow * linear_terms[i]
	end

	# Compute the right‑hand side sum of previous Jordan vector components
	tmp_vec = sum(view(eigenvectors, :, j, index-1) for j in 1:ORD)

	x_last = tmp_mat \ tmp_vec
	@assert length(x_last) == fom "Vector x_last must have length $fom"
	@assert size(eigenvectors, 2) == ORD "eigenvectors[:,:,index] must have ORD columns"

	eigenvectors[:, ORD, index] .= x_last

	# Pre‑allocate temporary vectors for the adjoint products
	tmp1 = similar(x_last)
	tmp2 = similar(x_last)

	@inbounds for j in ORD:-1:3
		# tmp1 = linear_terms[j]' * x_last
		mul!(tmp1, adjoint(linear_terms[j]), x_last)
		@views eigenvectors[:, j-1, index] .= λ .* eigenvectors[:, j, index] .-
											  eigenvectors[:, j, index-1] .+ tmp1
	end

	# Special handling for the first component when λ ≠ 0
	if !iszero(λ)  # corrected condition
		mul!(tmp1, adjoint(linear_terms[1]), x_last)
		@views eigenvectors[:, 1, index] .= (-1 / λ) .* (tmp1 .+ eigenvectors[:, 1, index-1])
	end
end

"""
	propagate_right_eigenvector_from_first(param, y_first, λ, index)

Fills the complete right eigenvector from its first component.
"""
function propagate_right_eigenvector_from_first(
	param::Parametrisation{ORD, NVAR, T},
	y_first::Vector{T2},
	λ::Number,
	index::Int,
) where {ORD, NVAR, T, T2}
	coeff = param.poly.coefficients
	fom = size(coeff, 1)

	@assert length(y_first) == fom "y_first must have length $fom"

	coeff[:, 1, index] .= y_first
	λ_tmp = one(λ)
	@inbounds for j in 1:(ORD-1)
		λ_tmp *= λ
		coeff[:, j+1, index] .= λ_tmp .* y_first
	end
end

"""
	propagate_right_jordan_vector(model, param, λ, index)

Fills the k‑th right Jordan vector from the (k‑1)‑th vector.
Fixes the recurrence bug: now uses column `j` (not `j+1`) on the right‑hand side.
"""
function propagate_right_jordan_vector(
	model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT},
	param::Parametrisation{ORD, NVAR, T2},
	λ::Number,
	index::Int,
) where {ORD, ORDP1, N_NL, N_EXT, T, T2, MT, NVAR}
	linear_terms = model.linear_terms
	coeff = param.poly.coefficients
	fom = size(coeff, 1)

	# Build P(λ) as before
	tmp_mat = copy(linear_terms[1])
	λ_pow = one(λ)
	for i in 2:ORDP1
		λ_pow *= λ
		tmp_mat .+= λ_pow * linear_terms[i]
	end

	# Right‑hand side: -B_{ORD+1} * Y[ORD]_{k-1}
	rhs = -linear_terms[end] * view(coeff, :, ORD, index-1)
	y_first = tmp_mat \ rhs

	@assert length(y_first) == fom "y_first must have length $fom"

	coeff[:, 1, index] .= y_first

	# Correct recurrence: Y[l+1]_k = λ * Y[l]_k + Y[l]_{k-1}
	@inbounds for j in 1:(ORD-1)
		@views coeff[:, j+1, index] .= λ .* coeff[:, j, index] .+ coeff[:, j, index-1]
	end
end

end