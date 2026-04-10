module EigenModesPropagation

using ..FullOrderModel
using ..ParametrisationMethod: Parametrisation

"""
    propagate_left_eigenvector_from_last(model, eigenvectors, x_last, λ, index)

Fills complete left eigenvector from the ORD th component (X[ORD])
resulting from turning the N-th order system in to a first order system.
It uses the relations:

    -B[1]*X[ORD] = \\overline{λ} X[1] 
    X[i-1]^H = λ*X[i]^H + X[ORD]^H B[j]

where B[i] are the linear matrices from the NDOrderModel.

# Fields
- `model`: NDOrderModel 
- `eigenvectors`: Stores the result / the eigenvector 
- `x_last`: ORD-Component of left eigenvector  
- `λ`: Eigenvalue of the corresponding eigenvector
- `index`: Describes position of the current eigenvector in eigenvectors
"""
function propagate_left_eigenvector_from_last(
        model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT},
        eigenvectors::Array{FOM, ORD, M},
        x_last::Vector{T},
        λ::Number,
        index::Int) where {T}
    linear_terms = model.linear_terms

    fom = size(param_coeff, 1)
    @assert length(x_last)==fom "Vector x_last needs to have the same size as the full order model: $fom"
    #@assert index?
    eigenvectors[:, ORD, index] .= x_last
    tmp_conj = conj(x_last)
    x_last_conj = conj(x_last)
    for j in ORD:-1:3
        tmp_conj .= conj(eigenvectors[:, j, index])
        eigenvectors[:, j - 1, index] .= conj(λ .* tmp_conj .+
                                              x_last_conj * linear_terms[j])
    end
    # X[1]
    if iszero(λ) != false
        eigenvectors[:, 1, index] .= (-1) / λ * x_last_conj * linear_terms[1]
    else
        eigenvectors[:, 1, index] .= x_last_conj * linear_terms[2]
    end
end

"""
    propagate_left_jordan_vector(model, param, λ, index)

Fills complete left jordan vector from previous jordan vector.
It uses the relations:

    X[ORD]_k^H * (λ^ORD B[ORDP1] + ... + λ^0B[1]) = X[1]_(k-1)^H + ... + X[ORD-1]_(k-1)^H
    X[i-1]_k^H = λ*X[i]_k^H + X[ORD]_k^H B[j] - X[j]_(k-1)^H

where B[i] are the linear matrices from the NDOrderModel and the k index describes the index in the jordan chain.
It is assumed that the index of the previous Jordan vector is `index-1`.

# Fields
- `model`: NDOrderModel 
- `eigenvectors`: Stores the result / the eigenvector 
- `λ`: Eigenvalue of the corresponding eigenvector
- `index`: Describes position of the current jordan vector in eigenvectors
"""
function propagate_left_jordan_vector(
        model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT},
        eigenvectors::Array{FOM, ORD, M},
        λ::Number,
        index::Int) where {T}
    linear_terms = model.linear_terms

    fom = size(param_coeff, 1)
    tmp_mat = linear_terms[1]
    for i in 2:ORDP1
        tmp_mat .+= λ^(i - 1) * linear_terms[i]
    end
    tmp_vec = eigenvectors[:, 1, index - 1]
    for j in 2:ORD
        tmp_vec += eigenvectors[:, j, index - 1]
    end
    x_last = tmp_mat \ tmp_vec
    @assert length(x_last)==fom "Vector x_last needs to have the same size as the full order model: $fom"

    eigenvectors[:, ORD, index] .= x_last
    tmp_conj = conj(x_last)
    tmp_conj2 = conj(x_last)
    x_last_conj = conj(x_last)
    for j in ORD:-1:3
        tmp_conj .= conj(eigenvectors[:, j, index])
        tmp_conj2 .= conj(eigenvectors[:, j, index - 1])
        eigenvectors[:, j - 1, index] .= conj(λ .* tmp_conj .- tmp_conj2 .+
                                              x_last_conj * linear_terms[j])
    end
    # X[1]
    if iszero(λ) != false
        tmp_conj2 .= conj(eigenvectors[:, 1, index - 1])
        eigenvectors[:, 1, index] .= (-1) / λ * (x_last_conj * linear_terms[1] .+ tmp_conj2)
    end
end

"""
    propagate_right_eigenvector_form_first(param, y_first, λ, index)

Fills complete right eigenvector from the ORD th component (X[1])
resulting from turning the N-th order system in to a first order system.
It uses the relations:

    Y[l+1] = λ * Y[l]

where B[i] are the linear matrices from the NDOrderModel.

# Fields
- `param`: Parametrisation of DPIM, stores the result / the eigenvector 
- `y_first`: First-Component of left eigenvector  
- `λ`: Eigenvalue of the corresponding eigenvector
- `index`: Describes position of the current eigenvector in param
"""
function propagate_right_eigenvector_form_first(
        param::Parametrisation{ORD, NVAR, T},
        y_first::Vector{T},
        λ::Number,
        index::Int) where {T}
    param_coeff = param.poly.coefficients

    fom = size(param_coeff, 1)
    @assert length(y_first)==fom "Vector y_first needs to have the same size as the full order model: $fom"

    #@assert index?
    param_coeff[:, 1, index] .= y_first
    λ_tmp = 1.0
    for j in 1:(ORD - 1)
        λ_tmp *= λ
        param_coeff[:, j + 1, index] .= y_first
    end
end

"""
    propagate_right_jordan_vector(model, param, λ, index)

Fills complete right eigenvector from the eigenvector before in the jordan chain.
It uses the relations:

    (λ^ORD B[ORDP1] + ... + λ^0B[1])*Y[0]_k = -B[ORDP1]*Y[ORD]_(k-1)
    Y[l+1]_k = λ * Y[l]_k + Y[l]_(k-1) 

where B[i] are the linear matrices from the NDOrderModel and the k index describes the index in the jordan chain.
It is assumed that the index of the previous Jordan vector is `index-1`.

# Fields
- `model`: NDOrderModel 
- `param`: Parametrisation of DPIM, stores the result / the eigenvector
- `λ`: Eigenvalue of the corresponding eigenvector
- `index`: Describes position of the current jordan vector in param
"""
function propagate_right_jordan_vector(
        model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT},
        param::Parametrisation{ORD, NVAR, T},
        λ::Number,
        index::Int) where {T}
    linear_terms = model.linear_terms
    param_coeff = param.poly.coefficients

    fom = size(param_coeff, 1)
    tmp_mat = linear_terms[1]
    for i in 2:ORDP1
        tmp_mat .+= λ^(i - 1) * linear_terms[i]
    end
    y_first = tmp_mat \ (-linear_terms[end]param_coeff[:, ORD, index - 1])

    @assert length(y_first)==fom "Vector y_first needs to have the same size as the full order model: $fom"

    param_coeff[:, 1, index] .= y_first
    for j in 1:(ORD - 1)
        param_coeff[:, j + 1, index] .= λ * param_coeff[:, j + 1, index] .+
                                        param_coeff[:, j + 1, index - 1]
    end
end

end