"""
Module `Eigenproblems` — solve and manage the generalised eigenproblem for an `NDOrderModel`.

Starting from an `NDOrderModel`, the companion-form first-order matrices `(A, B)` are
assembled and the generalised eigenproblem `A v = λ B v` is solved.  The results are
stored in an `Eigenproblem` struct together with:

---
# Module contents
### Eigensolver
Eigensolver is a type that implements different methods to solve the eigenproblem. 
Every Eigensolver must have the parent type [`AbstractEigensolver`](@ref) needs to implement to methods: [`solve`](@ref) and [`solve_left`](@ref).
Implemented eigensolvers are:
1. [`DefaultEigensolver`](@ref): Uses the `eigen()` from the package LinearAlgebra.
    Calculates full spectrum. This solver is only recommended for small problems. 
2. [`ArpackEigensolver`](@ref): Uses the `eigs()` from the package Arpack.
    Calculates a specific number of modes of smalles magnitude by default. 
    Works only for sparse matrices
3. MorfeEigensolver: Uses the `eigs()` from the package Arpack with a shift of the spectrum.
4. [`StructureModalDampingEigensolver`](@ref): Uses the `eigs()` from the package Arpack, but operates
    specifically  on the second order system with Raleigh damping.

### EigenProblem
Struct with the attributs:
- `model`: [`NDOrderModel`](@ref)
- `solver`: [`AbstractEigensolver`](@ref)
- `eigenvalues`: Vector containing the eigenvalues
- `eigenvectors`: 3D Array containing the eigenvectors as columns
- `master_modes`: boolean vector of the same length as eigenvectors, showing wich eigenvectors are master_modes
- `left_eigenvectors`: Matrix containing the left eigenvectors as columns

### Compute eigen problem [`compute_eigenproblem`](@ref)
Function that takes a [`NDOrderModel`](@ref) and a [`AbstractEigensolver`](@ref) and calculates the eigenvalues and left and right eigenvectors.
Additionally it takes a function to sort the eigenvectors and a function to normalize.
It generates and returns an EigenProblem and is the main interaction of this module.

### Master mode selection 
The selection of the master modes is done after using `compute_eigenproblem`.
There are the options:
1. [`select_master_modes_by_hand`](@ref)
2. [`select_master_modes_by_sorting`](@ref)
3. [`select_master_modes_by_target_frequency`](@ref)

"""
module Eigenproblems

using ..FullOrderModel
using ..Eigensolvers

using LinearAlgebra
using Arpack

export AbstractEigensolver, DefaultEigensolver, ArpackEigensolver, MorfeEigensolver,
       StructureModalDampingEigensolver
export solve, solve_left, sort_by_magnitude!, normalize_biorthogonal!
export Eigenproblem, compute_eigenproblem, get_eigenpairs, select_master_modes_by_hand,
       select_master_modes_by_sorting, select_master_modes_by_target_frequency

"""
	AbstractEigensolver

Abstract supertype for all eigensolvers accepted by `compute_eigenproblem`.

Concrete subtypes must implement `solve` and `solve_left` for `NDOrderModel`.
Two built-in subtypes are provided: `DefaultEigensolver` and `ArpackEigensolver`.
"""
abstract type AbstractEigensolver end

function solve(model::NDOrderModel, solver::AbstractEigensolver, args...)
    error("solve not implemented for $(typeof(solver))")
end
function solve_left(model::NDOrderModel, solver::AbstractEigensolver, args...)
    error("solve not implemented for $(typeof(solver))")
end

"""
	DefaultEigensolver <: AbstractEigensolver

Dense eigensolver backed by `LinearAlgebra.eigen`. Computes all eigenpairs.
Suitable for small to medium full-order models (dense matrices).
"""
struct DefaultEigensolver <: AbstractEigensolver end

"""
    solve(model::NDOrderModel, solver::DefaultEigensolver)

Solves right eigenproblem using `eigen` from LinearAlgebra. 
Let `A` and `B` be the first order matrices of `model`. Then it returns the eigenpairs
```math
    (A-\\lambda_k B)y_k = 0
```
"""
function solve(model::NDOrderModel, solver::DefaultEigensolver)
    A, B = linear_first_order_matrices(model)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    eig_result = eigen(A, B)

    # Reshape eigenvectors from (ORD*FOM) x (ORD*FOM) to (FOM x ORD x number of eigenvalues)
    num_eigenvals = length(eig_result.values)
    reshaped_eigenvectors = reshape(eig_result.vectors, FOM, ORD, num_eigenvals)
    return eig_result.values, reshaped_eigenvectors
end

"""
    solve_left(model::NDOrderModel, solver::DefaultEigensolver)

Solves left eigenproblem using `eigen` from LinearAlgebra. 
let `A` and `B` be the first order matrices of `model`. Then it returns the eigenpairs
```math
    x_k^H(A-\\lambda_k B) = 0
```
"""
function solve_left(model::NDOrderModel, solver::DefaultEigensolver)
    A, B = linear_first_order_matrices(model)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    eig_result = eigen(A', B')

    # Reshape eigenvectors from (ORD*FOM) x (ORD*FOM) to (FOM x ORD x number of eigenvalues)
    num_eigenvals = length(eig_result.values)
    reshaped_eigenvectors = reshape(eig_result.vectors, FOM, ORD, num_eigenvals)
    return conj(eig_result.values), reshaped_eigenvectors
end

"""
	ArpackEigensolver <: AbstractEigensolver

Sparse eigensolver backed by `Arpack.eigs`. Computes `nev` smallest-magnitude
eigenpairs. If `nev` is not specified, Arpack's default count is used.
"""
mutable struct ArpackEigensolver <: AbstractEigensolver
    nev::Union{Nothing, Int64}
    eigenvalues::Union{Nothing, Vector{ComplexF64}}

    function ArpackEigensolver()
        @warn "Initialized ArpackEigensolver with no nev. Not recommended! Use DefaultEigensolver instead."
        new(nothing)
    end
    function ArpackEigensolver(nev::Int64)
        @assert nev>0 "nev must be greater than zero!"
        new(nev)
    end
end

"""
    solve(model::NDOrderModel, solver::ArpackEigensolver)

Solves right eigenproblem using `eigs` from Arpack. 
Let `A` and `B` be the first order matrices of `model`. Then it returns the eigenpairs
```math
    (A-\\lambda_k B)y_k = 0
```
"""
function solve(model::NDOrderModel, solver::ArpackEigensolver)
    A, B = linear_first_order_matrices(model)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    if solver.nev === nothing
        solver.nev = FOM * ORD
        (values, vectors) = eigs(A, B, which = :SM)
    else
        (values, vectors) = eigs(A, B, nev = solver.nev, which = :SM)
    end

    # Reshape eigenvectors from (ORD*FOM) x (number of eigenvalues) to (FOM x ORD x number of eigenvalues)
    num_eigenvals = length(values)
    reshaped_eigenvectors = reshape(vectors, FOM, ORD, num_eigenvals)
    solver.eigenvalues = values
    return values, reshaped_eigenvectors
end

"""
    solve_left(model::NDOrderModel, solver::ArpackEigensolver)

Solves left eigenproblem using `eigs` from Arpack. 
let `A` and `B` be the first order matrices of `model`. Then it returns the eigenpairs
```math
    x_k^H(A-\\lambda_k B) = 0
```
"""
# function solve_left(model::NDOrderModel, solver::ArpackEigensolver)
#     A, B = linear_first_order_matrices(model)
#     if solver.nev === nothing
#         (values, vectors) = eigs(A', B', which = :SM)
#     else
#         (values, vectors) = eigs(A', B', nev = (solver.nev + 1), which = :SM)
#     end

#     # Reshape eigenvectors from (ORD*FOM) x (number of eigenvalues) to (FOM x ORD x number of eigenvalues)
#     FOM = size(model.linear_terms[1], 1)
#     ORD = length(model.linear_terms) - 1
#     num_eigenvals = length(values)
#     reshaped_eigenvectors = reshape(vectors, FOM, ORD, num_eigenvals)
#     return conj(values), reshaped_eigenvectors
# end
"""
    solve_left(model::NDOrderModel, solver::ArpackEigensolver)

Solves left eigenproblem using `eigs` from Arpack with σ-shift to recover the correct eigenvalues.
"""
function solve_left(model::NDOrderModel, solver::ArpackEigensolver)
    A, B = linear_first_order_matrices(model)
    A_c = complex.(A)
    B_c = complex.(B)
    A_adjoint = A_c'
    B_adjoint = B_c'
    @assert solver.nev == length(solver.eigenvalues)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    left_eigenvectors = Array{ComplexF64}(undef, FOM, ORD, solver.nev)
    eigenvalues = Vector{ComplexF64}(undef, solver.nev)
    for i in 1:(solver.nev)
        values, vectors = eigs(
            A_adjoint, B_adjoint, sigma = conj(solver.eigenvalues[i]), which = :LM, nev = 1, ncv = 30)
        left_eigenvectors[:, :, i] = reshape(vectors[:, 1], FOM, ORD)
        eigenvalues[i] = conj(values[1])
    end
    return eigenvalues, left_eigenvectors
end

"""
struct MorfeEigensolver <: AbstractEigensolver
	nev::Union{Nothing, Int64}
end

Solves shifted eigenproblem using Arpack eigs for sparse matrices. 
shift = nothing -> uses standard eigs
Computes nev eigenpairs.
"""
mutable struct MorfeEigensolver <: AbstractEigensolver
    nev::Union{Nothing, Int64}
    shift::Union{Nothing, ComplexF64}
    eigenvalues::Union{Nothing, Vector{ComplexF64}}

    function MorfeEigensolver()
        new(nothing, nothing)
    end
    function MorfeEigensolver(nev::Int64, shift::ComplexF64)
        @assert nev>0 "nev must be greater than zero!"
        new(nev, shift)
    end
end

"""
    solve(model::NDOrderModel, solver::MorfeEigensolver)

Solves right eigenproblem using `generalised_eigenpairs` from `MORFE.Eigensolvers`. 
Let `A` and `B` be the first order matrices of `model`. Then it returns the eigenpairs
```math
    (A-\\lambda_k B)y_k = 0
```
"""
function solve(model::NDOrderModel, solver::MorfeEigensolver)
    A, B = linear_first_order_matrices(model)
    if solver.nev === nothing
        solver.nev = size(A, 1)
    end
    eig_result = generalised_eigenpairs(A, B; solver.nev, shift = solver.shift)

    # Reshape eigenvectors from (ORD*FOM) x (number of eigenvalues) to (FOM x ORD x number of eigenvalues)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    num_eigenvals = length(eig_result.values)
    reshaped_eigenvectors = reshape(eig_result.vectors, FOM, ORD, num_eigenvals)
    solver.eigenvalues = eig_result.values
    return eig_result.values, reshaped_eigenvectors
end

"""
    solve_left(model::NDOrderModel, solver::MorfeEigensolver)

Solves left eigenproblem using `eigen` from LinearAlgebra. 
let `A` and `B` be the first order matrices of `model`. Then it returns the eigenpairs
```math
    x_k^H(A-\\lambda_k B) = 0
```
"""
# function solve_left(model::NDOrderModel, solver::MorfeEigensolver)
#     A, B = linear_first_order_matrices(model)
#     if solver.nev === nothing
#         solver.nev = size(A, 1)
#     end

#     eig_result = generalised_eigenpairs(A', B'; nev = solver.nev + 1, shift = solver.shift)

#     # Reshape eigenvectors from (ORD*FOM) x (number of eigenvalues) to (FOM x ORD x number of eigenvalues)
#     FOM = size(model.linear_terms[1], 1)
#     ORD = length(model.linear_terms) - 1
#     num_eigenvals = length(eig_result.values)
#     reshaped_eigenvectors = reshape(eig_result.vectors, FOM, ORD, num_eigenvals)
#     return conj(eig_result.values), reshaped_eigenvectors
# end
function solve_left(model::NDOrderModel, solver::MorfeEigensolver)
    A, B = linear_first_order_matrices(model)
    A_c = complex.(A)
    B_c = complex.(B)
    A_adjoint = A_c'
    B_adjoint = B_c'
    @assert solver.nev == length(solver.eigenvalues)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    left_eigenvectors = Array{ComplexF64}(undef, FOM, ORD, solver.nev)
    eigenvalues = Vector{ComplexF64}(undef, solver.nev)
    for i in 1:(solver.nev)
        eig_results = generalised_eigenpairs(
            A', B'; nev = 1, ncv = 30, shift = conj(solver.eigenvalues[i]))
        left_eigenvectors[:, :, i] = reshape(eig_results.vectors[:, 1], FOM, ORD)
        eigenvalues[i] = conj(eig_results.values[1])
    end
    return eigenvalues, left_eigenvectors
end

"""
	StructureModalDampingEigensolver <: AbstractEigensolver

solves eigenproblem of the mechanical 2nd order problem:
```math
M \\ddot{U} + C \\dot{U} + K U + n(U) = 0
```
with ``C = \\alpha*M + \\beta*K`` under the assumption that M and K are spd matrices.

The steps are:
1) calculate eigenpair \$\\omega_k, \\varphi_k\$ of: \$(K-\\omega_k^2*M)\\varphi_k=0\$
2) calculate: \$\\xi_k=0.5(\\frac{\\alpha}{\\omega_k} + \\beta * \\omega_k)\$
3) calculate eigenvalues: \$\\lambda_k = -\\xi_k*\\omega_k \\sqrt{1-\\xi_k^2}\$

Calculates only the first `nev` eigenvectors.
"""
mutable struct StructureModalDampingEigensolver <: AbstractEigensolver
    right_eig_result::Union{Nothing, Array}
    eigenvalues::Union{Nothing, Vector}
    nev::Int64
    α::Float64
    β::Float64

    function StructureModalDampingEigensolver(nev::Int64, α::Float64, β::Float64)
        @assert nev>0 "nev must be greater than zero!"
        new(nothing, nothing, nev, α, β)
    end
end

"""
    solve(model::NDOrderModel, solver::StructureModalDampingEigensolver)

Calculates right eigenvectors in the secod order form.
Uses the relations:
```math
    (K-\\omega_k^2*M)\\y_{k,U}=0 
```
```math
    y_{k,V}=\\lambda_k y_{k,U} 
```
"""
function solve(model::NDOrderModel, solver::StructureModalDampingEigensolver)
    function mass_normalization!(ϕ, M, neig)
        #
        for i in 1:neig
            c = transpose(ϕ[:, i]) * M * ϕ[:, i]
            for j in 1:(M.m)
                ϕ[j, i] /= sqrt(c)
            end
        end
        #
        return nothing
    end

    @time ω2,
    ϕ = eigs(
        model.linear_terms[1], model.linear_terms[3], nev = solver.nev,
        which = :LM, sigma = 1e-6, maxiter = 500, ncv = 20)
    mass_normalization!(ϕ, model.linear_terms[3], solver.nev)
    FOM = length(ϕ[:, 1])
    ω = sqrt.(real(ω2))
    ϕ = real(ϕ)
    λ = zeros(ComplexF64, solver.nev * 2)
    for i in 1:(solver.nev)
        ξ = 0.5 * (solver.α / ω[i] + solver.β * ω[i])
        λ[2 * i - 1] = ω[i] * (-ξ + sqrt(Complex(1.0 - ξ^2)) * im)
        λ[2 * i] = ω[i] * (-ξ - sqrt(Complex(1.0 - ξ^2)) * im)
    end
    eigenvectors = Array{ComplexF64}(undef, FOM, 2, solver.nev * 2)
    for i in 1:(solver.nev)
        eigenvectors[1:FOM, 1, (i * 2 - 1)] .= ϕ[:, i]
        eigenvectors[1:FOM, 1, (i * 2)] .= ϕ[:, i]
        #velocity part
        eigenvectors[1:FOM, 2, (i * 2 - 1)] .= λ[(i * 2 - 1)] * ϕ[:, i]
        eigenvectors[1:FOM, 2, (i * 2)] .= λ[(i * 2)] * ϕ[:, i]
    end
    # store results in solver for left eigen_vectors
    solver.right_eig_result = eigenvectors
    solver.eigenvalues = λ
    return λ, eigenvectors
end

"""
    solve_left(model::NDOrderModel, solver::StructureModalDampingEigensolver)

"""
function solve_left(model::NDOrderModel, solver::StructureModalDampingEigensolver)
    @assert solver.right_eig_result!==nothing "First run solve()"
    left_eigenvectors = similar(solver.right_eig_result)
    FOM = size(left_eigenvectors, 1)
    for i in 1:Int(0.5 * size(left_eigenvectors, 3))
        left_eigenvectors[1:FOM, 2, (i * 2 - 1)] = solver.right_eig_result[
            1:FOM, 1, (i * 2)]
        left_eigenvectors[1:FOM, 2, (i * 2)] = solver.right_eig_result[
            1:FOM, 1, (i * 2 - 1)]
    end
    for (i, λ) in enumerate(solver.eigenvalues)
        # left_eigenvectors[(FOM + 1):end, i] = solver.right_eig_result[1:FOM, i]
        left_eigenvectors[1:FOM, 1, i] = -(1 / conj(λ)) * model.linear_terms[1]' *
                                         left_eigenvectors[1:FOM, 2, i]
    end
    return solver.eigenvalues, left_eigenvectors
end

"""
mutable struct Eigenproblem{T}
	model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT}
	solver::AbstractEigensolver
	eigenvalues::Vector{Complex{T}}
	eigenmodes::Array{Complex{T}}
	master_modes::Union{Nothing, Vector{Bool}}
	left_eigenvectors::Union{Nothing, Array{Complex{T}}}

Stores sorted and biorthogonally normalised left/right eigenpairs of the
generalised eigenproblem `A x = λ B x`, together with a master-mode selector.
Eigenmodes are stored in a 3D array of the size (FOM x ORD x Number_of_eigenvalues).
Additionally it defines which modes are master_modes.

# Fields
- `model`: the `NDOrderModel` from which the eigenproblem was derived
- `solver`: the `AbstractEigensolver` used to compute eigenpairs
- `eigenvalues`: sorted eigenvalues `λ`
- `eigenmodes`: right eigenvectors (columns), sorted to match `eigenvalues`
- `left_eigenmodes`: left eigenvectors matched to right eigenvectors
- `master_modes`: `Vector{Bool}` flagging master modes; `nothing` until set by a `select_master_modes_*` call
"""
mutable struct Eigenproblem{T}
    model::NDOrderModel
    solver::AbstractEigensolver
    eigenvalues::Array{Complex{T}}
    eigenmodes::Array{Complex{T}}
    left_eigenmodes::Union{Nothing, Array{Complex{T}}}
    master_modes::Union{Nothing, Vector{Bool}}
    external_modes::Union{Nothing, Vector{Bool}} # TODO
    function Eigenproblem(
            model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT},
            solver::AbstractEigensolver,
            eigenvalues::Array{Complex{T}},
            eigenmodes::Array{Complex{T}},
            left_eigenmodes::Array{Complex{T}}) where {
            ORD, ORDP1, N_NL, N_EXT, T, MT <: AbstractMatrix{T}}
        #Assert sizes
        FOM = size(model.linear_terms[1], 1)
        @assert ndims(eigenmodes)==3 "eigenmodes need to be an array of dimension 3!"
        @assert size(eigenvalues, 1)==size(eigenmodes, 3) "Size of eigenvalues and eigenmodes doesnt match!"
        @assert size(eigenmodes)==size(left_eigenmodes) "Size of left and right eigenmodes must be the same! $(size(eigenmodes)), $(size(left_eigenmodes)) "
        @assert size(eigenmodes, 1)==FOM "Eigenmode has wrong size in dimension 1!"
        @assert size(eigenmodes, 2)==ORD "Eigenmode has wrong size in dimension 2!"
        new{T}(model, solver, eigenvalues, eigenmodes, left_eigenmodes, nothing, nothing)
    end
end

"""
	compute_eigenproblem(
		model::NDOrderModel;
		solver::AbstractEigensolver = DefaultEigensolver(),
		sorter::Function = sort_by_magnitude,
		normalizer::Function = normalize_biorthogonal)

Computes left and right eigenpairs of the problem described in `model` by using the defined `solver`.
Additionally `sorter` sorts the eigenpairs and `normalizer` is used to normalize the eigenmodes.
"""
function compute_eigenproblem(
        model::NDOrderModel;
        solver::AbstractEigensolver = DefaultEigensolver(),
        sorter!::Function = sort_by_magnitude!,
        normalizer!::Function = normalize_biorthogonal!)

    #calculate right eigenmodes
    (eigenvalues, eigenmodes) = solve(model, solver)

    #sort eigenpairs
    sorter!(eigenvalues, eigenmodes)

    #calculate left eigenmodes
    (left_eigenvalues, left_eigenmodes) = solve_left(model, solver)
    left_eigenvalues, left_eigenmodes = sort_left_eigenmodes(
        eigenvalues, left_eigenvalues, left_eigenmodes)

    # normalize eigenpairs
    normalizer!(model, eigenmodes, left_eigenmodes)

    # Construct Eigenproblem
    return Eigenproblem(model, solver, eigenvalues, eigenmodes, left_eigenmodes)
end

"""
	sort_by_magnitude!(eigenvalues, eigenmodes)

Sorts eigenpairs to resemble the order:
	|λ[1]| ≤ |λ[2]| ≤ ...  
where λ=eigenvalues.
"""
function sort_by_magnitude!(eigenvalues, eigenmodes)
    idx = sortperm(eigenvalues; by = abs)
    eigenvalues .= eigenvalues[idx]
    eigenmodes .= eigenmodes[:, :, idx]
end

"""
	sort_left_eigenmodes(eigenvalues, left_eigenvalues, left_eigenmodes)

Sorts the left eigenpairs to match right eigenpairs, by using the distance function
	dist(a, b) = abs(real(a - b)) + abs(imag(a - b))
"""
function sort_left_eigenmodes(eigenvalues, left_eigenvalues, left_eigenmodes)
    tol = 1e-8
    n = length(eigenvalues)
    n_left = length(left_eigenvalues)
    @assert abs(n - n_left)<=1 "Number of left and right eigenvalues doesnt match (+-1)!"
    perm = zeros(Int, n)
    used = falses(n_left)
    dist(a, b) = abs(real(a - b)) + abs(imag(a - b))

    for i in 1:n
        diffs = map(μj -> dist(μj, eigenvalues[i]), left_eigenvalues)

        # ignore already used indices
        for j in 1:n_left
            if used[j]
                diffs[j] = Inf
            end
        end

        j = argmin(diffs)

        if diffs[j] > tol
            @warn "No good match for eigenvalue $i (distance=$(diffs[j]))"
            println("Eigenvalues", eigenvalues)
            println("Left eigenvalues ", left_eigenvalues)
        end

        perm[i] = j
        used[j] = true
    end
    return left_eigenvalues[perm], left_eigenmodes[:, :, perm]
end

"""
	normalize_biorthogonal!(
		model::NDOrderModel,
		eigenmodes::Matrix{T},
		left_eigenmodes::Matrix{T})

Normalize eigenmodes to fulfill the equation
	x_i^H * B * y_j = δ_{ij}
"""
function normalize_biorthogonal!(
        model::NDOrderModel,
        eigenmodes::Array{T},
        left_eigenmodes::Array{T}) where {T}
    @assert size(eigenmodes)==size(left_eigenmodes) "Size of left and right eigenmodes must be the same!"

    (_, B) = linear_first_order_matrices(model)
    for i in 1:size(eigenmodes, 3)
        tmp = vec(left_eigenmodes[:, :, i])' * B * vec(eigenmodes[:, :, i])
        @views eigenmodes[:, :, i] ./= tmp
    end
end

"""
	get_eigenpairs(ep::Eigenproblem)

Returns eigenvalues, (right) eigenmodes and left eigenmodes of an Eigenproblem.
"""
function get_eigenpairs(ep::Eigenproblem)
    return (ep.eigenvalues, ep.eigenmodes, ep.left_eigenmodes)
end

"""
	select_master_modes_by_hand(ep::Eigenproblem, mastermodes::Vector{Bool})

Define master_modes of Eigenproblem by passing a vector of booleans.
	master_modes[i] = true  ===> eigen_modes[:,i] is a mastermode
"""
function select_master_modes_by_hand(ep::Eigenproblem, mastermodes::Vector{Bool})
    @assert length(mastermodes)==size(ep.eigenmodes, 3) "mastermodes has wrong length!"
    ep.master_modes = mastermodes
end

"""
	select_master_modes_by_sorting(ep::Eigenproblem, nev::Int64)

Defines master_modes as the first nev eigenpairs. Sorting was done in compute_eigenproblem.
"""
function select_master_modes_by_sorting(ep::Eigenproblem, nev::Int64)
    @assert nev>0 "nev must be bigger then zero"
    n = size(ep.eigenmodes, 3)
    ep.master_modes = [i <= nev for i in 1:n]
end

"""
	select_master_modes_by_target_frequency(ep::Eigenproblem, target_frequencies::Vector, tol)

Defines master_modes by specified target values. 
All eigenvalues that are of distance of one target value smaller than `tol` are marked for mastermodes.
Distance used:
	dist(a, b) = abs(real(a - b)) + abs(imag(a - b)).
"""
function select_master_modes_by_target_frequency(
        ep::Eigenproblem,
        target_frequencies::Vector,
        tol::Float64)
    n = length(ep.eigenvalues)
    if length(target_frequencies) > n
        @warn "target_frequencies has more entries than calculated eigenvalues. Everything after index $n is neglected!"
    end

    dist(a, b) = abs(real(a - b)) + abs(imag(a - b))
    master_modes = falses(n)

    for target_frequency in target_frequencies
        dists = [dist(ep.eigenvalues[j], target_frequency) for j in 1:n]
        tmp = 0
        for j in 1:n
            if dists[j] < tol
                master_modes[j] = true
                tmp += 1
            end
        end
        if tmp == 0
            j = argmin(dists)
            @warn("No eigenvalue found for target $target_frequency. Closest distance = $(dists[j]) at eigenvalue $j")
        end
    end
    println("Chosen mastermodes: ", master_modes)
    ep.master_modes = master_modes
end

function group_conjugate_pairs()
    #TODO
end

end # module
