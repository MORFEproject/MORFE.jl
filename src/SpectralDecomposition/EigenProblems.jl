"""
    EigenProblems

Structures the handling of the eigenproblem, including solving of the eigenproblem and choosing of the master_modes.

TODO:  detailed explanation
"""
module EigenProblems

using ..FullOrderModel

using LinearAlgebra
using Arpack

export AbstractEigenSolver, DefaultEigenSolver, ArpackEigenSolver, solve, solve_left
export EigenProblem, compute_eigen_problem, get_eigenpairs, select_master_modes_by_hand,
       select_master_modes_by_sorting, select_master_modes_by_sorting

"""
    EigenSolver
"""
abstract type AbstractEigenSolver end

function solve(model::NDOrderModel, solver::AbstractEigenSolver, args...)
    error("solve not implemented for $(typeof(solver))")
end
function solve_left(model::NDOrderModel, solver::AbstractEigenSolver, args...)
    error("solve not implemented for $(typeof(solver))")
end

"""
    struct DefaultEigenSolver <: AbstractEigenSolver end    

Default LinearAlgebra eigen solver. 
"""
struct DefaultEigenSolver <: AbstractEigenSolver end

function solve(model::NDOrderModel, solver::DefaultEigenSolver)
    A, B = linear_first_order_matrices(model)
    eig_result = eigen(A, B)
    return eig_result.values, eig_result.vectors
end

function solve_left(model::NDOrderModel, solver::DefaultEigenSolver)
    A, B = linear_first_order_matrices(model)
    eig_result = eigen(A', B')
    return conj(eig_result.values), eig_result.vectors
end

"""
struct ArpackEigenSolver <: AbstractEigenSolver
    nev::Union{Nothing, UInt64}
end

Use Arpack eigs for sparse matrices. Computes nev eigenpairs.

"""
struct ArpackEigenSolver <: AbstractEigenSolver
    nev::Union{Nothing, UInt64}

    function DefaultEigenSolver()
        new{nothing}
    end
    function DefaultEigenSolver(nev::UInt64)
        new{nev}
    end
end

function solve(model::NDOrderModel, solver::ArpackEigenSolver)
    A, B = linear_first_order_matrices(model)
    if solver.nev === nothing
        eig_result = eigs(A, B, which = :SM)
    else
        eig_result = eigs(A, B, nev = solver.nev, which = :SM)
    end
    return eig_result.values, eig_result.vectors
end

function solve_left(model::NDOrderModel, solver::ArpackEigenSolver)
    A, B = linear_first_order_matrices(model)
    if solver.nev === nothing
        eig_result = eigs(A', B', which = :SM)
    else
        eig_result = eigs(A', B', nev = solver.nev, which = :SM)
    end
    return conj(eig_result.values), eig_result.vectors
end

"""
mutable struct EigenProblem{T}
    model::NDOrderModel
    solver::AbstractEigenSolver
    eigenvalues::Vector{Complex{T}}
    eigenmodes::Matrix{Complex{T}}
    master_modes::Union{Nothing, Vector{UInt64}}
    left_eigenvectors::Union{Nothing, Matrix{Complex{T}}}

Struct to save sorted and matched left and right eigenpairs.
Additionally it defines which modes are master_modes.
"""
mutable struct EigenProblem{T}
    model::NDOrderModel
    solver::AbstractEigenSolver
    eigenvalues::Vector{Complex{T}}
    eigenmodes::Matrix{Complex{T}}
    left_eigenmodes::Union{Nothing, Matrix{Complex{T}}}
    master_modes::Union{Nothing, Vector{Bool}}
    external_modes::Union{Nothing, Vector{Bool}} # TODO
    function EigenProblem(
            model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT},
            solver::AbstractEigenSolver,
            eigenvalues::Vector{C},
            eigenmodes::Matrix{C},
            left_eigenmodes::Matrix{C}) where {
            ORD, ORDP1, N_NL, N_EXT, T, MT <: AbstractMatrix{T}, C}
        #Assert sizes
        FOM = size(model.linear_terms[1], 1) * ORD
        @assert size(eigenvalues, 1)==size(eigenmodes, 2) "Size of eigenvalues and eigenmodes doesnt match!"
        @assert size(eigenmodes)==size(left_eigenmodes) "Size of left and right eigenmodes must be the same!"
        @assert size(eigenmodes, 1)==FOM "Eigenmode has wrong size!"
        new{T}(model, solver, eigenvalues, eigenmodes, nothing, nothing)
    end
end

"""
    compute_eigen_problem(
        model::NDOrderModel;
        solver::AbstractEigenSolver = DefaultEigenSolver(),
        sorter::Function = sort_by_magnitude,
        normalizer::Function = normalize_biorthogonal)

Computes left and right eigenpairs of the problem described in `model` by using the defined `solver`.
Additionally `sorter` sorts the eigenpairs and `normalizer` is used to normalize the eigenmodes.
"""
function compute_eigen_problem(
        model::NDOrderModel;
        solver::AbstractEigenSolver = DefaultEigenSolver(),
        sorter!::Function = sort_by_magnitude!,
        normalizer!::Function = normalize_biorthogonal!)

    #calculate right eigenmodes
    (eigenvalues, eigenmodes) = solve(model, solver)

    #sort eigenpairs
    sorter!(eigenvalues, eigenmodes)

    #calculate left eigenmodes
    (left_eigenvalues, left_eigenmodes) = solve_left(model, solver)
    sort_left_eigenmodes!(eigenvalues, left_eigenvalues, left_eigenmodes)

    # normalize eigenpairs
    normalizer!(model, eigenmodes, left_eigenmodes)

    # Construct EigenProblem
    EigenProblem(model, solver, eigenvalues, eigenmodes, left_eigenmodes)
end

"""
    sort_by_magnitude!(eigenvalues, eigenmodes)

Sorts eigenpairs to resamble the order:
    |λ[1]| ≤ |λ[2]| ≤ ...  
where λ=eigenvalues.
"""
function sort_by_magnitude!(eigenvalues, eigenmodes)
    idx = sortperm(eigenvalues; by = abs)
    eigenvalues .= eigenvalues[idx]
    eigenmodes .= eigenmodes[:, idx]
end

"""
    sort_left_eigenmodes!(eigenvalues, left_eigenvalues, left_eigenmodes)

Sorts the left eigenpairs to match richt eigenpairs, by using the distance function
    dist(a, b) = abs(real(a - b)) + abs(imag(a - b))
"""
function sort_left_eigenmodes!(eigenvalues, left_eigenvalues, left_eigenmodes)
    tol = 1e-8
    n = length(eigenvalues)
    perm = zeros(Int, n)
    used = falses(n)
    dist(a, b) = abs(real(a - b)) + abs(imag(a - b))

    for i in 1:n
        diffs = map(μj -> dist(μj, eigenvalues[i]), left_eigenvalues)

        # ignore already used indices
        for j in 1:n
            if used[j]
                diffs[j] = Inf
            end
        end

        j = argmin(diffs)

        if diffs[j] > tol
            @warn "No good match for eigenvalue $i (distance=$(diffs[j]))"
        end

        perm[i] = j
        used[j] = true
    end
    left_eigenvalues .= left_eigenvalues[perm]
    left_eigenmodes .= left_eigenmodes[perm]
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
        eigenmodes::Matrix{T},
        left_eigenmodes::Matrix{T}) where {T}
    @assert size(eigenmodes)==size(left_eigenmodes) "Size of left and right eigenmodes must be the same!"

    (_, B) = linear_first_order_matrices(model)
    for i in 1:size(eigenmodes, 2)
        tmp = left_eigenmodes[:, i]' * B * eigenmodes[:, i]
        tmp = sqrt(tmp)
        left_eigenmodes[:, i] /= tmp
        eigenmodes[:, i] /= tmp
    end
end

"""
    get_eigenpairs(ep::EigenProblem)

Returns eigenvalues, (right) eigenmodes and left eigenmodes of an EigenProblem.
"""
function get_eigenpairs(ep::EigenProblem)
    return (ep.eigenvalues, ep.eigenmodes, ep.left_eigenmodes)
end

"""
    select_master_modes_by_hand(ep::Eigenproblem, mastermodes::Vector{Bool})

Define master_modes of EigenProblem by passing a vector of booleans.
    master_modes[i] = true  ===> eigen_modes[:,i] is a mastermode
"""
function select_master_modes_by_hand(ep::EigenProblem, mastermodes::Vector{Bool})
    @assert length(mastermodes)==size(ep.eigenmodes, 2) "mastermodes has wrong length!"
    ep.master_modes = mastermodes
end

"""
    select_master_modes_by_sorting(ep::EigenProblem, nev::UInt64)

Defines master_modes as the first nev eigenpairs. Sorting was done in compute_eigen_problem.
"""
function select_master_modes_by_sorting(ep::EigenProblem, nev::UInt64)
    n = size(ep.eigenmodes, 2)
    ep.master_modes = [i <= nev for i in 1:n]
end

"""
    select_master_modes_by_target_frequency(ep::EigenProblem, target_frequencies::Vector, tol)

Defines master_modes by specified target values. 
All eigenvalues that are of distance of one target value smaller than `tol` are marked for mastermodes.
Distance used:
    dist(a, b) = abs(real(a - b)) + abs(imag(a - b)).
"""
function select_master_modes_by_sorting(
        ep::EigenProblem,
        target_frequencies::Vector,
        tol::Float64)
    n = lenght(ep.eigenvalues)
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
            println("No eigenvalue found for target $target_frequency. Closest distance = $(dists[j]) at eigenvalue $j")
        end
    end
    println("Chosen mastermodes:")
    println(master_modes)
    ep.master_modes = master_modes
end

function group_conjugate_pairs()
    #TODO
end

end # module