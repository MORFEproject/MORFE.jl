# JordanChain.jl
"""
    JordanChain

Module for computing Jordan chains of a generalized eigenvalue problem `A v = λ B v`.
"""
module JordanChain

using LinearAlgebra
using SparseArrays
using Printf

export compute_jordan_chain

"""
    compute_jordan_chain(A, B, λ, v, L; tol=1e-12)

Compute a Jordan chain for the generalized eigenvalue problem `A v = λ B v`.

A Jordan chain for a given eigenvalue λ and eigenvector v is a sequence of vectors
`v_1 = v, v_2, …, v_ℓ` such that:
- `(λB - A) v_1 = 0`
- `(λB - A) v_j = B v_{j-1}` for j = 2, …, ℓ

The chain length ℓ is the largest integer for which the equations are consistent.

# Arguments
- `A`, `B`: matrices (sparse or dense) of size n × n. They represent the problem `A v = λ B v`.
- `λ`: eigenvalue (complex or real).
- `v`: eigenvector corresponding to λ, provided as a vector of length n.
- `L`: desired length of the chain (≥ 1). This includes the initial eigenvector.
- `tol`: tolerance for checking consistency of the linear systems. A new vector is
         accepted if `‖(λB - A) v_{j+1} - B v_j‖ ≤ tol`. The same tolerance is used
         for the relative tolerance of `lsqminnorm` when solving the singular system.

# Returns
- `chain`: a matrix of size n × ℓ, where ℓ is the actual length of the computed chain.
           Each column is a chain vector, with `chain[:,1] = v`.
- `actual_length`: the length ℓ of the computed chain (≤ L).

"""
function compute_jordan_chain(A, B, λ, v, L; tol=1e-12)
    n = size(A, 1)
    v = reshape(v, n, 1)

    M = λ * B - A
    residual0 = norm(M * v)
    ref_norm = norm(B * v)
    if residual0 > tol * ref_norm
        @warn "Starting vector may not be an eigenvector: " *
              "‖(λB - A)v‖ / ‖Bv‖ = $(residual0/ref_norm)"
    end

    if L == 1
        return v, 1
    end

    # Precompute a factorization of M
    if issparse(M)
        F = qr(M)               # sparse QR
    else
        F = qr(M, Val(true))    # dense QR with column pivoting
    end

    chain = zeros(eltype(v), n, L)
    chain[:, 1] = vec(v)
    actual_length = 1

    for j in 2:L
        rhs = B * chain[:, j-1]
        x = F \ rhs

        resid = norm(M * x - rhs)
        if resid > tol
            @warn "Chain ends at length $(actual_length): cannot extend further (residual = $resid)"
            break
        end

        chain[:, j] = vec(x)
        actual_length += 1
    end

    return chain[:, 1:actual_length], actual_length
end

function solve_forced_frequency(A, B, λ, v, L; tol=1e-12)
    n = size(A, 1)
    v = reshape(v, n, 1)

    M = λ * B - A
    residual0 = norm(M * v)
    ref_norm = norm(B * v)
    if residual0 > tol * ref_norm
        @warn "Starting vector may not be an eigenvector: " *
              "‖(λB - A)v‖ / ‖Bv‖ = $(residual0/ref_norm)"
    end

    if L == 1
        return v, 1
    end

    # Precompute a factorization of M
    if issparse(M)
        F = qr(M)               # sparse QR
    else
        F = qr(M, Val(true))    # dense QR with column pivoting
    end

    chain = zeros(eltype(v), n, L)
    chain[:, 1] = vec(v)
    actual_length = 1

    for j in 2:L
        rhs = B * chain[:, j-1]
        x = F \ rhs

        resid = norm(M * x - rhs)
        if resid > tol
            @warn "Chain ends at length $(actual_length): cannot extend further (residual = $resid)"
            break
        end

        chain[:, j] = vec(x)
        actual_length += 1
    end

    return chain[:, 1:actual_length], actual_length
end

end # module