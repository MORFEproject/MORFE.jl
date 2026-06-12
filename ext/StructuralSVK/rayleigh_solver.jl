"""
    RayleighEigenSolver <: AbstractEigensolver

Solves the undamped eigenproblem K ϕ = ω² M ϕ and recovers the damped
eigenvalues λ = ω(-ξ ± i√(1-ξ²)) using Rayleigh damping ξ = ½(α/ω + βω).

Promoted verbatim from `examples/01_clamped_beam_ferrite/low_level.jl`
(formerly `Mechanical_Problem_Solver`); behavioural equivalence is enforced
by the `structural_svk` test group.
"""
mutable struct RayleighEigenSolver <: AbstractEigensolver
    right_eig_result::Union{Nothing, Matrix}
    eigenvalues::Union{Nothing, Vector}
    nev::Int
    α::Float64
    β::Float64
end

function MORFE.Eigenproblems.solve(model::NDOrderModel, solver::RayleighEigenSolver)
    ω2,
    ϕ = eigs(model.linear_terms[1], model.linear_terms[3];
        nev = solver.nev, which = :SM)
    idx = sortperm(real(ω2))[1:solver.nev]
    ω2 = real.(ω2[idx])
    ϕ = real.(ϕ[:, idx])
    ω = sqrt.(ω2)
    FOM = size(ϕ, 1)

    λ_all = zeros(ComplexF64, 2 * solver.nev)
    for i in 1:solver.nev
        ξ = 0.5 * (solver.α / ω[i] + solver.β * ω[i])
        λ_all[2i - 1] = ω[i] * (-ξ + sqrt(Complex(1.0 - ξ^2)) * im)
        λ_all[2i] = ω[i] * (-ξ - sqrt(Complex(1.0 - ξ^2)) * im)
    end

    evecs = Matrix{ComplexF64}(undef, 2FOM, 2 * solver.nev)
    for i in 1:solver.nev
        evecs[1:FOM, 2i - 1] .= ϕ[:, i]
        evecs[1:FOM, 2i] .= ϕ[:, i]
        evecs[(FOM + 1):end, 2i - 1] .= λ_all[2i - 1] .* ϕ[:, i]
        evecs[(FOM + 1):end, 2i] .= λ_all[2i] .* ϕ[:, i]
    end
    solver.right_eig_result = evecs
    solver.eigenvalues = λ_all
    return λ_all, reshape(evecs, FOM, 2, 2 * solver.nev)
end

function MORFE.Eigenproblems.solve_left(model::NDOrderModel, solver::RayleighEigenSolver)
    @assert solver.right_eig_result !== nothing "Run solve() first"
    R = solver.right_eig_result
    FOM = size(R, 1) ÷ 2
    L = similar(R)

    for i in 1:(length(solver.eigenvalues) ÷ 2)
        L[(FOM + 1):end, 2i - 1] = R[1:FOM, 2i]
        L[(FOM + 1):end, 2i] = R[1:FOM, 2i - 1]
    end
    for (i, λ) in enumerate(solver.eigenvalues)
        L[1:FOM, i] = -(1 / conj(λ)) * model.linear_terms[1]' * L[(FOM + 1):end, i]
    end
    return solver.eigenvalues, reshape(L, FOM, 2, size(L, 2))
end
