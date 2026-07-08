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
    Y = reshape(R, FOM, 2, size(R, 2))
    # Analytic sesquilinear left blocks φ = [(conj(λ)M + C)ϕ; ϕ] — algebraically
    # equal to -(1/conj(λ))Kᵀϕ via the quadratic eigenrelation, but built from
    # the moderate-norm M, C instead of K (which amplifies eigensolver noise by
    # (ω_max/ω₁)²). Shared with StructureModalDampingEigensolver.
    left = MORFE.Eigenproblems._structural_left_eigenmode_orders(
        solver.eigenvalues, Y, model.linear_terms[3], model.linear_terms[2])
    return solver.eigenvalues, left
end
