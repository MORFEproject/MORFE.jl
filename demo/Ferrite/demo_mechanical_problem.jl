"""
Structural mechanical problem (clamped-clamped beam) using Ferrite.jl as FEM backend.

Demonstrates the MORFE.jl pipeline with the RHS-C batched FEM path:
  1. Assemble K, M via Ferrite element loops.
  2. Wrap geometric nonlinearity as FerriteGeometricNonlinearity (FEMMultilinearMap).
  3. Build NDOrderModel and solve cohomological equations.
  4. Realify reduced dynamics.

Material: St. Venant-Kirchhoff (nonlinearity up to cubic order in u).
Geometry: clamped-clamped beam, 1.0 × 0.1 × 0.1, 20×3×3 Hex8 elements.
"""

using MORFE
using Ferrite
using SparseArrays
using LinearAlgebra
using Arpack
using StaticArrays

include(joinpath(@__DIR__, "ferrite_assembly.jl"))

# -----------------------------------------------------------------------
# 1. Mesh and FE setup
# -----------------------------------------------------------------------

L, b, h = 1.0, 0.1, 0.1
grid = generate_grid(Hexahedron, (20, 3, 3),
    Vec((0.0, 0.0, 0.0)), Vec((L, b, h)))

addfaceset!(grid, "Dirichlet", x -> x[1] ≈ 0.0 || x[1] ≈ L)

ip = Lagrange{RefHexahedron, 1}()^3
qr = QuadratureRule{RefHexahedron}(2)   # degree-2 quadrature (2 points per direction)
cv = CellValues(qr, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfaceset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch)
update!(ch, 0.0)

println("Total DOFs : ", ndofs(dh))

# -----------------------------------------------------------------------
# 2. Linear matrices
# -----------------------------------------------------------------------

E  = 160e3
ν  = 0.22
ρ  = 2.32e-3
λ  = (E * ν) / ((1 + ν) * (1 - 2ν))
μ  = E / (2(1 + ν))
α  = 0.5370828278264171 / 100.0      # Rayleigh mass coefficient
β  = 1.0 / (0.5370828278264171 * 100.0)  # Rayleigh stiffness coefficient

K_full = allocate_matrix(dh)
M_full = allocate_matrix(dh)

assemble_KM!(K_full, M_full, dh, cv, λ, μ, ρ)

# Free DOFs = all DOFs minus the prescribed (Dirichlet) ones.
free          = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free        = length(free)

K = complex.(K_full[free, free])
M = complex.(M_full[free, free])
C = α * M + β * K

println("Free DOFs  : ", n_free)

# -----------------------------------------------------------------------
# 3. Nonlinear terms (FEMMultilinearMap — RHS-C batched path)
# -----------------------------------------------------------------------

term_quad  = FerriteGeometricNonlinearity{2}(dh, cv, free_to_local, n_free, λ, μ)
term_cubic = FerriteGeometricNonlinearity{3}(dh, cv, free_to_local, n_free, λ, μ)

# -----------------------------------------------------------------------
# 4. NDOrderModel:  M ü + C u̇ + K u = F(u)
#    linear_terms = (K, C, M) = (B0, B1, B2)
# -----------------------------------------------------------------------

model = NDOrderModel(
    (K, C, M),
    (term_quad, term_cubic),
)

# -----------------------------------------------------------------------
# 5. Eigenproblem
# -----------------------------------------------------------------------

"""
    Mechanical_Problem_Solver <: AbstractEigenSolver

Solves the undamped eigenproblem K ϕ = ω² M ϕ and recovers the damped
eigenvalues λ = ω(-ξ ± i√(1-ξ²)) using Rayleigh damping ξ = ½(α/ω + βω).
"""
mutable struct Mechanical_Problem_Solver <: AbstractEigenSolver
    right_eig_result::Union{Nothing, Matrix}
    eigenvalues::Union{Nothing, Vector}
    nev::Int
    α::Float64
    β::Float64
end

function MORFE.Eigenproblems.solve(model::NDOrderModel, solver::Mechanical_Problem_Solver)
    ω2, ϕ = eigs(model.linear_terms[1], model.linear_terms[3];
                 nev = solver.nev, which = :SM)
    idx = sortperm(real(ω2))[1:solver.nev]
    ω2  = real.(ω2[idx])
    ϕ   = real.(ϕ[:, idx])
    ω   = sqrt.(ω2)
    FOM = size(ϕ, 1)

    λ_all = zeros(ComplexF64, 2 * solver.nev)
    for i in 1:solver.nev
        ξ = 0.5 * (solver.α / ω[i] + solver.β * ω[i])
        λ_all[2i-1] = ω[i] * (-ξ + sqrt(Complex(1.0 - ξ^2)) * im)
        λ_all[2i]   = ω[i] * (-ξ - sqrt(Complex(1.0 - ξ^2)) * im)
    end

    evecs = Matrix{ComplexF64}(undef, 2FOM, 2 * solver.nev)
    for i in 1:solver.nev
        evecs[1:FOM,        2i-1] .= ϕ[:, i]
        evecs[1:FOM,        2i]   .= ϕ[:, i]
        evecs[FOM+1:end,    2i-1] .= λ_all[2i-1] .* ϕ[:, i]
        evecs[FOM+1:end,    2i]   .= λ_all[2i]   .* ϕ[:, i]
    end
    solver.right_eig_result = evecs
    solver.eigenvalues       = λ_all
    return λ_all, evecs
end

function MORFE.Eigenproblems.solve_left(model::NDOrderModel, solver::Mechanical_Problem_Solver)
    @assert solver.right_eig_result !== nothing "Run solve() first"
    R   = solver.right_eig_result
    FOM = size(R, 1) ÷ 2
    L   = similar(R)

    for i in 1:length(solver.eigenvalues) ÷ 2
        L[FOM+1:end, 2i-1] = R[1:FOM, 2i]
        L[FOM+1:end, 2i]   = R[1:FOM, 2i-1]
    end
    for (i, λ) in enumerate(solver.eigenvalues)
        L[1:FOM, i] = -(1 / conj(λ)) * model.linear_terms[1]' * L[FOM+1:end, i]
    end
    return solver.eigenvalues, L
end

eigenproblem = compute_eigenproblem(
    model,
    solver  = Mechanical_Problem_Solver(nothing, nothing, 10, α, β),
    sorter! = (args...) -> nothing,
)
(eigenvalues, Y, X) = get_eigenpairs(eigenproblem)

println("\nFirst eigenvalues:")
for (i, λ) in enumerate(eigenvalues[1:min(6, end)])
    println("  mode $i: λ = $λ")
end

# -----------------------------------------------------------------------
# 6. Master modes and reduced-variable structure
# -----------------------------------------------------------------------

FOM   = n_free
ROM   = 2
N_EXT = 0
NVAR  = ROM + N_EXT

select_master_modes_by_sorting(eigenproblem, ROM)

master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes       = Y[1:FOM,      1:ROM]
left_eigenmodes    = Y[FOM+1:end,  1:ROM]

ORD_model = length(model.linear_terms) - 1   # = 2
master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
for r in 1:ROM
    for k in 1:(ORD_model - 1)
        master_modes_derivatives[:, k, r] .= Y[k*FOM+1:(k+1)*FOM, r]
    end
end

# -----------------------------------------------------------------------
# 7. Multiindex set and resonance set
# -----------------------------------------------------------------------

max_degree = 3
mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)
println("\nMultiindex set: degree ≤ $max_degree in $NVAR variables → $(length(mset)) monomials")

super_eigenvalues  = Vector{ComplexF64}(master_eigenvalues)
target_eigenvalues = Vector{ComplexF64}(master_eigenvalues)

resonance_set = resonance_set_from_complex_normal_form_style(
    ROM, mset, super_eigenvalues, target_eigenvalues, 0.05)

println("\nResonance set:")
for (idx, mi) in enumerate(mset.exponents)
    res_str = join(findall(resonance_set.resonances[:, idx]), ", ")
    isempty(res_str) && (res_str = "none")
    println("  $mi → [$res_str]")
end

# -----------------------------------------------------------------------
# 8. Solve cohomological equations
# -----------------------------------------------------------------------

@time W, R = solve_cohomological_problem(
    model, mset,
    master_eigenvalues,
    master_modes, left_eigenmodes,
    resonance_set;
    master_modes_derivatives,
)

# -----------------------------------------------------------------------
# 9. Realify reduced dynamics
# -----------------------------------------------------------------------

conj_map = zeros(Int, NVAR)
for i in 1:NVAR
    conj_map[i] = isodd(i) ? i + 1 : i - 1
end
Rr = ReducedDynamics(realify(R.poly, conj_map), R.external_system_size)

println("\nReduced dynamics coefficients (real part):")
for m in 1:length(Rr.poly.multiindex_set.exponents)
    mi = Rr.poly.multiindex_set.exponents[m]
    c  = Rr.poly.coefficients[:, m]
    any(abs.(c) .> 1e-12) && println("  $mi : $(real.(c))")
end
