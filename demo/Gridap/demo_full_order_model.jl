"""
Structural mechanical problem without forcing.

This script demonstrates the formulation and discretization of a structural mechanics 
problem using finite element methods, followed by conversion to a first-order system.
"""

using Gridap
using GridapGmsh
using WriteVTK
using Gmsh
using SparseArrays

include(joinpath(@__DIR__, "../../src/FullOrderModel.jl"))

"""
Structural mechanical problem without forcing.
First build second order ODE:
Md_t² U + Dd_tU + KU + n(U) = 0
"""

# Load GMSH-Mesh
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
model = GmshDiscreteModel("demo/Gridap/clamped_clamped_beam.msh")

# Define FEM Space
order = 2
degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
Γ = BoundaryTriangulation(model, tags = "Neumann")
Γ_D = BoundaryTriangulation(model, tags = "Dirichlet")
dΓ = Measure(Γ, degree)
reffe = ReferenceFE(lagrangian, VectorValue{3, Float64}, order)
V = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = ["Dirichlet"],
    dirichlet_masks = [(true, true, true)])
g(x) = VectorValue(0.0, 0.0, 0.0)
U = TrialFESpace(V, g)

# Material properties and constitutive relations
sym(u) = 1 / 2 * (u + u')
E = 160e3
ν = 0.22
ρ = 2.32e-3
λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
μ = E / (2 * (1 + ν))
σ(ε) = λ * tr(ε) * one(ε) + 2 * μ * ε

#Linear forms
a(u, v) = ∫(ε(v) ⊙ (σ ∘ ε(u)))dΩ
m(dt2u, v) = ∫(ρ * dt2u ⋅ v)dΩ
E_nl(u1, u2) = 0.25 * ((∇(u1)') ⋅ ∇(u2) + (∇(u2)') ⋅ ∇(u1))
E_nl_grad(∇u1, ∇u2) = 0.25 * ((∇u1') ⋅ ∇u2 + (∇u2') ⋅ ∇u1)
σ_nln(ε) = λ * tr(ε) * one(TensorValue{3, 3, Float64}) + 2 * μ * ε

#Quadratic nonlinear terms
function g_quad(u1, u2, v)
    ∫(ε(v) ⊙ (σ_nln(E_nl(u1, u2))) +
      0.5 * (sym(∇(u1)' ⋅ ∇(v)) ⊙ σ_nln(ε(u2))
             +
             sym(∇(u2)' ⋅ ∇(v)) ⊙ σ_nln(ε(u1))))dΩ
end
#Cubic nonlinear terms
function h_cube(u1, u2, u3, v)
    1 / 3 *
    ∫(sym(∇(u1)' ⋅ ∇(v)) ⊙ (σ_nln(E_nl(u2, u3))) +
      sym(∇(u2)' ⋅ ∇(v)) ⊙ (σ_nln(E_nl(u1, u3))) +
      sym(∇(u3)' ⋅ ∇(v)) ⊙ (σ_nln(E_nl(u1, u2))))dΩ
end

# Assemble matrices of second order system
stiffness_matrix = assemble_matrix((u, v) -> a(u, v), U, V)
mass_matrix = assemble_matrix((u, v) -> m(u, v), U, V)
α = 0.5370828278264171 / (100.0)
β = 1.0 / (0.5370828278264171 * 100.0)
damping_matrix = α * mass_matrix + β * stiffness_matrix

"""
Second order system to first order system conversion

M dt_V + CV + KU + n(U) = 0
M dt_U = MV

    |
    V

B d_t X = AX + N(X)
with:
X = [V     ; U]
B = [M, O  ; 0, M]
A = [-C,-K ; M,0]
N = [n(u)  ; 0]
"""

# Construct block matrices for first-order system
n_dofs = size(mass_matrix, 1)
Z = spzeros(n_dofs, n_dofs)
# Inertia matrix B = [M, 0; 0, M]
# B_full = [mass_matrix zeros(n_dofs, n_dofs);
#           zeros(n_dofs, n_dofs) mass_matrix]
B_full = [mass_matrix Z;
          Z mass_matrix]
# Jacobian matrix A = [-C, -K; M, 0]
# A_full = [-damping_matrix -stiffness_matrix;
#           mass_matrix zeros(n_dofs, n_dofs)]
A_full = [-damping_matrix -stiffness_matrix;
          mass_matrix Z]
# Define nonlinear terms for FullOrderModel of First Order
# Note: Current implementation needs fe space to be available in scope and is not efficiently implemented for complex values
function evaluate_polynomial_nonlinearity!(res, vec1, vec2)
    N = length(vec1)
    N2 = N / 2
    u1r = FEFunction(V, real(vec1[N2 + 1, :]))
    u1i = FEFunction(V, imag(vec1[N2 + 1, :]))
    u2r = FEFunction(V, real(vec2[N2 + 1, :]))
    u2i = FEFunction(V, imag(vec2[N2 + 1, :]))
    res .+= [
        assemble_vector(v -> (g_quad(u1r, u2r, v) - g_quad(u1i, u2i, v)), V), zeros(N2)]
    res .+= [
        assemble_vector(v -> (g_quad(u1r, u2i, v) + g_quad(u1i, u2r, v)), V), zeros(N2)]
end
function evaluate_polynomial_nonlinearity!(res, vec1, vec2, vec3)
    N = length(vec1)
    N2 = N / 2
    u1r = FEFunction(V, real(vec1[N2 + 1, :]))
    u2r = FEFunction(V, real(vec2[N2 + 1, :]))
    u3r = FEFunction(V, real(vec3[N2 + 1, :]))
    u1i = FEFunction(V, imag(vec1[N2 + 1, :]))
    u2i = FEFunction(V, imag(vec2[N2 + 1, :]))
    u3i = FEFunction(V, imag(vec3[N2 + 1, :]))
    res .+= [
        assemble_vector(
            v -> (h_cube(u1r, u2r, u3r, v) - h_cube(u1r, u2i, u3i, v) -
                  h_cube(u1i, u2r, u3i, v) - h_cube(u1i, u2i, u3r, v)),
            V),
        zeros(N2)]
    res .+= [
        assemble_vector(
            v -> (h_cube(u1i, u2r, u3r, v) + h_cube(u1r, u2i, u3r, v) +
                  h_cube(u1r, u2r, u3i, v) - h_cube(u1i, u2i, u3i, v)),
            V),
        zeros(N2)]
end

@inline evaluate_polynomial_nonlinearity!(res, args...) = nothing

model = FullOrderModel{Float64}(
    B_full, A_full, evaluate_polynomial_nonlinearity!, 3, nothing, nothing)
