
using Gridap
using GridapGmsh
using WriteVTK

#=
InputModelGridapMechanical implements an ODE system of second order:
    M d_t² U  +  C d_t U + KU  +  F_nl(U) = F(t) 
with the nonlinearity up to order 3.
=#

mutable struct InputModelGridapMechanical <: InputModelAbstractSecondOrder
    model        # Gridap FEModel
    # FEM Storage
    V  # TestFESpace
    U  # TrialFEspace
    stiffness_form::Function   # a(u,v)
    mass_form::Function        # m(u,v)
    load_form::Function        # f(v)
    quadratic_form::Function
    cubic_form::Function 
    M
    K
    C
    F
end

function InputModelGridapMechanical(mesh_strategy::String, mesh_file::String)
    #TODO implement different strategies / forms ...
    if mesh_strategy == "gmsh"
        model = model_from_gmsh!(mesh_file)
    else
        error("mesh_strategy $(mesh_strategy) not implemented")
    end
    
    order = 2
    degree = 2*order
    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)
    Γ = BoundaryTriangulation(model,tags="Neumann")
    Γ_D = BoundaryTriangulation(model,tags="Dirichlet")
    dΓ = Measure(Γ,degree)
    reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order) 
    V = TestFESpace(model,reffe; conformity=:H1, dirichlet_tags=["Dirichlet"], dirichlet_masks=[(true,true,true)]) # what part of the vector is constraint by Dirichlet constraints
    g(x) = VectorValue(0.0,0.0,0.0)  
    U=TrialFESpace(V, g)

    E = 160e3;
    ν = 0.22
    ρ = 2.32e-3;
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε

    a(u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )dΩ
    m(dt2u, v) = ∫( ρ*dt2u⋅v )dΩ
    l(v) = 0 
    E_nl(u1,u2) = 0.25*((∇(u1)')⋅∇(u2) + (∇(u2)')⋅∇(u1))
    E_nl_grad(∇u1,∇u2) = 0.25*((∇u1')⋅∇u2 + (∇u2')⋅∇u1)
    σ_nln(ε) = λ*tr(ε)*one(TensorValue{3,3,Float64})  + 2*μ*ε
    g_quad(u1,u2,v) = ∫( ε(v) ⊙ (σ_nln(E_nl(u1,u2))) + 0.5*(sym(∇(u1)'⋅∇(v)) ⊙ σ_nln(ε(u2)) 
                                                            + sym(∇(u2)'⋅∇(v)) ⊙ σ_nln(ε(u1))))dΩ
    h_cube(u1,u2,u3,v) = 1/3 * ∫( sym(∇(u1)'⋅∇(v)) ⊙ (σ_nln(E_nl(u2,u3))) + 
                                sym(∇(u2)'⋅∇(v)) ⊙ (σ_nln(E_nl(u1,u3))) + 
                                sym(∇(u3)'⋅∇(v)) ⊙ (σ_nln(E_nl(u1,u2))))dΩ

    return InputModelGridapMechanical(
        model,
        V,
        U,
        m, a, l, g_quad, h_cube, 
        nothing, nothing, nothing, nothing)
end

# helper Function for InputModelGridapMechanical
function model_from_gmsh!(gmsh_file::String)
    model = GmshDiscreteModel(gmsh_file)
    return model
end

function assemble_mass_matrix!(fem::InputModelGridapMechanical)
    fem.M = assemble_matrix((u,v)->fem.mass_form(u,v), fem.U, fem.V)
end

function assemble_stiffness_matrix!(fem::InputModelGridapMechanical)
    fem.K = assemble_matrix((u,v)->fem.stiffness_form(u,v), fem.U, fem.V)
end

function assemble_load_vector!(fem::InputModelGridapMechanical)
    fem.f = assemble_vector(v->fem.load_form(v), fem.U, fem.V)
end

function mass_matrix(fem::InputModelGridapMechanical)
     if isnothing(fem.M)
        error("Mass matrix not assembled")
    end
    return fem.M
end

function stiffness_matrix(fem::InputModelGridapMechanical)
     if isnothing(fem.K)
        error("Stiffness matrix not assembled")
    end
    return fem.K
end

function damping_matrix(fem::InputModelGridapMechanical)
     if isnothing(fem.C)
        error("Damping matrix not assembled")
    end
    return fem.C
end

function load_vector(fem::InputModelGridapMechanical)
     if isnothing(fem.F)
        error("Load vector not assembled")
    end
    return fem.F
end

function evaluate_nonlinearity(fem::InputModelGridapMechanical, Ψ...)
    if length(Ψ)==2
        Ψ₁, Ψ₂ = Ψ
        return evaluate_quadratic_nonlinearity(fem, Ψ₁, Ψ₂)
    elseif length(Ψ)==3
        Ψ₁, Ψ₂, Ψ₃ = Ψ
        return evaluate_cubic_nonlinearity(fem, Ψ₁, Ψ₂, Ψ₃)
    else
        error("evaluate_nonlinearity not defined for $(typeof(fem)) at order $(length(Ψ))")
    end
end

function evaluate_quadratic_nonlinearity(fem::InputModelGridapMechanical, Ψ₁,Ψ₂)
    Ψ1r  = FEFunction(fem.V,real(Ψ₁))
    Ψ1i  = FEFunction(fem.V,imag(Ψ₁))
    Ψ2r  = FEFunction(fem.V,real(Ψ₂))
    Ψ2i  = FEFunction(fem.V,imag(Ψ₂))
    resR = assemble_vector(v -> (fem.quadratic_form(Ψ1r,Ψ2r,v)-fem.quadratic_form(Ψ1i,Ψ2i,v)), fem.V)
    resI = assemble_vector(v -> (fem.quadratic_form(Ψ1r,Ψ2i,v)+fem.quadratic_form(Ψ1i,Ψ2r,v)), fem.V)
    return (resR .+ im*resI)
end

function evaluate_cubic_nonlinearity(fem::InputModelGridapMechanical, Ψ₁,Ψ₂,Ψ₃)
    Ψ1r  = FEFunction(fem.V,real(Ψ₁))
    Ψ1i  = FEFunction(fem.V,imag(Ψ₁))
    Ψ2r  = FEFunction(fem.V,real(Ψ₂))
    Ψ2i  = FEFunction(fem.V,imag(Ψ₂))
    Ψ3r  = FEFunction(fem.V,real(Ψ₃))
    Ψ3i  = FEFunction(fem.V,imag(Ψ₃))
    resR = assemble_vector(v -> (fem.cubic_form(Ψ1r,Ψ2r,Ψ3r,v) - fem.cubic_form(Ψ1r,Ψ2i,Ψ3i,v) - fem.cubic_form(Ψ1i,Ψ2r,Ψ3i,v) - fem.cubic_form(Ψ1i,Ψ2i,Ψ3r,v)), fem.V)
    resI = assemble_vector(v -> (fem.cubic_form(Ψ1i,Ψ2r,Ψ3r,v) + fem.cubic_form(Ψ1r,Ψ2i,Ψ3r,v) + fem.cubic_form(Ψ1r,Ψ2r,Ψ3i,v) - fem.cubic_form(Ψ1i,Ψ2i,Ψ3i,v)), fem.V)
    return (resR .+ im*resI)
end

function ndofs(fem::InputModelGridapMechanical)
    return num_free_dofs(fem.U)
end

function field_from_vector(fem::InputModelGridapMechanical, u)
    return FEFunction(V,u)
end

function visualize(fem::InputModelGridapMechanical, u; kwargs...)
    error("visualize not implemented for $(typeof(fem))")
end