using Gridap
using Gridap.TensorValues
using SparseArrays, LinearAlgebra, SuiteSparse
using Random
using Base.Threads


println("-----------------------------------------------")
# prototype mesh
order = 1

domain = (0,1, 0,1, 0,1)
d = 4
partition = (d,d,d)
model = CartesianDiscreteModel(domain,partition)

Ω = Triangulation(model)
dΩ = Measure(Ω,order*2)

reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order) 
V = TestFESpace(model,reffe; conformity=:H1, dirichlet_tags="boundary", dirichlet_masks=[(true,true,true)])

g(x)=VectorValue(0.0,0.0,0.0) # dirichlet bound.
U = TrialFESpace(V,g)

N=num_free_dofs(U)
println("N: $N")

const E = 160e3;
const ν = 0.22
const ρ = 2.32e-3;
const λ = (E*ν)/((1+ν)*(1-2*ν))
const μ = E/(2*(1+ν))

σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε
sym(u) = 1/2*(u+u')
ε(u) = sym(∇(u))
a(u,v)      = ∫( ε(v) ⊙ (σ∘ε(u)) )dΩ
m(dt2u, v) = ∫( dt2u⋅v )dΩ
E_nl(u1,u2)  = 0.25*((∇(u1)')⋅∇(u2) + (∇(u2)')⋅∇(u1))
σ_nln(ε) = λ*tr(ε)*one(TensorValue{3,3,Float64})  + 2*μ*ε
g_quad(u1,u2,v)    = ∫( ε(v) ⊙ (σ_nln(E_nl(u1,u2))) 
                        + 0.5*(sym(∇(u1)'⋅∇(v)) ⊙ σ_nln(ε(u2)) + sym(∇(u2)'⋅∇(v)) ⊙ σ_nln(ε(u1))))dΩ

function eval_g_quad_vec!(res, u1_vec, u2_vec)
  u1 = FEFunction(V,u1_vec)
  u2 = FEFunction(V,u2_vec)
  res .= assemble_vector(v -> (g_quad(u1,u2,v)), V)
end

function phi(i, V)
  coeffs = zeros(N)
  coeffs[i] = 1.0
  return FEFunction(V, coeffs)
end
phi1 = phi(1,V)

#Get sparsity pattern
G1 = assemble_matrix((u,v) -> g_quad(u,v,phi1), U, U)
G = [similar(G1) for _ in 1:N]

function calculate_G!(G::Vector, U, V,  g_quad, N, nmax)
  coeffs = zeros(N)
  ϕ = FEFunction(V, coeffs)
  for i in 1:nmax
    fill!(coeffs, 0.0)
    coeffs[i] = 1.0
    G[i] = assemble_matrix((u,v) -> g_quad(u,v,ϕ),  U, U)
  end
end

#Calculate G Tensor
println("Build tensor")
calculate_G!(G, U, V, g_quad, N, 1) # for @time
@time calculate_G!(G, U, V, g_quad, N, N)

function evaluate_G!(res, G::Vector, u1, u2)
  N = length(u1)
  @assert N == length(u2) == length(res) == length(G)

  tmp = zeros(N)
  for k in 1:N
    mul!(tmp, G[k], u2)
    res[k]=dot(u1,tmp)
  end
end

# Test evaluation
u1 = rand(N)
u2 = rand(N)
res_G = zeros(N)
res_v = zeros(N)
println("Evaluate G Tensor")
@time evaluate_G!(res_G, G, u1, u2)
println("Evaluate vectorized")
eval_g_quad_vec!(res_v, u1, u2) 
res_v .= zeros(N)
@time eval_g_quad_vec!(res_v, u1, u2)
println(norm(res_G-res_v))