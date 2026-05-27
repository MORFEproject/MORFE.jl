"""
    check_eigenvalue_scaling.jl

Compute dω₀/dθ|_{θ=0} from the FEM eigenvalue problem directly,
without using the DPIM ROM.

Assembles K(θ) and M(θ) at θ = ±δ (δ = 0.001), solves for the
first eigenfrequency with Arpack, and evaluates the central-difference
derivative.  Compares against the ROM polynomial coefficient and the
Euler-Bernoulli (1+θ)^{-2} prediction.
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Ferrite
using FerriteGmsh
using SparseArrays
using LinearAlgebra
using Arpack
using Tensors
using Printf

include(joinpath(@__DIR__, "theta_polynomials.jl"))
include(joinpath(@__DIR__, "parametric_geometry.jl"))
include(joinpath(@__DIR__, "parametric_assembly.jl"))

# ------------------------------------------------------------------
# 1.  Mesh and FE space  (same as parametric_beam_demo.jl)
# ------------------------------------------------------------------
const _msh = joinpath(@__DIR__, "..", "BenchmarkFerrite", "beam_h27.msh")
isfile(_msh) || error("Mesh not found.  Run generate_beam_mesh.jl in BenchmarkFerrite/ first.")

println("Loading mesh …")
grid = togrid(_msh)
ip = Lagrange{RefHexahedron, 2}()^3
geo_ip = Lagrange{RefHexahedron, 2}()
qr = QuadratureRule{RefHexahedron}(3)
cv = CellValues(qr, ip, geo_ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch)
update!(ch, 0.0)

free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))

# ------------------------------------------------------------------
# 2.  Material (same as parametric_beam_demo.jl)
# ------------------------------------------------------------------
E_mod = 160e3
ν = 0.22
ρ = 2.32e-3
λ_lame = (E_mod * ν) / ((1 + ν) * (1 - 2ν))
μ_lame = E_mod / (2 * (1 + ν))

# ------------------------------------------------------------------
# 3.  Parametric geometry  (J₁ = e₁⊗e₁, axial stretch)
# ------------------------------------------------------------------
J₀ = one(Tensor{2, 3, Float64})
J₁ = Tensor{2, 3, Float64}((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0)

# Only need first order for the derivative: N_θ = 1
const N_θ_check = 1
detJ_coeffs, adjJ_coeffs = det_and_adj_series(J₀, J₁)
inv_detJ_coeffs = reciprocal_series(detJ_coeffs, N_θ_check)

# ------------------------------------------------------------------
# 4.  Assemble K_k (k=0,1) and M_k (k=0,1)
# ------------------------------------------------------------------
K_full = [allocate_matrix(dh) for _ in 0:N_θ_check]
M_full = [allocate_matrix(dh) for _ in 0:N_θ_check]

println("Assembling K₀, K₁, M₀, M₁ …")
@time assemble_K_M_polynomial!(K_full, M_full, dh, cv, λ_lame, μ_lame, ρ,
    adjJ_coeffs, detJ_coeffs, inv_detJ_coeffs)

K_k = [Kf[free, free] for Kf in K_full]
M_k = [Mf[free, free] for Mf in M_full]

# ------------------------------------------------------------------
# 5.  Eigenfrequency at a given θ
# ------------------------------------------------------------------
function ω_fem(θ_val; nev = 1)
    K_θ = K_k[1] + θ_val * K_k[2]
    M_θ = M_k[1] + θ_val * M_k[2]
    # sigma=0.0 triggers shift-and-invert; :LM finds eigenvalues closest to sigma
    vals, _ = eigs(K_θ, M_θ; nev = nev, which = :LM, sigma = 0.0, check = 1)
    return sqrt(abs(real(vals[1])))
end

# ------------------------------------------------------------------
# 6.  Central-difference derivative at θ = 0
# ------------------------------------------------------------------
δ = 0.001
println("\nSolving eigenvalue problem at θ = $δ …")
ω_p = ω_fem(+δ)
println("Solving eigenvalue problem at θ = 0 …")
ω_0 = ω_fem(0.0)
println("Solving eigenvalue problem at θ = -$δ …")
ω_m = ω_fem(-δ)

dω_dθ = (ω_p - ω_m) / (2δ)
norm_deriv = dω_dθ / ω_0

@printf "\nω₀( 0)      = %.8f rad/s\n" ω_0
@printf "ω₀(+%.3f)  = %.8f rad/s\n" δ ω_p
@printf "ω₀(-%.3f)  = %.8f rad/s\n" δ ω_m
@printf "\ndω₀/dθ|₀   = %+.6f rad/s\n" dω_dθ
@printf "Normalised  = dω₀/dθ / ω₀ = %+.6f\n" norm_deriv
@printf "\nEuler-Bernoulli (1+θ)^{-2} predicts: -2.000000\n"
@printf "ROM polynomial coefficient:           %+.6f\n" -1.499144
