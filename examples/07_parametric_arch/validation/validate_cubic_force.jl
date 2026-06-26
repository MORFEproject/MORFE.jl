"""
validate_cubic_force.jl
========================
Validates the parametric assembly of the cubic force H(u1, u2, u3; θ) by
comparing the Taylor-polynomial approximation against the exact force at
the physical arch height h₀(1+θ).

Methodology
-----------
  H_exact(u1,u2,u3;θ) — assembled directly at arch height h₀(1+θ) (k=0 term
                          of the expansion *around that exact geometry*).
  H_approx(u1,u2,u3;θ) — Σ_{k=0}^{N} H_k(u1,u2,u3) · θ^k  using the parametric
                           maps from the base arch at h₀.

For each truncation order N and each θ, the mean relative error over a
Gaussian cloud of (u1, u2, u3) triples is written to a CSV and printed.

Analytical expectation
----------------------
Since J(θ) = J₀ + J₁θ with J₁ rank-1 nilpotent (J₁² = 0, det J = 1),
adj(J(θ)) is degree-1 in θ.  The cubic integrand involves two E_nl_adj
factors (each degree 2 as a product of two degree-1 adj series), so:
  H(u1,u2,u3;θ)  is an exact polynomial of degree 4 in θ.
  H_k = 0 for k > 4.

  - N=1,2,3,4 : error scales as O(|θ|^{N+1})  → log-log slopes 2, 3, 4, 5
  - N ≥ 5     : error ≈ machine precision       (remaining H_k are analytically zero)

A failure (H_k ≠ 0 for k > 4) appears as N ≥ 5 lines NOT collapsing to zero.

Output
------
  results/validation/cubic_force_errors.csv
  (also printed as a table)
"""

# -----------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------
using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Ferrite
using FerriteGmsh
using Tensors
using StaticArrays
using LinearAlgebra
using Printf
using Statistics
using Random

include(joinpath(@__DIR__, "..", "fem", "theta_polynomials.jl"))
include(joinpath(@__DIR__, "..", "fem", "parametric_geometry.jl"))
include(joinpath(@__DIR__, "..", "fem", "arch_geometry.jl"))
include(joinpath(@__DIR__, "..", "fem", "arch_assembly.jl"))
include(joinpath(@__DIR__, "..", "config.jl"))

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------

const N_TH_ORDERS = [1, 2, 3, 4, 5, 6, 7]
const N_TH_MAX    = maximum(N_TH_ORDERS)
const THETA_VALS  = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
const N_SAMPLES   = 15

const L  = 1000.0
const h₀ = h0_L_ratio * L

# -----------------------------------------------------------------------
# §1  Mesh + DOF setup
# -----------------------------------------------------------------------

const _msh = joinpath(@__DIR__, "..", "..", "..", "benchmark", "ferrite",
	"beam_h27_10x2x2.msh")
isfile(_msh) || error("Mesh not found: $_msh  (run generate_beam_mesh.jl first)")

println("§1  Loading mesh …")
grid = togrid(_msh)

ip     = Lagrange{RefHexahedron, 2}()^3
geo_ip = Lagrange{RefHexahedron, 2}()
qr     = QuadratureRule{RefHexahedron}(3)
cv     = CellValues(qr, ip, geo_ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch)
update!(ch, 0.0)

free          = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free        = length(free)

@printf "    Total DOFs : %d\n" ndofs(dh)
@printf "    Free  DOFs : %d\n" n_free

# -----------------------------------------------------------------------
# §2  Material
# -----------------------------------------------------------------------

const _E = 160e3
const _ν = 0.22
const _λ = (_E * _ν) / ((1 + _ν) * (1 - 2_ν))
const _μ = _E / (2 * (1 + _ν))

# -----------------------------------------------------------------------
# §3  Pre-compute H_k(u1_s, u2_s, u3_s) for k=0..N_TH_MAX (θ-independent)
# -----------------------------------------------------------------------

println("\n§3  Pre-computing H_k coefficients (k = 0 … $N_TH_MAX, $N_SAMPLES samples) …")

pgn_base = ArchGeometricNonlinearity{3}(
	dh, cv, _λ, _μ, h₀, L, free_to_local, n_free, N_TH_MAX)

rng = MersenneTwister(42)
u1_samples = [randn(rng, n_free) for _ in 1:N_SAMPLES]
u2_samples = [randn(rng, n_free) for _ in 1:N_SAMPLES]
u3_samples = [randn(rng, n_free) for _ in 1:N_SAMPLES]

# H_k_store[k+1][s] = H_k(u1_s, u2_s, u3_s)
H_k_store = [[zeros(Float64, n_free) for _ in 1:N_SAMPLES] for _ in 0:N_TH_MAX]

@time for k in 0:N_TH_MAX
	for s in 1:N_SAMPLES
		evaluate_kth_cubic!(H_k_store[k+1][s], pgn_base, k,
			u1_samples[s], u2_samples[s], u3_samples[s])
	end
	mean_norm = mean(norm(H_k_store[k+1][s]) for s in 1:N_SAMPLES)
	@printf "    k=%d : mean ‖H_k‖ = %.3e\n" k mean_norm
end

# -----------------------------------------------------------------------
# §4  Main loop — compare exact vs approximation at each θ
# -----------------------------------------------------------------------

println("\n§4  Computing errors …")
println("    " * "-"^72)
@printf "    %-8s  %-8s  %-16s  %-16s\n" "theta" "N_theta" "mean_rel_error" "max_rel_error"
println("    " * "-"^72)

struct Row
	theta          :: Float64
	N_theta        :: Int
	mean_rel_error :: Float64
	max_rel_error  :: Float64
end

rows = Row[]

for θ in THETA_VALS
	# Exact reference: pgn at physical arch h₀*(1+θ), N_θ=0, k=0
	h₀_exact = h₀ * (1 + θ)
	pgn_exact = ArchGeometricNonlinearity{3}(
		dh, cv, _λ, _μ, h₀_exact, L, free_to_local, n_free, 0)

	h_exact = [zeros(Float64, n_free) for _ in 1:N_SAMPLES]
	for s in 1:N_SAMPLES
		evaluate_kth_cubic!(h_exact[s], pgn_exact, 0,
			u1_samples[s], u2_samples[s], u3_samples[s])
	end

	for N in N_TH_ORDERS
		rel_errors = Float64[]
		for s in 1:N_SAMPLES
			h_approx = sum(H_k_store[k+1][s] .* (θ^k) for k in 0:N)
			push!(rel_errors, norm(h_exact[s] .- h_approx) / (norm(h_exact[s]) + 1e-30))
		end
		m_rel = mean(rel_errors)
		x_rel = maximum(rel_errors)
		push!(rows, Row(θ, N, m_rel, x_rel))
		@printf "    %-8.3f  %-8d  %-16.3e  %-16.3e\n" θ N m_rel x_rel
	end
end

println("    " * "-"^72)

# -----------------------------------------------------------------------
# §5  Write CSV
# -----------------------------------------------------------------------

_out_dir = joinpath(@__DIR__, "..", "results", "validation")
mkpath(_out_dir)
_out_csv = joinpath(_out_dir, "cubic_force_errors.csv")

open(_out_csv, "w") do io
	println(io, "theta,N_theta,mean_rel_error,max_rel_error")
	for r in rows
		@printf io "%.6f,%d,%.6e,%.6e\n" r.theta r.N_theta r.mean_rel_error r.max_rel_error
	end
end

println("\nSaved → $_out_csv")
println("Run:  python validation/plots/plot_cubic_force.py")
