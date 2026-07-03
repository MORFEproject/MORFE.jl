"""
	validate_tke.jl — independent check of the Python TKE pipeline.

Reconstructs the perturbation field along the orbit with MORFE's own polynomial
evaluator (`evaluate(W1, SVector(z, conj(z), η))`) and computes the period-averaged
fluctuation kinetic energy by DIRECT integration in the full DOF space.

The reference-paper TKE (main_workingVersion.tex:540-547) is

	⟨TKE⟩ = (1/|Ω|)(1/T) ∫_Ω ∫_0^T ½ ‖u'(x,t)−ū(x)‖² dt dx,

which in FEM discretizes to  ½ u'_vel^T M_vel u'_vel / |Ω|  (velocity DOFs only,
pressure excluded).  This script assembles M_vel fresh and provides an independent
cross-check of the Gram-matrix path in run_tke.py.
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using MORFE
using MORFE.Polynomials: DensePolynomial, evaluate
using Ferrite
using FerriteGmsh
using LinearAlgebra
using SparseArrays
using Serialization
using StaticArrays: SVector
using Printf

const EXAMPLE_DIR = realpath(joinpath(@__DIR__, ".."))
include(joinpath(EXAMPLE_DIR, "config.jl"))
include(joinpath(EXAMPLE_DIR, "fem", "mesh.jl"))
include(joinpath(EXAMPLE_DIR, "fem", "fem_setup.jl"))

length(ARGS) >= 1 ||
	error("usage: julia --project=. validation/validate_tke.jl results/ReXX.XX_ordN [orbit.csv]")
const DATA_DIR = joinpath(abspath(ARGS[1]), "data")
const ORBIT_CSV = length(ARGS) >= 2 ? abspath(ARGS[2]) :
				  joinpath(DATA_DIR, "orbit_max_amplitude.csv")
const ETA = -0.00003571047174310654 + 0im

# ── velocity-DOF mask + mass matrix + area ───────────────────────────────────
function velocity_dof_mask(fom)
	is_vel = falses(ndofs(fom.dh))
	for element in CellIterator(fom.dh)
		cd = celldofs(element)
		for i in fom.dof_range_u
			is_vel[cd[i]] = true
		end
	end
	return is_vel
end

function assemble_velocity_mass_full(fom)
	n_dpc = ndofs_per_cell(fom.dh)
	n_vel = length(fom.dof_range_u)
	M     = Ferrite.allocate_matrix(fom.dh)
	fill!(M, 0.0)
	Me    = zeros(n_dpc, n_dpc)
	asm   = start_assemble(M)
	for element in CellIterator(fom.dh)
		reinit!(fom.cv_vel, element)
		fill!(Me, 0.0)
		for q in 1:getnquadpoints(fom.cv_vel)
			dΩ = getdetJdV(fom.cv_vel, q)
			for i in 1:n_vel
				ri = fom.dof_range_u[i]
				φᵢ = shape_value(fom.cv_vel, q, i)
				for j in 1:n_vel
					rj = fom.dof_range_u[j]
					φⱼ = shape_value(fom.cv_vel, q, j)
					Me[ri, rj] += (φᵢ ⋅ φⱼ) * dΩ
				end
			end
		end
		assemble!(asm, celldofs(element), Me)
	end
	return M
end

function domain_area(fom)
	area = 0.0
	for element in CellIterator(fom.dh)
		reinit!(fom.cv_vel, element)
		for q in 1:getnquadpoints(fom.cv_vel)
			area += getdetJdV(fom.cv_vel, q)
		end
	end
	return area
end

trapz(t, f) = sum(0.5 * (f[i] + f[i+1]) * (t[i+1] - t[i]) for i in 1:length(t)-1)

# ── load W and orbit ─────────────────────────────────────────────────────────
W    = deserialize(joinpath(DATA_DIR, "W.jls"))
C    = MORFE.ParametrisationMethod.coefficients(W)
mset = MORFE.ParametrisationMethod.multiindex_set(W)
W1   = DensePolynomial(C[:, 1, :], mset)
FOM  = size(C, 1)

rows = Tuple{Float64, Float64, Float64}[]
for l in readlines(ORBIT_CSV)[2:end]
	p = split(strip(l), ',')
	length(p) >= 3 && !isempty(strip(l)) &&
		push!(rows, (parse(Float64, p[1]), parse(Float64, p[2]), parse(Float64, p[3])))
end
ts = [r[1] for r in rows]; xs = [r[2] for r in rows]; ys = [r[3] for r in rows]
Ns = length(ts); T = ts[end] - ts[1]
@printf("orbit: %d samples, T = %.6g, |z|max = %.6e, η = %.6e\n",
	Ns, T, maximum(hypot.(xs, ys)), real(ETA))

# ── FEM setup + velocity DOFs + M_vel ────────────────────────────────────────
fom = setup_fem(joinpath(EXAMPLE_DIR, "fem", "cylinder_flow.msh"))
dofset = length(fom.free_dpim) == FOM ? fom.free_dpim :
		 length(fom.free) == FOM ? fom.free :
		 error("W FOM=$FOM matches neither free_dpim nor free")
is_vel    = velocity_dof_mask(fom)
vel_rows  = findall(d -> is_vel[d], dofset)
vel_gdofs = dofset[vel_rows]
M_full    = assemble_velocity_mass_full(fom)
M_vel     = M_full[vel_gdofs, vel_gdofs]
area      = domain_area(fom)
@printf("velocity DOFs: %d / %d, |Ω| = %.6e\n", length(vel_rows), FOM, area)

# ── reconstruct field along the orbit, integrate fluctuation KE ──────────────
U = Matrix{Float64}(undef, Ns, length(vel_rows))
for k in 1:Ns
	z = SVector(xs[k] + im * ys[k], xs[k] - im * ys[k], ETA)
	U[k, :] = real.(evaluate(W1, z))[vel_rows]
end

nvel = length(vel_rows)
ubar = [trapz(ts, @view U[:, d]) / T for d in 1:nvel]
Uf   = U .- ubar'
MUf  = M_vel * Uf'                 # n_vel × Ns: M_vel applied per snapshot
KE   = [0.5 * dot(@view(Uf[k, :]), @view(MUf[:, k])) / area for k in 1:Ns]
TKE  = trapz(ts, KE) / T

@printf("\nindependent Julia TKE = %.10e\n", TKE)
@printf("KE range = [%.6e, %.6e]\n", minimum(KE), maximum(KE))
@printf("(compare to run_tke.py for the same orbit/η — must agree to ~10 sig. figs)\n")
