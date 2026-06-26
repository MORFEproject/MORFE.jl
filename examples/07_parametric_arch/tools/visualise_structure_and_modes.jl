"""
Paraview visualisation for the parametric arch beam (example 07).

Exports to results/paraview/:
  mesh.vtu              — arched geometry at θ = θ_vis (open to inspect the mesh)
  mode_01.vtu …         — Re_u field per eigenmode; use Filters → Warp by Vector → Re_u
  modes.pvd             — PVD collection: open and step through modes with the time slider
  deformation_*.vtu     — full deformation Re[W(z₁,z̄₁,θ)] at a backbone amplitude point;
						  fields: u_total (full), u_nonlinear (nonlinear correction only)

Usage:
  julia --project tools/visualise_structure_and_modes.jl

Requires results/data/arch_h<h0_L_ratio>/W.jls (run main.jl first).
No cohomological solve is performed here.
"""

# -----------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------
using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../../..")))
	Pkg.add([
		"Ferrite", "FerriteGmsh",
		"Arpack", "LinearMaps",
		"Tensors", "StaticArrays",
		"Serialization", "WriteVTK",
	])
end
if !haskey(Pkg.project().dependencies, "WriteVTK")
	Pkg.add("WriteVTK")
end
Pkg.instantiate()

# loading Ferrite + WriteVTK activates MORFEWriteVTKExt automatically
using MORFE
using Ferrite
using FerriteGmsh
using WriteVTK
using Arpack, LinearMaps
using Tensors
using StaticArrays
using Serialization
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "config.jl"))             # h0_L_ratio, N_INCREMENTS
include(joinpath(@__DIR__, "..", "fem", "theta_polynomials.jl"))
include(joinpath(@__DIR__, "..", "fem", "parametric_geometry.jl"))
include(joinpath(@__DIR__, "..", "fem", "arch_geometry.jl"))
include(joinpath(@__DIR__, "..", "fem", "arch_assembly.jl"))

# -----------------------------------------------------------------------
# Configuration  (edit here)
# -----------------------------------------------------------------------

const N_MODES = 6      # number of eigenmodes to export
const r_amplitude = 30.0   # modal amplitude |z₁| for deformation snapshot
const θ_vis = 9.0    # arch parameter for visualisation (0 = base arch; 9 → 50 mm arch)

const L = 1000.0
const h₀ = h0_L_ratio * L

# -----------------------------------------------------------------------
# §1  Mesh + DOF setup
# -----------------------------------------------------------------------

const _msh = joinpath(@__DIR__, "..", "..", "..", "benchmark", "ferrite",
	"beam_h27_10x2x2.msh")
isfile(_msh) || error("Mesh not found at $_msh.  Run generate_beam_mesh.jl in benchmark/ferrite/ first.")

println("\n§1  Loading mesh …")
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
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free = length(free)

@printf "  Total DOFs : %d\n" ndofs(dh)
@printf "  Free DOFs  : %d\n" n_free

# -----------------------------------------------------------------------
# §2  Arch node coordinates at θ_vis
# -----------------------------------------------------------------------

println("\n§2  Computing arched node positions (θ = $θ_vis) …")
arch_points = let
	pts = Matrix{Float64}(undef, 3, length(grid.nodes))
	for (i, node) in enumerate(grid.nodes)
		x0 = node.x
		w = h₀ * (1 + θ_vis) * sin(π * x0[1] / L)
		pts[1, i] = x0[1]
		pts[2, i] = x0[2] + w
		pts[3, i] = x0[3]
	end
	pts
end
@printf "  Arch rise at midpoint: %.3f mm\n" (h₀ * (1 + θ_vis))

# -----------------------------------------------------------------------
# §3  Assemble K₀, M₀  (base arch, θ = 0)
# -----------------------------------------------------------------------

const _E = 160e3
const _ν = 0.22
const _ρ = 2.32e-3
const _λ = (_E * _ν) / ((1 + _ν) * (1 - 2_ν))
const _μ = _E / (2 * (1 + _ν))

h₀_vis = h₀ * (1 + θ_vis)   # physical arch height at θ_vis (50 mm for θ_vis = 9)
println("\n§3  Assembling K, M at arch height $(round(h₀_vis; digits=1)) mm …")
K_arr_full = [allocate_matrix(dh)]
M_arr_full = [allocate_matrix(dh)]
@time assemble_K_M_arch!(K_arr_full, M_arr_full, dh, cv,
	_λ, _μ, _ρ, h₀_vis, L, free_to_local, 0)

K = K_arr_full[1][free, free]
M = M_arr_full[1][free, free]

# -----------------------------------------------------------------------
# §4  Eigenproblem
# -----------------------------------------------------------------------

println("\n§4  Eigenproblem ($N_MODES modes) …")
solver_eig = StructureModalDampingEigensolver(N_MODES, 0.0, 0.0)
t_eig = @timed solve_eigenproblem(K, M, solver_eig; sorter! = (args...) -> nothing)
eigenproblem = t_eig.value
(eigenvalues, Y, _) = get_eigenpairs(eigenproblem)

println("  Eigenfrequencies (Hz):")
for i in 1:min(N_MODES, length(eigenvalues))
	@printf "    mode %2d:  %10.4f Hz\n" i abs(eigenvalues[i]) / (2π)
end
@printf "  (solved in %.1f s)\n" t_eig.time

# -----------------------------------------------------------------------
# §5  Load parametric ROM W
# -----------------------------------------------------------------------

const _param_dir = joinpath(@__DIR__, "..", "results", "data",
	@sprintf("arch_h%.3f", h0_L_ratio))
isfile(joinpath(_param_dir, "W.jls")) ||
	error("ROM not found at $_param_dir/W.jls — run main.jl first.")

println("\n§5  Loading W …")
W = deserialize(joinpath(_param_dir, "W.jls"))

# -----------------------------------------------------------------------
# §6  VTK export
# -----------------------------------------------------------------------

const _out = joinpath(@__DIR__, "..", "results", "paraview")
println("\n§6  Writing Paraview files to $(_out)/ …")

write_paraview_mesh(joinpath(_out, "mesh"), grid;
	dh = dh, prescribed_dofs = ch.prescribed_dofs,
	node_positions = arch_points)

write_paraview_modes(_out, grid, dh, eigenvalues, Y, free;
	n_modes = N_MODES, node_positions = arch_points, re_only = true)

lbl = @sprintf("deformation_r%.1f_theta0.00", r_amplitude)
write_paraview_deformation(_out, grid, dh, W, free, r_amplitude;
	theta = 0.0,
	extra_states = ComplexF64[0.0],   # base-arch ROM (θ_vis=9 is outside convergence radius)
	node_positions = arch_points,
	label = lbl)

println("\nDone.")
@printf "  Arch height visualised: %.1f mm\n" h₀_vis
println("  Open results/paraview/mesh.vtu       — arched geometry")
println("  Open results/paraview/modes.pvd      — step through eigenmodes (Warp by Vector → Re_u)")
println("  Open results/paraview/$lbl.vtu  — deformation at base-arch ROM point (Warp by Vector → u_total)")
