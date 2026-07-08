"""
Paraview deformation visualisation for the arch_2_force backbone.

Loads W from results/ and writes results/paraview/deformation_*.vtu with two
vector point-data fields:
  u_total      — full deformation Re[W(z₁, z̄₁)]
  u_nonlinear  — nonlinear part   Re[W(z₁, z̄₁) − W_lin(z₁, z̄₁)]

Usage:
  julia --project demo/ArchComsolWedge/arch_2_force/results/mode_1_order_9_cnf/postprocess/visualise_deformation.jl

Requires results/W.jls (produced by arch_2_force.jl).
No eigenproblem solve or assembly — only mesh setup + polynomial evaluation.
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, "../../.."))
if !haskey(Pkg.project().dependencies, "WriteVTK")
    Pkg.add("WriteVTK")
end
Pkg.instantiate()

using MORFE
using Ferrite
using WriteVTK
using Serialization
using Printf

include(joinpath(@__DIR__, "../../../mesh/load_mesh.jl"))

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# Modal amplitude |z₁| at which to evaluate the deformation (mm units).
# Set to the max backbone amplitude (r_cap from compute_backbone.jl).
const r_amplitude = 5e-4

# Phase of z₁: 0 → real displacement aligned with eigenvector (peak deformation).
const theta_phase = 0.0

# -----------------------------------------------------------------------
# §1  Mesh + DOF setup  (no assembly, no eigenproblem)
# -----------------------------------------------------------------------

const _mesh_path = joinpath(@__DIR__, "../../../arch_2_force.mphtxt")
isfile(_mesh_path) || error("Mesh not found: $_mesh_path")

println("\n§1  Loading mesh …")
(grid, constrained_nodes) = load_arch_mesh(_mesh_path)

const ip = Lagrange{RefPrism, 2}()^3

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, constrained_nodes, (x, t) -> zeros(3), [1, 2, 3]))
close!(ch)
update!(ch, 0.0)

free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
println("  Free DOFs: ", length(free))

# -----------------------------------------------------------------------
# §2  Load ROM
# -----------------------------------------------------------------------

const _results = joinpath(@__DIR__, "..")
isfile(joinpath(_results, "W.jls")) ||
    error("results/W.jls not found — run arch_2_force.jl first.")

println("\n§2  Loading W …")
W = deserialize(joinpath(_results, "W.jls"))

# -----------------------------------------------------------------------
# §3  Write VTK
# -----------------------------------------------------------------------

println("\n§3  Writing deformation VTK …")
@printf("  r_amplitude = %.2e mm   theta = %.4f rad\n", r_amplitude, theta_phase)

const _outdir = joinpath(_results, "paraview")
write_paraview_deformation(
    _outdir, grid, dh, W, free, r_amplitude;
    theta = theta_phase,
    label = @sprintf("deformation_r%s", replace(@sprintf("%.2e", r_amplitude), "." => "p")),
)

println("\nDone.  Open results/paraview/deformation_*.vtu in Paraview.")
println("  → Filters → Warp by Vector → u_total   (full deformation)")
println("  → Filters → Warp by Vector → u_nonlinear  (nonlinear part only)")
