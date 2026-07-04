"""
	visualise_orbit_snapshot.jl

Single-snapshot post-processing for the KVS DPIM ROM — writes one Paraview VTU file
for the orbit point at maximum |z|.

Reads the orbit from `orbit_max_amplitude.csv` (columns: t, x=Re(z₁), y=Im(z₁)),
evaluates W(z₁, conj(z₁), η) at the snapshot, and writes all velocity/pressure/
vorticity fields (total and perturbation) to a single VTU file.

Prerequisites: run main.jl for Re49.03_ord7 first to produce W.jls and vtk_data.jls.

Usage:
  julia results/paraview/visualise_orbit_snapshot.jl
"""

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

# η for this orbit (constant along the limit cycle, taken from CSV first data row)
const ETA = -0.00186059929953265725 + 0im

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

using Pkg: Pkg
const _demo_root = joinpath(@__DIR__, "../../..")
Pkg.activate(_demo_root)
if !haskey(Pkg.project().dependencies, "WriteVTK")
	Pkg.add("WriteVTK")
end
Pkg.instantiate()

using Serialization
using WriteVTK
using LinearAlgebra
using Printf
using StaticArrays: SVector
using MORFE
using MORFE.Polynomials: DensePolynomial, evaluate

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

const _results = joinpath(@__DIR__, "..", "Re49.03_ord7")
const _data    = joinpath(_results, "data")

isfile(joinpath(_data, "vtk_data.jls")) ||
	error("vtk_data.jls not found in $_data — run main.jl first.")
isfile(joinpath(_data, "W.jls")) ||
	error("W.jls not found in $_data — run main.jl first.")

println("Loading vtk_data.jls …")
d = deserialize(joinpath(_data, "vtk_data.jls"))

println("Loading W.jls …")
W = deserialize(joinpath(_data, "W.jls"))

const OUT_DIR = @__DIR__
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
	expand(v_free, free_dpim, ndofs_total) -> Vector

Insert the free-DOF values `v_free` into a zero vector of length `ndofs_total`.
Constrained DOFs (walls, cylinder) become 0, matching no-slip BCs.
"""
function expand(v_free::AbstractVector, free_dpim, ndofs_total)
	out = zeros(eltype(v_free), ndofs_total)
	out[free_dpim] .= v_free
	return out
end

"""
	scatter_to_nodes(u_full, cell_dofs, cell_connectivity, n_nodes, n_vel)
		-> (vel 2×n_nodes, pres n_nodes)

Extract corner-node velocity (u_x, u_y) and pressure from the full DOF vector.

P2/P1 Taylor-Hood local layout per cell:
  velocity: 12 DOFs, node-major (u_x, u_y per node), nodes 1-3 = corners, 4-6 = edge mids
  pressure:  3 DOFs, nodes 1-3 = corners, placed at gdofs[n_vel+j]
"""
function scatter_to_nodes(
	u_full::AbstractVector,
	cell_dofs,
	cell_connectivity,
	n_nodes::Int,
	n_vel::Int,
)
	T = real(eltype(u_full))
	vel = zeros(T, 2, n_nodes)
	pres = zeros(T, n_nodes)
	vis_v = falses(n_nodes)
	vis_p = falses(n_nodes)
	for (gdofs, conn) in zip(cell_dofs, cell_connectivity)
		for (j, node) in enumerate(conn)   # j = 1, 2, 3 (corner nodes only)
			if !vis_v[node]
				vis_v[node]  = true
				vel[1, node] = real(u_full[gdofs[2j-1]])   # u_x
				vel[2, node] = real(u_full[gdofs[2j]])     # u_y
			end
			if !vis_p[node]
				vis_p[node] = true
				pres[node]  = real(u_full[gdofs[n_vel+j]])   # p
			end
		end
	end
	return vel, pres
end

"""
	vtk_write(filename, points, cells, vel, pres; extra...)

Write a single VTU file. `extra` is a sequence of (name, array) pairs
for additional point-data arrays.
"""
function vtk_write(filename, points, cells, vel, pres; extra...)
	vtk_grid(filename, points, cells) do vtk
		vtk_point_data(vtk, vel, "velocity")
		vtk_point_data(vtk, pres, "pressure")
		for (name, arr) in extra
			vtk_point_data(vtk, arr, string(name))
		end
	end
	println("  → $(basename(filename)).vtu")
end

"""
	compute_vorticity(vel, node_coords, cell_connectivity, n_nodes) -> Vector

Nodal vorticity ω_z = ∂uy/∂x − ∂ux/∂y via area-weighted average over the
linear (P1) corner triangles.  Works on the scattered corner-node velocity
`vel` (2 × n_nodes) from `scatter_to_nodes`.
"""
function compute_vorticity(
	vel::AbstractMatrix,
	node_coords,
	cell_connectivity,
	n_nodes::Int,
)
	ω = zeros(Float64, n_nodes)
	w = zeros(Float64, n_nodes)
	for conn in cell_connectivity
		i1, i2, i3 = Int(conn[1]), Int(conn[2]), Int(conn[3])
		x1, y1 = node_coords[1, i1], node_coords[2, i1]
		x2, y2 = node_coords[1, i2], node_coords[2, i2]
		x3, y3 = node_coords[1, i3], node_coords[2, i3]
		detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
		abs(detJ) < eps() && continue
		area = 0.5 * abs(detJ)
		duy_dx = ((y3 - y1) * (vel[2, i2] - vel[2, i1]) - (y2 - y1) * (vel[2, i3] - vel[2, i1])) / detJ
		dux_dy = (-(x3 - x1) * (vel[1, i2] - vel[1, i1]) + (x2 - x1) * (vel[1, i3] - vel[1, i1])) / detJ
		ω_e = duy_dx - dux_dy
		for i in (i1, i2, i3)
			ω[i] += area * ω_e
			w[i] += area
		end
	end
	for i in 1:n_nodes
		w[i] > 0 && (ω[i] /= w[i])
	end
	return ω
end

# ─────────────────────────────────────────────────────────────────────────────
# VTK geometry (linear triangles, z = 0)
# ─────────────────────────────────────────────────────────────────────────────

n_nodes = size(d.node_coords, 2)
points  = vcat(d.node_coords, zeros(Float64, 1, n_nodes))   # 3 × n_nodes
cells   = [MeshCell(VTKCellTypes.VTK_TRIANGLE, conn) for conn in d.cell_connectivity]

# ─────────────────────────────────────────────────────────────────────────────
# W polynomial (first-order slice)
# ─────────────────────────────────────────────────────────────────────────────

C    = MORFE.ParametrisationMethod.coefficients(W)      # (FOM, 1, L)
mset = MORFE.ParametrisationMethod.multiindex_set(W)
W1   = DensePolynomial(@view(C[:, 1, :]), mset)          # (FOM, L)

# ─────────────────────────────────────────────────────────────────────────────
# Base flow
# ─────────────────────────────────────────────────────────────────────────────

println("\n[1/2] Base flow …")
s0_full     = expand(d.s0_free_dpim, d.free_dpim, d.ndofs_total)
vel0, pres0 = scatter_to_nodes(s0_full, d.cell_dofs, d.cell_connectivity,
n_nodes, d.n_vel_dofs_per_cell)
vort0       = compute_vorticity(vel0, d.node_coords, d.cell_connectivity, n_nodes)

# ─────────────────────────────────────────────────────────────────────────────
# Orbit snapshot — max |z|
# ─────────────────────────────────────────────────────────────────────────────

println("\n[2/2] Orbit snapshot at max |z| …")

# Parse CSV (columns: t, x=Re(z₁), y=Im(z₁); η is constant = ETA above)
csv_path = joinpath(_data, "orbit_max_amplitude.csv")
lines    = readlines(csv_path)
rows     = [begin
	p = split(strip(l), ',')
	(parse(Float64, p[1]), parse(Float64, p[2]), parse(Float64, p[3]))
end for l in lines[2:end] if length(split(strip(l), ',')) >= 3 && !isempty(strip(l))]

ts = [r[1] for r in rows]
xs = [r[2] for r in rows]
ys = [r[3] for r in rows]

k  = argmax(hypot.(xs, ys))
z1 = xs[k] + im * ys[k]
z  = SVector(z1, conj(z1), ETA)
@printf("  max |z| = %.6e  at  t = %.6f  (row %d)\n", abs(z1), ts[k], k)
@printf("  z₁ = %+.4e %+.4e·i,  η = %.4e\n", real(z1), imag(z1), real(ETA))

# Evaluate W
wz = evaluate(W1, z)
δu_free = real.(wz)

δu_full = expand(δu_free, d.free_dpim, d.ndofs_total)
u_full = expand(real.(d.s0_free_dpim) .+ δu_free, d.free_dpim, d.ndofs_total)

vel_tot, pres_tot   = scatter_to_nodes(u_full, d.cell_dofs, d.cell_connectivity,
n_nodes, d.n_vel_dofs_per_cell)
vel_pert, pres_pert = scatter_to_nodes(δu_full, d.cell_dofs, d.cell_connectivity,
n_nodes, d.n_vel_dofs_per_cell)

vort_tot  = compute_vorticity(vel_tot, d.node_coords, d.cell_connectivity, n_nodes)
vort_pert = compute_vorticity(vel_pert, d.node_coords, d.cell_connectivity, n_nodes)

fname = joinpath(OUT_DIR, "orbit_snapshot_Re49.03_ord7")
vtk_grid(fname, points, cells) do vtk
	vtk_point_data(vtk, vel_tot, "velocity")
	vtk_point_data(vtk, pres_tot, "pressure")
	vtk_point_data(vtk, vort_tot, "vorticity_total")
	vtk_point_data(vtk, vel0, "velocity_base")
	vtk_point_data(vtk, pres0, "pressure_base")
	vtk_point_data(vtk, vort0, "vorticity_base")
	vtk_point_data(vtk, vel_pert, "velocity_perturbation")
	vtk_point_data(vtk, pres_pert, "pressure_perturbation")
	vtk_point_data(vtk, vort_pert, "vorticity_perturbation")
end
println("  → orbit_snapshot_Re49.03_ord7.vtu")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

println("\nParaview file written to: $OUT_DIR")
println("  orbit_snapshot_Re49.03_ord7.vtu")
println("\nParaview tips:")
println("  • colour by 'vorticity_total'        — full physical vorticity at this orbit snapshot")
println("  • colour by 'vorticity_perturbation' — W-predicted deviation from base flow")
println("  • colour by 'vorticity_base'         — steady base-flow vorticity (reference)")
