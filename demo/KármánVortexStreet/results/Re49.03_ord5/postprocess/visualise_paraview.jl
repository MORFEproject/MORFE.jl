"""
	visualise_paraview.jl

Post-processing for the KVS DPIM ROM — writes Paraview-compatible VTK files.

Outputs (written to ../paraview/):
  base_flow.vtu           — steady base flow: velocity vector + pressure
  eigenmode_NNN.vtu       — Re(φₖ) and Im(φₖ) velocity + pressure for each mode k
  mode_NNN_fff.vtu        — linear-orbit animation frames for mode k
  mode_NNN.pvd            — PVD collection for each modal animation
  vortex_NNN.vtu          — N_FRAMES snapshots of one vortex-shedding period (nonlinear W)
  vortex.pvd              — PVD collection for the nonlinear animation

Prerequisites: run main.jl first to produce W.jls and vtk_data.jls.

Usage:
  julia results/Re49.03_ord5/postprocess/visualise_paraview.jl
"""

# ─────────────────────────────────────────────────────────────────────────────
# User-adjustable parameters
# ─────────────────────────────────────────────────────────────────────────────

# Peak perturbation velocity for modal animations (same units as the FEM velocity field).
# Because eigenvectors have arbitrary ARPACK norms, each φₖ is normalised to unit ∞-norm
# before scaling, so this value directly sets the maximum |u_perturbation| in every frame.
const ANIM_AMPLITUDE = 5e-1

# Set to false to skip normalisation and use the raw eigenvector norms instead.
const NORMALIZE_MODES = true

const N_FRAMES = 50     # frames per period for each animation

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

const _results = joinpath(@__DIR__, "..")

isfile(joinpath(_results, "vtk_data.jls")) ||
	error("vtk_data.jls not found — re-run main.jl first (vtk_data bundle was added in the updated main.jl).")
isfile(joinpath(_results, "W.jls")) ||
	error("W.jls not found — run main.jl first.")

println("Loading vtk_data.jls …")
d = deserialize(joinpath(_results, "vtk_data.jls"))

println("Loading W.jls …")
W = deserialize(joinpath(_results, "W.jls"))

const OUT_DIR = joinpath(_results, "paraview")
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
				vel[2, node] = real(u_full[gdofs[2j]])       # u_y
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
	vtk_write(filename, points, cells, vel, pres, extra_fields...)

Write a single VTU file. `extra_fields` is a sequence of (name, array) pairs
for additional point-data arrays (e.g. Re/Im of eigenmode).
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
# W polynomial helper (first-order slice, used for nonlinear vortex animation)
# ─────────────────────────────────────────────────────────────────────────────

C = MORFE.ParametrisationMethod.coefficients(W)      # (FOM, 1, L)
mset = MORFE.ParametrisationMethod.multiindex_set(W)
W1 = DensePolynomial(@view(C[:, 1, :]), mset)          # (FOM, L)

# ─────────────────────────────────────────────────────────────────────────────
# 1 — Base flow
# ─────────────────────────────────────────────────────────────────────────────

println("\n[1/4] Base flow …")
s0_full = expand(d.s0_free_dpim, d.free_dpim, d.ndofs_total)
vel0, pres0 = scatter_to_nodes(s0_full, d.cell_dofs, d.cell_connectivity,
	n_nodes, d.n_vel_dofs_per_cell)
vmag0 = sqrt.(vec(sum(vel0 .^ 2; dims = 1)))
vort0 = compute_vorticity(vel0, d.node_coords, d.cell_connectivity, n_nodes)
vtk_write(joinpath(OUT_DIR, "base_flow"), points, cells, vel0, pres0;
	velocity_magnitude = vmag0,
	vorticity = vort0)

# ─────────────────────────────────────────────────────────────────────────────
# 2 — All eigenmodes
# ─────────────────────────────────────────────────────────────────────────────

n_modes = 4# haskey(d, :all_modes) ? size(d.all_modes, 2) : 1

if haskey(d, :all_modes)
	println("\n[2/4] All eigenmodes ($(n_modes) modes) …")
	for k in 1:n_modes
		local φk_raw = d.all_modes[:, k]
		local λk     = d.all_eigenvalues[k]
		local φk_inf = maximum(abs.(φk_raw))
		local φk     = NORMALIZE_MODES ? φk_raw ./ φk_inf : φk_raw

		local φk_re_full = expand(real.(φk), d.free_dpim, d.ndofs_total)
		local φk_im_full = expand(imag.(φk), d.free_dpim, d.ndofs_total)

		local vel_re_k, pres_re_k = scatter_to_nodes(φk_re_full, d.cell_dofs, d.cell_connectivity,
			n_nodes, d.n_vel_dofs_per_cell)
		local vel_im_k, pres_im_k = scatter_to_nodes(φk_im_full, d.cell_dofs, d.cell_connectivity,
			n_nodes, d.n_vel_dofs_per_cell)
		local vort_re_k = compute_vorticity(vel_re_k, d.node_coords, d.cell_connectivity, n_nodes)
		local vort_im_k = compute_vorticity(vel_im_k, d.node_coords, d.cell_connectivity, n_nodes)

		@printf("  mode %2d:  λ = %+.6f %+.6f·i  ||φ||_∞ = %.4e%s\n",
			k, real(λk), imag(λk), φk_inf, NORMALIZE_MODES ? "  (normalised)" : "")

		local fname = joinpath(OUT_DIR, @sprintf("eigenmode_%03d", k))
		vtk_grid(fname, points, cells) do vtk
			vtk_point_data(vtk, vel_re_k, "velocity_Re")
			vtk_point_data(vtk, pres_re_k, "pressure_Re")
			vtk_point_data(vtk, vort_re_k, "vorticity_Re")
			vtk_point_data(vtk, vel_im_k, "velocity_Im")
			vtk_point_data(vtk, vec(pres_im_k), "pressure_Im")
			vtk_point_data(vtk, vort_im_k, "vorticity_Im")
		end
		println("  → eigenmode_$(lpad(k,3,'0')).vtu")
	end
else
	# Fallback for vtk_data.jls produced by an older main.jl
	println("\n[2/4] Hopf eigenmode (legacy vtk_data bundle — no all_modes field) …")
	phi1 = C[:, 1, 1]
	phi1_re_full = expand(real.(phi1), d.free_dpim, d.ndofs_total)
	phi1_im_full = expand(imag.(phi1), d.free_dpim, d.ndofs_total)
	vel_re, pres_re = scatter_to_nodes(phi1_re_full, d.cell_dofs, d.cell_connectivity,
		n_nodes, d.n_vel_dofs_per_cell)
	vel_im, pres_im = scatter_to_nodes(phi1_im_full, d.cell_dofs, d.cell_connectivity,
		n_nodes, d.n_vel_dofs_per_cell)
	vtk_write(joinpath(OUT_DIR, "eigenmode_001"), points, cells,
		vel_re, pres_re;
		velocity_Im = vel_im,
		pressure_Im = vec(pres_im))
end

# ─────────────────────────────────────────────────────────────────────────────
# 3 — Linear modal animations (one period per mode, using raw eigenvectors)
# ─────────────────────────────────────────────────────────────────────────────

if haskey(d, :all_modes)
	println("\n[3/4] Linear modal animations ($N_FRAMES frames each, amplitude = $ANIM_AMPLITUDE) …")
	thetas = range(0.0, 2π; length = N_FRAMES + 1)[1:N_FRAMES]

	for k in 1:n_modes
		local φk_raw = d.all_modes[:, k]
		local φk     = NORMALIZE_MODES ? φk_raw ./ maximum(abs.(φk_raw)) : φk_raw
		local pvd     = paraview_collection(joinpath(OUT_DIR, @sprintf("mode_%03d", k)))

		for (f, θ) in enumerate(thetas)
			# perturbation only (free-DOF space): 2·ρ·Re(φₖ·exp(iθ))
			local δu_free = 2 .* ANIM_AMPLITUDE .* real.(φk .* cis(θ))
			local u_full = expand(d.s0_free_dpim .+ δu_free, d.free_dpim, d.ndofs_total)
			local δu_full = expand(δu_free, d.free_dpim, d.ndofs_total)

			local vel_tot, pres_tot   = scatter_to_nodes(u_full, d.cell_dofs, d.cell_connectivity,
			n_nodes, d.n_vel_dofs_per_cell)
			local vel_pert, pres_pert = scatter_to_nodes(δu_full, d.cell_dofs, d.cell_connectivity,
			n_nodes, d.n_vel_dofs_per_cell)
			local vmag_tot            = sqrt.(vec(sum(vel_tot .^ 2; dims = 1)))
			local vort_tot            = compute_vorticity(vel_tot, d.node_coords, d.cell_connectivity, n_nodes)
			local vort_pert           = compute_vorticity(vel_pert, d.node_coords, d.cell_connectivity, n_nodes)

			local fname = joinpath(OUT_DIR, @sprintf("mode_%03d_%03d", k, f))
			vtk_grid(fname, points, cells) do vtk
				vtk_point_data(vtk, vel_tot, "velocity")
				vtk_point_data(vtk, pres_tot, "pressure")
				vtk_point_data(vtk, vmag_tot, "velocity_magnitude")
				vtk_point_data(vtk, vort_tot, "vorticity")
				vtk_point_data(vtk, vel0, "velocity_base")
				vtk_point_data(vtk, pres0, "pressure_base")
				vtk_point_data(vtk, vort0, "vorticity_base")
				vtk_point_data(vtk, vel_pert, "velocity_perturbation")
				vtk_point_data(vtk, pres_pert, "pressure_perturbation")
				vtk_point_data(vtk, vort_pert, "vorticity_perturbation")
				pvd[Float64(f)] = vtk
			end
		end

		vtk_save(pvd)
		@printf("  → mode_%03d.pvd  (%d frames)\n", k, N_FRAMES)
	end
end

# ─────────────────────────────────────────────────────────────────────────────
# 4 — Vortex-shedding animation (one period, nonlinear W parametrisation)
# ─────────────────────────────────────────────────────────────────────────────

println("\n[4/4] Nonlinear vortex animation ($N_FRAMES frames, amplitude = $ANIM_AMPLITUDE) …")

pvd    = paraview_collection(joinpath(OUT_DIR, "vortex"))
thetas = range(0.0, 2π; length = N_FRAMES + 1)[1:N_FRAMES]

for (k, θ) in enumerate(thetas)
	local z1 = ANIM_AMPLITUDE * cis(θ)
	local z = SVector(z1, conj(z1), 0.0 + 0.0im)
	local wz = evaluate(W1, z)
	local δu_free = real.(wz)
	local u_full = expand(d.s0_free_dpim .+ δu_free, d.free_dpim, d.ndofs_total)
	local δu_full = expand(δu_free, d.free_dpim, d.ndofs_total)

	local vel_k, pres_k       = scatter_to_nodes(u_full, d.cell_dofs, d.cell_connectivity,
	n_nodes, d.n_vel_dofs_per_cell)
	local vel_pert, pres_pert = scatter_to_nodes(δu_full, d.cell_dofs, d.cell_connectivity,
	n_nodes, d.n_vel_dofs_per_cell)
	local vmag_k              = sqrt.(vec(sum(vel_k .^ 2; dims = 1)))
	local vort_k              = compute_vorticity(vel_k, d.node_coords, d.cell_connectivity, n_nodes)
	local vort_pert           = compute_vorticity(vel_pert, d.node_coords, d.cell_connectivity, n_nodes)

	local fname = joinpath(OUT_DIR, @sprintf("vortex_%03d", k))
	vtk_grid(fname, points, cells) do vtk
		vtk_point_data(vtk, vel_k, "velocity")
		vtk_point_data(vtk, pres_k, "pressure")
		vtk_point_data(vtk, vmag_k, "velocity_magnitude")
		vtk_point_data(vtk, vort_k, "vorticity")
		vtk_point_data(vtk, vel0, "velocity_base")
		vtk_point_data(vtk, pres0, "pressure_base")
		vtk_point_data(vtk, vort0, "vorticity_base")
		vtk_point_data(vtk, vel_pert, "velocity_perturbation")
		vtk_point_data(vtk, pres_pert, "pressure_perturbation")
		vtk_point_data(vtk, vort_pert, "vorticity_perturbation")
		pvd[Float64(k)] = vtk
	end
	@printf("  frame %3d / %d  (θ = %5.2f°)\n", k, N_FRAMES, rad2deg(θ))
end

vtk_save(pvd)
println("  → vortex.pvd")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

println("\nParaview files written to: $OUT_DIR")
if haskey(d, :all_modes)
	println("  base_flow.vtu")
	println("  eigenmode_001 … eigenmode_$(lpad(n_modes,3,'0')).vtu")
	println("  mode_001.pvd … mode_$(lpad(n_modes,3,'0')).pvd  (linear orbit, $N_FRAMES frames each)")
	println("  vortex.pvd  (nonlinear W, $N_FRAMES frames)")
else
	println("  base_flow.vtu, eigenmode_001.vtu, vortex.pvd")
	println("  Note: re-run main.jl to get all-modes data in vtk_data.jls")
end
println("\nParaview tips:")
println("  • base_flow.vtu      → colour by 'vorticity' for the Kármán wake structure")
println("  • eigenmode_NNN.vtu  → colour by 'vorticity_Re' or 'vorticity_Im' to see mode shape")
println("  • mode_NNN.pvd / vortex.pvd → animate; switch between:")
println("      'vorticity'              — total field (base + mode)")
println("      'vorticity_base'         — steady base vorticity (constant reference)")
println("      'vorticity_perturbation' — mode contribution only (clearest for comparing modes)")
println("  • Add 'Warp by Vector' on 'velocity_perturbation' to exaggerate the mode displacement")
