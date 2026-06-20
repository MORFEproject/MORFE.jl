"""
	eigenfrequency_error_heatmap.jl

Heat map (with isolines) of the relative error in the linear eigenfrequency
of the parametric ROM over a (θ₁, θ₂) grid.

For each grid point (θ₁, θ₂):
  1. Deform the reference mesh to the exact geometry x(θ₁, θ₂).
  2. Assemble exact K, M on the deformed mesh and solve the generalised
	 eigenproblem (Arpack, shift-invert at σ = 0).
  3. Select the first-bending eigenpair by MAC against the reference
	 master mode (robust to mode crossings at large θ₂).
  4. Convert the undamped FEM frequency to the damped one (β = 0 ⇒
	 ω_d = √(ω² − (α/2)²)) and compare with ω₀_ROM(θ₁,θ₂) = Im[∂R₁/∂x](0,0,θ₁,θ₂).

Axes:
  x — θ₁  (uniform axial stretch),       θ₁ ∈ [-0.167, 0.25]
  y — midpoint vertical displacement induced by θ₂:
	  w_mid(θ₂) = θ₂ · |φ̂₁,⊥(x_mid)|,    θ₂ ∈ [0, 6]
	  (φ̂₁ is the max-normalised arch mode; its dominant transverse
	   component at the beam midpoint is ≈ 1 by construction)

Outputs:
  results/figures/eigenfrequency_error_heatmap.png
  results/data/eigenfrequency_error_grid.csv

Run after main.jl has produced results/data/R.jls.
"""

using Pkg: Pkg
const _val_env = joinpath(@__DIR__, "validation_env")
Pkg.activate(_val_env)
if !haskey(Pkg.project().dependencies, "MORFE")
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "..")))
	Pkg.add(["Ferrite", "FerriteGmsh", "LinearMaps", "Tensors", "StaticArrays", "Plots"])
	Pkg.add(Pkg.PackageSpec(name = "Arpack", version = "0.5.3"))
end
Pkg.instantiate()

using MORFE
using Ferrite, FerriteGmsh
using SparseArrays, LinearAlgebra
using Arpack, LinearMaps
using Tensors, StaticArrays
using MORFE.Polynomials: DensePolynomial, evaluate, extract_component, each_term, similar_poly
using MORFE.Realification: realify
using Serialization
using Printf
using Plots
ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "exact_geometry_assembly.jl"))

# ------------------------------------------------------------------
# Constants  (match main.jl)
# ------------------------------------------------------------------
const _msh = joinpath(@__DIR__, "..", "..", "..", "benchmark", "ferrite", "beam_h27_10x2x2.msh")
const _data = joinpath(@__DIR__, "..", "results", "data")
const _figs = joinpath(@__DIR__, "..", "results", "figures")
const E_val = 160e3
const ν_val = 0.22
const ρ_val = 2.32e-3
const λ_lame = (E_val * ν_val) / ((1 + ν_val) * (1 - 2ν_val))
const μ_lame = E_val / (2(1 + ν_val))
const α_damp = 0.5369754008568333 / 500.0
const β_damp = 0.0
const J₁_def = Tensor{2, 3, Float64}((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0)
const CONJ_PARAM = [2, 1, 3, 4]

# ------------------------------------------------------------------
# Grid
# ------------------------------------------------------------------
const N_θ₁ = 100
const N_θ₂ = 100
const θ₁_grid = range(-0.167, 0.25, N_θ₁)
const θ₂_grid = range(0.0, 6.0, N_θ₂)

# ------------------------------------------------------------------
# Reference FE setup
# ------------------------------------------------------------------
isfile(_msh) || error("Mesh not found at $_msh.  Run generate_beam_mesh.jl first.")
grid_ref = togrid(_msh)
ip       = Lagrange{RefHexahedron, 2}()^3
geo_ip   = Lagrange{RefHexahedron, 2}()
qr       = QuadratureRule{RefHexahedron}(3)

dh_ref = DofHandler(grid_ref)
add!(dh_ref, :u, ip)
close!(dh_ref)
ch_ref = ConstraintHandler(dh_ref)
add!(ch_ref, Dirichlet(:u, getfacetset(grid_ref, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch_ref)
update!(ch_ref, 0.0)
free_ref          = sort(setdiff(1:ndofs(dh_ref), ch_ref.prescribed_dofs))
free_to_local_ref = Dict(d => i for (i, d) in enumerate(free_ref))
n_free_ref        = length(free_ref)

# ------------------------------------------------------------------
# Parametric ROM:  ω₀_param(θ₁, θ₂)
# ------------------------------------------------------------------
isfile(joinpath(_data, "R.jls")) ||
	error("results/data/R.jls not found.  Run main.jl first.")
R_param = deserialize(joinpath(_data, "R.jls"))

function poly_deriv(p::DensePolynomial{T, NVAR, 1}, var_idx::Int) where {T, NVAR}
	dict = Dict{SVector{NVAR, Int}, T}()
	for (α, c) in each_term(p)
		n = α[var_idx]
		iszero(n) && continue
		new_α = SVector{NVAR, Int}(ntuple(j -> j == var_idx ? α[j] - 1 : α[j], Val(NVAR)))
		dict[new_α] = get(dict, new_α, zero(T)) + T(n) * c
	end
	return similar_poly(dict)
end

R1_param = extract_component(realify(R_param.poly, CONJ_PARAM), 1)
dR1dx_param = poly_deriv(R1_param, 1)
ω₀_param(θ₁, θ₂) = imag(evaluate(dR1dx_param, [0.0, 0.0, θ₁, θ₂]))

# ------------------------------------------------------------------
# Arch mode  (must match the one used by main.jl)
# ------------------------------------------------------------------
arch_mode_free = if isfile(joinpath(_data, "arch_mode.jls"))
	deserialize(joinpath(_data, "arch_mode.jls"))
else
	println("arch_mode.jls not found — recomputing from reference eigenproblem …")
	_cv0 = CellValues(qr, ip, geo_ip)
	_K0f = allocate_matrix(dh_ref)
	_M0f = allocate_matrix(dh_ref)
	assemble_KM!(_K0f, _M0f, dh_ref, _cv0, λ_lame, μ_lame, ρ_val)
	_K0 = _K0f[free_ref, free_ref]
	_M0 = _M0f[free_ref, free_ref]
	_slvr = StructureModalDampingEigensolver(4, α_damp, β_damp)
	_ep = solve_eigenproblem(_K0, _M0, _slvr; sorter! = (args...) -> nothing)
	(_, _Y, _) = get_eigenpairs(_ep)
	_arch = real(_Y[:, 1, 1])
	_arch ./= maximum(abs, _arch)
	_arch
end

# Reference master-mode shape for MAC-based eigenpair selection.
const φ_ref = arch_mode_free / norm(arch_mode_free)

# ------------------------------------------------------------------
# Midpoint node and the y-axis scaling  w_mid(θ₂) = θ₂ · v_mid
# ------------------------------------------------------------------
# node → free-DOF triple, via the cell-wise layout (3 dofs per node, Q2 hex)
node_dofs = Dict{Int, NTuple{3, Int}}()
for cell in CellIterator(dh_ref)
	dofs  = celldofs(cell)
	nodes = grid_ref.cells[Ferrite.cellid(cell)].nodes
	for l in eachindex(nodes)
		node_dofs[nodes[l]] = (dofs[3l-2], dofs[3l-1], dofs[3l])
	end
end

coords = [n.x for n in grid_ref.nodes]
lo = Vec{3}((minimum(c[1] for c in coords), minimum(c[2] for c in coords),
	minimum(c[3] for c in coords)))
hi = Vec{3}((maximum(c[1] for c in coords), maximum(c[2] for c in coords),
	maximum(c[3] for c in coords)))
centre = 0.5 * (lo + hi)
mid_node = argmin([norm(c - centre) for c in coords])

# Arch-mode displacement vector at the midpoint node (Dirichlet dofs → 0)
arch_at(dof) = haskey(free_to_local_ref, dof) ? arch_mode_free[free_to_local_ref[dof]] : 0.0
u_mid = collect(arch_at.(node_dofs[mid_node]))
v_comp = argmax(abs.(u_mid))           # dominant (transverse/bending) component
v_mid = abs(u_mid[v_comp])
@printf "Midpoint node %d at %s — arch-mode displacement %s\n" mid_node string(coords[mid_node]) string(round.(u_mid, digits = 4))
@printf "Dominant transverse component: %d, |φ̂₁,⊥(x_mid)| = %.6f\n" v_comp v_mid
w_mid_grid = collect(θ₂_grid) .* v_mid   # y-axis values

# ------------------------------------------------------------------
# Exact eigenfrequency on the deformed mesh
# ------------------------------------------------------------------
"""
	ω_exact(θ₁, θ₂) -> (ω_damped, mac)

First-bending undamped eigenfrequency on the exactly deformed mesh,
converted to the damped frequency (mass-proportional damping, β = 0).
The eigenpair is selected by MAC against the reference master mode.
"""
function ω_exact(θ₁, θ₂; nev = 6)
	grid_def = deform_grid(grid_ref, dh_ref, θ₁, θ₂,
		J₁_def, arch_mode_free, free_to_local_ref)
	ex = assemble_exact_KM(grid_def, ip, geo_ip, qr,
		λ_lame, μ_lame, ρ_val, α_damp, β_damp)
	# deform_grid preserves topology and dof layout ⇒ free-DOF numbering matches
	vals, vecs = eigs(ex.K, ex.M; nev = nev, which = :LM, sigma = 0.0, check = 1)
	macs = [abs2(dot(φ_ref, @view vecs[:, j])) /
			(abs2(norm(@view vecs[:, j]))) for j in 1:nev]
	j★ = argmax(macs)
	ω² = real(vals[j★])
	ω = sqrt(abs(ω²))
	ω_d = sqrt(max(ω^2 - (α_damp / 2)^2, 0.0))   # damped frequency (β = 0)
	return ω_d, macs[j★]
end

# ------------------------------------------------------------------
# Sweep the grid
# ------------------------------------------------------------------
rel_err = zeros(N_θ₂, N_θ₁)     # rows ↔ θ₂ (y), cols ↔ θ₁ (x)
ω_ex_g = zeros(N_θ₂, N_θ₁)
ω_p_g = zeros(N_θ₂, N_θ₁)

mkpath(_data)
mkpath(_figs)

t0 = time()
for (i, θ₂) in enumerate(θ₂_grid), (j, θ₁) in enumerate(θ₁_grid)
	ω_ex, mac = ω_exact(θ₁, θ₂)
	ω_p = ω₀_param(θ₁, θ₂)
	rel_err[i, j] = abs(ω_ex - ω_p) / abs(ω_ex)
	ω_ex_g[i, j] = ω_ex
	ω_p_g[i, j] = ω_p
	done = (i - 1) * N_θ₁ + j
	@printf "[%3d/%3d] θ₁=%+.4f θ₂=%.3f  ω_ex=%.6f ω_p=%.6f  relerr=%.3e  MAC=%.3f  (%.0fs)\n" done (N_θ₁ * N_θ₂) θ₁ θ₂ ω_ex ω_p rel_err[i, j] mac (time() - t0)
end

# ------------------------------------------------------------------
# CSV
# ------------------------------------------------------------------
open(joinpath(_data, "eigenfrequency_error_grid.csv"), "w") do io
	println(io, "theta1,theta2,w_mid,omega0_exact,omega0_param,rel_err")
	for (i, θ₂) in enumerate(θ₂_grid), (j, θ₁) in enumerate(θ₁_grid)
		@printf io "%.6f,%.6f,%.6f,%.8f,%.8f,%.6e\n" θ₁ θ₂ (θ₂ * v_mid) ω_ex_g[i, j] ω_p_g[i, j] rel_err[i, j]
	end
end
println("Saved → $(joinpath(_data, "eigenfrequency_error_grid.csv"))")

# ------------------------------------------------------------------
# Heat map with isolines  (log₁₀ relative error)
# ------------------------------------------------------------------
log_err = log10.(max.(rel_err, 1e-16))

plt = contourf(collect(θ₁_grid), w_mid_grid, log_err;
	color = :viridis, levels = 20, lw = 0,
	xlabel = "Axial stretch  θ₁",
	ylabel = "Midpoint vertical displacement  θ₂·|φ̂₁,⊥(x_mid)|",
	title = "Relative error of linear eigenfrequency  log₁₀|ω₀ᴿᴼᴹ − ω₀ᶠᴱᴹ|/ω₀ᶠᴱᴹ",
	colorbar_title = "log₁₀ rel. error",
	size = (820, 620), dpi = 150)
iso_levels = collect(ceil(Int, minimum(log_err)):1:floor(Int, maximum(log_err)))
contour!(plt, collect(θ₁_grid), w_mid_grid, log_err;
	levels = iso_levels, color = :black, lw = 1.0,
	contour_labels = true, colorbar_entry = false)
savefig(plt, joinpath(_figs, "eigenfrequency_error_heatmap.png"))
println("Saved → $(joinpath(_figs, "eigenfrequency_error_heatmap.png"))")
