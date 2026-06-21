"""
	backbone.jl

Backbone curves for the parametric arch ROM and all non-parametric references.

For each arch height ratio in `H_RATIOS`:
  1. Load the reference ROM (W_ref, R_ref) from results/reference/arch_h_*/data/.
  2. Compute the conservative backbone Ω(r) from R_ref.
  3. Map modal amplitude r to physical transverse displacement at the beam midpoint
	 via W_ref.
  4. Repeat for the parametric ROM (W_param, R_param from main.jl) evaluated at
	 the θ value corresponding to the same arch height, for four truncation orders
	 of the θ expansion: max, max-1, max-2, max-3.

Outputs (in results/backbone/):
  backbones.csv   — long-format (h_ratio, theta, model, theta_order, r, omega, amplitude)
                    theta_order = -1 for reference rows
  metrics.csv     — ω₀ per (h_ratio, model, theta_order)
"""

# -----------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------
using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using MORFE
using MORFE.Polynomials: DensePolynomial, evaluate, extract_component, each_term, similar_poly
using MORFE.Realification: realify
using Ferrite, FerriteGmsh
using Serialization, Printf, StaticArrays
using SparseArrays, LinearAlgebra

# -----------------------------------------------------------------------
# Constants  (match main.jl and compute_references.jl)
# -----------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "config.jl"))   # h0_L_ratio, N_INCREMENTS

const L = 1000.0
const h₀_L_ratio = h0_L_ratio
const H_RATIOS = collect(range(0.0, 2*h0_L_ratio; length = N_INCREMENTS + 1))
const N_R = 200    # backbone points per curve
const N_PHASE = 24     # orbit phases for max-over-period amplitude
const R_MAX = 60.0   # modal amplitude sweep upper bound
const A_MAX = 15.0   # physical amplitude cap (mm) for CSV truncation

const _msh = joinpath(@__DIR__, "..", "..", "..", "benchmark", "ferrite",
	"beam_h27_10x2x2.msh")
const _param_dir = joinpath(@__DIR__, "..", "results", "data",
	@sprintf("arch_h%.3f", h0_L_ratio))
const _ref_root = joinpath(@__DIR__, "..", "results", "reference")
const _out_dir = joinpath(@__DIR__, "..", "results", "backbone")

isfile(_msh) || error("Mesh not found at $_msh.  Run generate_beam_mesh.jl first.")
isfile(joinpath(_param_dir, "W.jls")) ||
	error("Parametric ROM not found at $_param_dir.  Run main.jl first.")

# -----------------------------------------------------------------------
# DOF setup — flat mesh is used only to identify i_mid
# (topology is identical for all arched meshes; shifts don't change DOF order)
# -----------------------------------------------------------------------

println("Building DOF map from flat mesh …")
_grid = togrid(_msh)
_ip = Lagrange{RefHexahedron, 2}()^3
_dh = DofHandler(_grid)
add!(_dh, :u, _ip)
close!(_dh)
_ch = ConstraintHandler(_dh)
add!(_ch, Dirichlet(:u, getfacetset(_grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(_ch)
update!(_ch, 0.0)

_free = sort(setdiff(1:ndofs(_dh), _ch.prescribed_dofs))
_free_to_local = Dict(d => i for (i, d) in enumerate(_free))

dof_to_nodecomp = Dict{Int, Tuple{Int, Int}}()
for cell in CellIterator(_dh)
	dofs = celldofs(cell)
	nodes = _grid.cells[Ferrite.cellid(cell)].nodes
	for l in eachindex(nodes), c in 1:3
		gdof = dofs[3(l-1)+c]
		haskey(_free_to_local, gdof) &&
			(dof_to_nodecomp[_free_to_local[gdof]] = (nodes[l], c))
	end
end

# Midpoint transverse DOF: y-component (comp=2) at node with x₁ closest to L/2
i_mid = let
	best_d = Inf
	best_i = 0
	for (i, (node, comp)) in dof_to_nodecomp
		comp == 2 || continue
		d = abs(_grid.nodes[node].x[1] - L / 2)
		d < best_d && (best_d = d; best_i = i)
	end
	best_i
end
let (node, comp) = dof_to_nodecomp[i_mid]
	@printf "  i_mid = %d  node = %d  comp = %d  x₀ = %s\n" i_mid node comp string(_grid.nodes[node].x)
end

# -----------------------------------------------------------------------
# Flat-beam mass matrix  (free-free; used for modal cross-projection)
# Volume-preserving arch → M_arch = M₀ for all θ, so one assembly suffices.
# -----------------------------------------------------------------------

println("Assembling flat-beam mass matrix …")
const _ρ = 2.32e-3   # g/mm³ — must match arch_assembly.jl

_M₀ = let
	ip_m     = Lagrange{RefHexahedron, 2}()^3
	geo_ip_m = Lagrange{RefHexahedron, 2}()   # quadratic geometry — mesh has 27-node hexahedra
	qr_m     = QuadratureRule{RefHexahedron}(3)
	cv_m     = CellValues(qr_m, ip_m, geo_ip_m)
	n    = length(_free)
	rows = Int[];
	cols = Int[];
	vals = Float64[]
	nd_c = ndofs_per_cell(_dh)
	Mₑ = zeros(nd_c, nd_c)
	for cell in CellIterator(_dh)
		reinit!(cv_m, cell)
		dofs = celldofs(cell)
		fill!(Mₑ, 0.0)
		for q in 1:getnquadpoints(cv_m)
			dΩ = getdetJdV(cv_m, q)
			for i in 1:nd_c
				Ni = shape_value(cv_m, q, i)
				for j in 1:nd_c
					Mₑ[i, j] += _ρ * (Ni ⋅ shape_value(cv_m, q, j)) * dΩ
				end
			end
		end
		for (li, di) in enumerate(dofs)
			fi = get(_free_to_local, di, 0);
			fi == 0 && continue
			for (lj, dj) in enumerate(dofs)
				fj = get(_free_to_local, dj, 0);
				fj == 0 && continue
				push!(rows, fi);
				push!(cols, fj);
				push!(vals, Mₑ[li, lj])
			end
		end
	end
	sparse(rows, cols, vals, n, n)
end
@printf "  M₀ assembled: %d × %d  (nnz = %d)\n" size(_M₀, 1) size(_M₀, 2) nnz(_M₀)

# -----------------------------------------------------------------------
# Polynomial helpers
# -----------------------------------------------------------------------

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

displacement_poly(W) = DensePolynomial(Matrix(W.poly.coefficients[:, 1, :]),
	W.poly.multiindex_set)

"Peak transverse displacement at i_mid over one oscillation period."
function amplitude_at_mid(Wdisp::DensePolynomial, i_mid::Int,
	r::Float64, θ_extra, u_static_mid::Float64)
	a = 0.0
	for φ in range(0.0, 2π; length = N_PHASE + 1)[1:N_PHASE]
		z = r * cis(φ)
		vals = ComplexF64[z, conj(z), θ_extra...]
		u_mid = real(evaluate(Wdisp, vals, i_mid))
		a = max(a, abs(u_mid - u_static_mid))
	end
	return a
end

"""
	mode_shape_at_θ(Wdisp, θ) → ComplexF64 vector (length = FOM)

Sum all monomials of the form z₁¹ z₂⁰ θᵏ in the displacement polynomial.
For NVAR=2 (reference, no θ) only the (1,0) monomial contributes (θ^0 = 1).
For NVAR=3 (parametric) sums contributions from all (1,0,k) monomials.
"""
function mode_shape_at_θ(Wdisp::DensePolynomial, θ::Float64)
	FOM = size(Wdisp.coefficients, 1)
	φ = zeros(ComplexF64, FOM)
	for (l, α) in enumerate(collect(Wdisp.multiindex_set))
		α[1] == 1 && α[2] == 0 || continue
		θ_pow = length(α) ≥ 3 ? α[3] : 0
		@views φ .+= Wdisp.coefficients[:, l] .* (θ^θ_pow)
	end
	return φ
end

"""
	truncate_theta_order(poly, k_max) → DensePolynomial

Copy of `poly` with all monomials where the θ-variable (index 3) power > k_max zeroed out.
"""
function truncate_theta_order(poly::DensePolynomial, k_max::Int)
	coeffs = copy(poly.coefficients)
	for (l, α) in enumerate(collect(poly.multiindex_set))
		if α[3] > k_max
			selectdim(coeffs, ndims(coeffs), l) .= 0
		end
	end
	return DensePolynomial(coeffs, poly.multiindex_set)
end

# -----------------------------------------------------------------------
# Load parametric ROM
# -----------------------------------------------------------------------

println("Loading parametric ROM from $_param_dir …")
W_param = deserialize(joinpath(_param_dir, "W.jls"))
R_param = deserialize(joinpath(_param_dir, "R.jls"))

R1_param    = extract_component(realify(R_param.poly, [2, 1, 3]), 1)
Wdisp_param = displacement_poly(W_param)

# Determine θ truncation orders: max, max-1, max-2, max-3
θ_max_order  = maximum(α[3] for α in collect(R1_param.multiindex_set))
theta_orders = max(0, θ_max_order - 3):θ_max_order
@printf "  θ_max_order = %d   truncation orders: %s\n" θ_max_order string(collect(theta_orders))

# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------

mkpath(_out_dir)
r_range = range(1e-6, R_MAX, N_R)

# Flat list: one entry per (h_ratio, model, theta_order)
curves = NamedTuple[]

println()
for h_ratio in H_RATIOS
	θ_fixed = h_ratio / h₀_L_ratio - 1.0
	tag = @sprintf("arch_h_%.3f", h_ratio)
	ref_data = joinpath(_ref_root, tag, "data")

	println("── h₀/L = $h_ratio  (θ = $θ_fixed) ──")

	if !isfile(joinpath(ref_data, "W.jls"))
		@printf "  [skip] reference not found at %s\n" ref_data
		continue
	end

	W_ref = deserialize(joinpath(ref_data, "W.jls"))
	R_ref = deserialize(joinpath(ref_data, "R.jls"))

	# Reference backbone
	R1_ref    = extract_component(realify(R_ref.poly, [2, 1]), 1)
	dR1dx_ref = poly_deriv(R1_ref, 1)
	ω₀_ref    = imag(evaluate(dR1dx_ref, [0.0, 0.0]))
	Ω_ref(r)  = imag(evaluate(R1_ref, [max(r, 1e-12), 0.0])) / max(r, 1e-12)

	Wdisp_ref = displacement_poly(W_ref)
	a_ref     = [amplitude_at_mid(Wdisp_ref, i_mid, r, (), 0.0) for r in r_range]
	Ω_ref_vec = [Ω_ref(r) for r in r_range]

	@printf "  ω₀ ref = %.6f\n" ω₀_ref

	push!(curves, (;
		h_ratio, θ_fixed,
		model      = "reference",
		theta_order = -1,
		r           = collect(r_range),
		Ω           = Ω_ref_vec,
		a           = a_ref,
		ω₀          = ω₀_ref,
		Δω₀_rel     = NaN,
		modal_proj  = NaN,
	))

	# Modal cross-projection: φ_ref^H M₀ φ_param(θ)
	# ≈ ±1 → same physical mode; ≈ 0 → orthogonal (different mode).
	φ_ref      = mode_shape_at_θ(Wdisp_ref,   0.0)
	φ_param_θ  = mode_shape_at_θ(Wdisp_param, θ_fixed)
	modal_proj = real(dot(φ_ref, _M₀ * φ_param_θ))
	@printf "  φ_ref^H M₀ φ_param(θ) = %+.4f  (|1|→same mode, 0→different)\n" modal_proj

	# Parametric backbone at θ_fixed — one curve per truncation order
	for theta_order in theta_orders
		R1_trunc    = truncate_theta_order(R1_param, theta_order)
		dR1dx_trunc = poly_deriv(R1_trunc, 1)
		Wdisp_trunc = truncate_theta_order(Wdisp_param, theta_order)

		ω₀_p       = imag(evaluate(dR1dx_trunc, [0.0, 0.0, θ_fixed]))
		Ω_p        = [imag(evaluate(R1_trunc, [max(r, 1e-12), 0.0, θ_fixed])) /
		               max(r, 1e-12) for r in r_range]
		u_static   = real(evaluate(Wdisp_trunc, ComplexF64[0.0, 0.0, θ_fixed], i_mid))
		a_p        = [amplitude_at_mid(Wdisp_trunc, i_mid, r, (θ_fixed,), u_static)
		              for r in r_range]

		Δω₀_rel = abs(ω₀_ref - ω₀_p) / abs(ω₀_ref)
		@printf "  [θ-order %d]  ω₀ param = %.6f   Δω₀_rel = %.3e\n" theta_order ω₀_p Δω₀_rel

		push!(curves, (;
			h_ratio, θ_fixed,
			model       = "parametric",
			theta_order,
			r           = collect(r_range),
			Ω           = Ω_p,
			a           = a_p,
			ω₀          = ω₀_p,
			Δω₀_rel,
			modal_proj,
		))
	end
end

# -----------------------------------------------------------------------
# backbones.csv  — long format
# -----------------------------------------------------------------------

open(joinpath(_out_dir, "backbones.csv"), "w") do io
	println(io, "h_ratio,theta,model,theta_order,r,omega,amplitude")
	for c in curves
		for k in eachindex(c.r)
			c.a[k] ≤ A_MAX &&
				@printf io "%.3f,%.4f,%s,%d,%.8f,%.8f,%.8f\n" c.h_ratio c.θ_fixed c.model c.theta_order c.r[k] c.Ω[k] c.a[k]
		end
	end
end
println("Saved → $(joinpath(_out_dir, "backbones.csv"))")

# -----------------------------------------------------------------------
# metrics.csv
# -----------------------------------------------------------------------

open(joinpath(_out_dir, "metrics.csv"), "w") do io
	println(io, "h_ratio,theta,model,theta_order,omega0,delta_omega0_rel,modal_proj")
	for c in curves
		@printf io "%.3f,%.4f,%s,%d,%.8f,%.4e,%.4f\n" c.h_ratio c.θ_fixed c.model c.theta_order c.ω₀ c.Δω₀_rel c.modal_proj
	end
end
println("Saved → $(joinpath(_out_dir, "metrics.csv"))")

println("\nDone. Backbone data in $_out_dir/")
