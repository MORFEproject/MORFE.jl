"""
	energy_gram.jl — kinetic-energy Gram matrix for the TKE observable.

The perturbation field is  u'(z) = Σ_m W_m · ζ_m(z),  ζ_m = z^a · z̄^b · η'^c, and the
reference-paper period-averaged TKE is the continuous L² integral

	⟨TKE⟩ = (1/|Ω|)(1/T) ∫_Ω ∫_0^T ½ ‖u'(x,t)−ū(x)‖² dt dx.

In FEM, ∫_Ω ‖u'‖² dx = u'^T M_vel u' (velocity DOFs only; pressure carries no kinetic
energy), which gives the Gram representation ½ ζᵀ G ζ with

	G = (W_velᵀ M_vel W_vel) / |Ω|        (L × L, complex, symmetric).

G plus the monomial exponents are all Python needs to evaluate ⟨TKE⟩ from a reduced
orbit (see validation/average_tke.py::tke_from_gram) — no FOM-sized data shipped.
"""

using Ferrite
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using MORFE

"""
	velocity_dof_mask(fom) -> BitVector

Mark every global DOF that belongs to the velocity field :u.
"""
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

"""
	assemble_velocity_mass_full(fom) -> SparseMatrixCSC

Assemble M_vel = ∫_Ω φᵢ·φⱼ dΩ for velocity shape functions only (full DOF space;
pressure rows/columns stay zero).
"""
function assemble_velocity_mass_full(fom)
	n_dpc = ndofs_per_cell(fom.dh)
	n_vel = length(fom.dof_range_u)
	M = Ferrite.allocate_matrix(fom.dh)
	fill!(M, 0.0)
	Me = zeros(n_dpc, n_dpc)
	asm = start_assemble(M)
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

"""
	domain_area(fom) -> Float64

Quadrature measure of the fluid domain |Ω|.
"""
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

"""
	prepare_energy_gram(fom) -> (M_vel, vel_rows, area)

Order-independent part: velocity mass restricted to the velocity rows of the
free_dpim DOF set, the row indices of velocity DOFs within the free vector, and |Ω|.
Call ONCE per FEM setup.
"""
function prepare_energy_gram(fom)
	is_vel = velocity_dof_mask(fom)
	vel_rows = findall(d -> is_vel[d], fom.free_dpim)
	M_full = assemble_velocity_mass_full(fom)
	vel_gdofs = fom.free_dpim[vel_rows]
	M_vel = M_full[vel_gdofs, vel_gdofs]
	area = domain_area(fom)
	return M_vel, vel_rows, area
end

"""
	write_energy_gram(data_dir, W, M_vel, vel_rows, area)

Per-order part: G = (W_velᵀ M_vel W_vel)/|Ω| and the exponent table, written as
tke_gram_re.csv / tke_gram_im.csv / tke_avector.csv into `data_dir`.
"""
function write_energy_gram(data_dir::AbstractString, W, M_vel, vel_rows, area)
	C = MORFE.ParametrisationMethod.coefficients(W)   # (FOM, 1, L)
	# Materialise the velocity block: a fancy-indexed view is not strided, so
	# both products below would fall back to generic element-wise matmul
	# (minutes); a dense Matrix routes them through the sparse kernel and BLAS
	# zgemm (seconds) at ~n_vel × L × 16 B of extra memory.
	W1_vel = C[vel_rows, 1, :]                        # n_vel × L dense
	G = (transpose(W1_vel) * (M_vel * W1_vel)) ./ area
	exps = MORFE.ParametrisationMethod.multiindex_set(W).exponents
	Avec = permutedims(reduce(hcat, [collect(Int, e) for e in exps]))   # L × 3
	writedlm(joinpath(data_dir, "tke_gram_re.csv"), real.(G), ',')
	writedlm(joinpath(data_dir, "tke_gram_im.csv"), imag.(G), ',')
	writedlm(joinpath(data_dir, "tke_avector.csv"), Avec, ',')
	return nothing
end
