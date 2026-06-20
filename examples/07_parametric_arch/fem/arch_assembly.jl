"""
	arch_assembly.jl

Univariate-in-θ FEM assembly for the sinusoidal arch beam.

Unlike the bivariate assembly in example 04, here J₀(x₀) and J₁(x₀) are both
spatially varying (they depend on the axial coordinate x₁ through the arch
gradient), so the det/adj series must be computed **per quadrature point**.

The weak form integrands follow exactly the same cofactor identity as in
examples 01/04:

	K integrand = ε_adj(v) ⊡ σ(ε_adj(u)) · (1/det J)¹ · dV₀
	M integrand = ρ (u · v) · det J                    · dV₀
	quad = [bracket] · (1/det J)²                · dV₀
	cubic = [bracket] · (1/det J)³                · dV₀

where ε_adj(u) = sym(∇₀u · adj(J(θ, x₀))) and the per-QP series

	det J(θ, x₀),  adj J(θ, x₀),  1/det J(θ, x₀)

are computed by calling `det_and_adj_series(J₀_qp, J₁_qp)` and
`reciprocal_series(det_ser, N_θ)` from `parametric_geometry.jl` at every QP.

This file is self-contained: it defines all helper functions it needs
(gather/scatter, σ_lame, ∇adj_series, E_nl_adj_series, polynomial bracket
series, closure factories, builder functions) and does not depend on
`parametric_assembly.jl` from example 04.

SIGN CONVENTION — all internal-force MultilinearMaps are negated: MORFE writes
nonlinear terms on the right-hand side of  M ẍ + C ẋ + K x = (maps).

N_EXT = 1 throughout: a single external state θ.  Each external closure slot
receives an SVector{1, Int}; its scalar component is accessed as `r[1]`.

Requires: theta_polynomials.jl, parametric_geometry.jl, arch_geometry.jl,
		  Ferrite, Tensors, SparseArrays, MORFE.
"""

using Ferrite
using Tensors
using LinearAlgebra
using SparseArrays
using MORFE

# `Tens3` is defined in parametric_geometry.jl (must be loaded first).

# ===========================================================================
# 0.  Shared QP helpers
# ===========================================================================

@inline function gather_local!(ue::Vector{T}, u::AbstractVector{T},
	dofs::Vector{Int}, free_to_local::Dict{Int, Int}) where {T}
	@inbounds for (i, d) in pairs(dofs)
		ue[i] = haskey(free_to_local, d) ? u[free_to_local[d]] : zero(T)
	end
	return ue
end

@inline function scatter_local!(res::AbstractVector{T}, re::Vector{T},
	dofs::Vector{Int}, free_to_local::Dict{Int, Int}) where {T}
	@inbounds for (i, d) in pairs(dofs)
		haskey(free_to_local, d) || continue
		res[free_to_local[d]] += re[i]
	end
	return res
end

@inline σ_lame(ε, λ::Float64, μ::Float64) = λ * tr(ε) * one(ε) + 2μ * ε

@inline function ∇adj_series(∇u, adjJ_coeffs::Vector)
	return [∇u ⋅ adjJ_coeffs[a+1] for a in 0:(length(adjJ_coeffs)-1)]
end

function E_nl_adj_series(∇uA_adj_ser::Vector, ∇uB_adj_ser::Vector, N_θ::Int)
	AB = poly_dot([transpose(g) for g in ∇uA_adj_ser], ∇uB_adj_ser, N_θ)
	BA = poly_dot([transpose(g) for g in ∇uB_adj_ser], ∇uA_adj_ser, N_θ)
	return [symmetric(0.25 * (AB[k+1] + BA[k+1])) for k in 0:N_θ]
end

# ===========================================================================
# 1.  Linear K_k, M_k assembly with per-QP J₀, J₁
# ===========================================================================

"""
	assemble_K_M_arch!(K_arr, M_arr, dh, cv, λ, μ, ρ, h₀, L, free_to_local, N_θ)

Fill the θ-power coefficient matrices:

	K_arr[k+1] = K_k     (θ^k coefficient of linear stiffness)
	M_arr[k+1] = M_k     (θ^k coefficient of linear mass)

At every quadrature point, the arch Jacobian pair (J₀, J₁) is computed
analytically from `arch_jacobian_pair(x₀_qp, h₀, L)` (defined in
arch_geometry.jl), then the univariate det/adj/inv_det series are derived.

K_arr and M_arr must each have length N_θ + 1 and hold sparse matrices with
identical sparsity pattern (use `allocate_matrix(dh)` for each entry).
"""
function assemble_K_M_arch!(K_arr::Vector, M_arr::Vector,
	dh::DofHandler, cv::CellValues,
	λ::Float64, μ::Float64, ρ::Float64,
	h₀::Float64, L::Float64,
	free_to_local::Dict{Int, Int},
	N_θ::Int)
	@assert length(K_arr) == N_θ + 1
	@assert length(M_arr) == N_θ + 1

	assemblers_K = [start_assemble(K) for K in K_arr]
	assemblers_M = [start_assemble(M) for M in M_arr]

	n_basefuncs = getnbasefunctions(cv)
	ke = [zeros(n_basefuncs, n_basefuncs) for _ in 0:N_θ]
	me = [zeros(n_basefuncs, n_basefuncs) for _ in 0:N_θ]

	for cell in CellIterator(dh)
		for k in eachindex(ke)
			;
			fill!(ke[k], 0.0);
		end
		for k in eachindex(me)
			;
			fill!(me[k], 0.0);
		end
		reinit!(cv, cell)
		coords = getcoordinates(cell)

		for q_point in 1:getnquadpoints(cv)
			dΩ₀ = getdetJdV(cv, q_point)
			x₀_qp = spatial_coordinate(cv, q_point, coords)

			J₀_qp, J₁_qp = arch_jacobian_pair(x₀_qp, h₀, L)
			det_ser, adj_ser = det_and_adj_series(J₀_qp, J₁_qp)
			inv_det_ser = reciprocal_series(det_ser, N_θ)

			for i in 1:n_basefuncs
				∇Ni = shape_gradient(cv, q_point, i)
				Ni = shape_value(cv, q_point, i)
				ε_adj_i = [symmetric(∇Ni ⋅ adj_ser[a+1]) for a in 0:(length(adj_ser)-1)]

				for j in 1:n_basefuncs
					∇Nj = shape_gradient(cv, q_point, j)
					Nj = shape_value(cv, q_point, j)
					ε_adj_j = [symmetric(∇Nj ⋅ adj_ser[a+1]) for a in 0:(length(adj_ser)-1)]
					σ_adj_j = [σ_lame(ε, λ, μ) for ε in ε_adj_j]

					bracket = poly_contract(ε_adj_i, σ_adj_j, N_θ)
					K_ser = poly_mul(bracket, inv_det_ser, N_θ)
					for k in 0:N_θ
						ke[k+1][i, j] += K_ser[k+1] * dΩ₀
					end

					NiNj = Ni ⋅ Nj
					for k in 0:min(N_θ, length(det_ser)-1)
						me[k+1][i, j] += ρ * NiNj * det_ser[k+1] * dΩ₀
					end
				end
			end
		end

		for k in eachindex(ke)
			;
			assemble!(assemblers_K[k], celldofs(cell), ke[k]);
		end
		for k in eachindex(me)
			;
			assemble!(assemblers_M[k], celldofs(cell), me[k]);
		end
	end
	return nothing
end

# ===========================================================================
# 2.  ArchGeometricNonlinearity — nonlinear maps with per-QP series
# ===========================================================================

"""
	ArchGeometricNonlinearity{N_input}

Holds the data needed to evaluate the N_input-displacement elastic form
at any θ-power, for the sinusoidal arch beam.

Unlike `ParametricGeometricNonlinearity` from example 04, no global adj/det
series is stored: the series are computed per quadrature point from `h₀`, `L`
and `arch_jacobian_pair`.

N_input = 2 for the quadratic (St-Venant–Kirchhoff second-order) form,
 = 3 for the cubic form.
"""
struct ArchGeometricNonlinearity{N_input}
	dh            :: DofHandler
	cv            :: CellValues
	λ            :: Float64
	μ            :: Float64
	h₀          :: Float64
	L             :: Float64
	free_to_local :: Dict{Int, Int}
	n_free        :: Int
	N_θ          :: Int
end

# ===========================================================================
# 3.  Quadratic kernel
# ===========================================================================

"""
	evaluate_kth_quadratic!(res, pgn::ArchGeometricNonlinearity{2}, k, u₁, u₂)

Compute the θ^k coefficient of the quadratic form g(u₁, u₂; θ) and accumulate
into `res`.  The per-QP inv_det² series is computed on the fly.
"""
function evaluate_kth_quadratic!(res::AbstractVector{T},
	pgn::ArchGeometricNonlinearity{2},
	k::Int,
	u₁::AbstractVector{T},
	u₂::AbstractVector{T}) where {T}
	@assert 0 ≤ k ≤ pgn.N_θ
	fill!(res, zero(T))

	cv = pgn.cv
	λ, μ = pgn.λ, pgn.μ
	N_θ = pgn.N_θ
	n_basefuncs = getnbasefunctions(cv)
	n_dofs_cell = ndofs_per_cell(pgn.dh)

	u₁e = zeros(T, n_dofs_cell)
	u₂e = zeros(T, n_dofs_cell)
	re = zeros(T, n_dofs_cell)

	for cell in CellIterator(pgn.dh)
		reinit!(cv, cell)
		dofs = celldofs(cell)
		coords = getcoordinates(cell)
		gather_local!(u₁e, u₁, dofs, pgn.free_to_local)
		gather_local!(u₂e, u₂, dofs, pgn.free_to_local)
		fill!(re, zero(T))

		for q_point in 1:getnquadpoints(cv)
			dΩ₀ = getdetJdV(cv, q_point)
			x₀_qp = spatial_coordinate(cv, q_point, coords)

			J₀_qp, J₁_qp = arch_jacobian_pair(x₀_qp, pgn.h₀, pgn.L)
			det_ser, adj_ser = det_and_adj_series(J₀_qp, J₁_qp)
			inv_det_ser = reciprocal_series(det_ser, N_θ)
			inv_det2_ser = poly_mul(inv_det_ser, inv_det_ser, N_θ)

			∇u1 = function_gradient(cv, q_point, u₁e)
			∇u2 = function_gradient(cv, q_point, u₂e)

			∇u1_adj = ∇adj_series(∇u1, adj_ser)
			∇u2_adj = ∇adj_series(∇u2, adj_ser)

			ε_u1 = [symmetric(g) for g in ∇u1_adj]
			ε_u2 = [symmetric(g) for g in ∇u2_adj]
			σ_u1 = [σ_lame(ε, λ, μ) for ε in ε_u1]
			σ_u2 = [σ_lame(ε, λ, μ) for ε in ε_u2]

			E12 = E_nl_adj_series(∇u1_adj, ∇u2_adj, N_θ)
			σE12 = [σ_lame(E, λ, μ) for E in E12]

			for I in 1:n_basefuncs
				∇NI = shape_gradient(cv, q_point, I)
				∇NI_adj = ∇adj_series(∇NI, adj_ser)
				ε_v = [symmetric(g) for g in ∇NI_adj]

				t1 = poly_contract(ε_v, σE12, N_θ)

				M2 = poly_dot([transpose(g) for g in ∇u1_adj], ∇NI_adj, N_θ)
				S2 = [symmetric(A) for A in M2]
				t2 = poly_contract(S2, σ_u2, N_θ)

				M3 = poly_dot([transpose(g) for g in ∇u2_adj], ∇NI_adj, N_θ)
				S3 = [symmetric(A) for A in M3]
				t3 = poly_contract(S3, σ_u1, N_θ)

				integ_ser = [t1[m+1] + 0.5 * (t2[m+1] + t3[m+1]) for m in 0:N_θ]
				with_invdet2 = poly_mul(integ_ser, inv_det2_ser, N_θ)
				re[I] += with_invdet2[k+1] * dΩ₀
			end
		end
		scatter_local!(res, re, dofs, pgn.free_to_local)
	end
	return res
end

# ===========================================================================
# 4.  Cubic kernel
# ===========================================================================

"""
	evaluate_kth_cubic!(res, pgn::ArchGeometricNonlinearity{3}, k, u₁, u₂, u₃)

Compute the θ^k coefficient of the cubic form h(u₁, u₂, u₃; θ).
"""
function evaluate_kth_cubic!(res::AbstractVector{T},
	pgn::ArchGeometricNonlinearity{3},
	k::Int,
	u₁::AbstractVector{T},
	u₂::AbstractVector{T},
	u₃::AbstractVector{T}) where {T}
	@assert 0 ≤ k ≤ pgn.N_θ
	fill!(res, zero(T))

	cv = pgn.cv
	λ, μ = pgn.λ, pgn.μ
	N_θ = pgn.N_θ
	n_basefuncs = getnbasefunctions(cv)
	n_dofs_cell = ndofs_per_cell(pgn.dh)

	u₁e = zeros(T, n_dofs_cell)
	u₂e = zeros(T, n_dofs_cell)
	u₃e = zeros(T, n_dofs_cell)
	re = zeros(T, n_dofs_cell)

	for cell in CellIterator(pgn.dh)
		reinit!(cv, cell)
		dofs = celldofs(cell)
		coords = getcoordinates(cell)
		gather_local!(u₁e, u₁, dofs, pgn.free_to_local)
		gather_local!(u₂e, u₂, dofs, pgn.free_to_local)
		gather_local!(u₃e, u₃, dofs, pgn.free_to_local)
		fill!(re, zero(T))

		for q_point in 1:getnquadpoints(cv)
			dΩ₀ = getdetJdV(cv, q_point)
			x₀_qp = spatial_coordinate(cv, q_point, coords)

			J₀_qp, J₁_qp = arch_jacobian_pair(x₀_qp, pgn.h₀, pgn.L)
			det_ser, adj_ser = det_and_adj_series(J₀_qp, J₁_qp)
			inv_det_ser = reciprocal_series(det_ser, N_θ)
			inv_det3_ser = poly_mul(poly_mul(inv_det_ser, inv_det_ser, N_θ),
				inv_det_ser, N_θ)

			∇u1 = function_gradient(cv, q_point, u₁e)
			∇u2 = function_gradient(cv, q_point, u₂e)
			∇u3 = function_gradient(cv, q_point, u₃e)

			∇u1_adj = ∇adj_series(∇u1, adj_ser)
			∇u2_adj = ∇adj_series(∇u2, adj_ser)
			∇u3_adj = ∇adj_series(∇u3, adj_ser)

			E12 = E_nl_adj_series(∇u1_adj, ∇u2_adj, N_θ);
			σE12 = [σ_lame(E, λ, μ) for E in E12]
			E13 = E_nl_adj_series(∇u1_adj, ∇u3_adj, N_θ);
			σE13 = [σ_lame(E, λ, μ) for E in E13]
			E23 = E_nl_adj_series(∇u2_adj, ∇u3_adj, N_θ);
			σE23 = [σ_lame(E, λ, μ) for E in E23]

			for I in 1:n_basefuncs
				∇NI = shape_gradient(cv, q_point, I)
				∇NI_adj = ∇adj_series(∇NI, adj_ser)

				A1 = poly_dot([transpose(g) for g in ∇u1_adj], ∇NI_adj, N_θ)
				A2 = poly_dot([transpose(g) for g in ∇u2_adj], ∇NI_adj, N_θ)
				A3 = poly_dot([transpose(g) for g in ∇u3_adj], ∇NI_adj, N_θ)
				S1 = [symmetric(A) for A in A1]
				S2 = [symmetric(A) for A in A2]
				S3 = [symmetric(A) for A in A3]

				t1 = poly_contract(S1, σE23, N_θ)
				t2 = poly_contract(S2, σE13, N_θ)
				t3 = poly_contract(S3, σE12, N_θ)

				integ_ser = [(1 / 3) * (t1[m+1] + t2[m+1] + t3[m+1]) for m in 0:N_θ]
				with_invdet3 = poly_mul(integ_ser, inv_det3_ser, N_θ)
				re[I] += with_invdet3[k+1] * dΩ₀
			end
		end
		scatter_local!(res, re, dofs, pgn.free_to_local)
	end
	return res
end

# ===========================================================================
# 5.  Fixed-arity closure factories (N_EXT = 1)
# ===========================================================================
#
# Each external slot is an SVector{1,Int}; its single component r[1] is the
# scalar prefactor for that slot.  MORFE's canonical tuple has all slots
# pointing at the same external state (θ), so the product is just r₁[1]·…·r_k[1].

const _ARCH_MAX_EXT = 8

for kk in 0:_ARCH_MAX_EXT
	ext_args = [Symbol("r$i") for i in 1:kk]
	if kk == 0
		@eval _arch_linK(::Val{0}, Kk) = (res, u) -> (res .-= (Kk * u))
		@eval _arch_linC(::Val{0}, Cc) = (res, v) -> (res .-= (Cc * v))
		@eval _arch_linM(::Val{0}, Mk) = (res, a) -> (res .-= (Mk * a))
		@eval _arch_quad(::Val{0}, pgn, k, buf) = (res, u₁, u₂) -> begin
			evaluate_kth_quadratic!(buf, pgn, k, u₁, u₂)
			res .-= buf
		end
		@eval _arch_cube(::Val{0}, pgn, k, buf) = (res, u₁, u₂, u₃) -> begin
			evaluate_kth_cubic!(buf, pgn, k, u₁, u₂, u₃)
			res .-= buf
		end
	else
		factor = Expr(:call, :*, [:(($r)[1]) for r in ext_args]...)
		@eval _arch_linK(::Val{$kk}, Kk) =
			(res, u, $(ext_args...)) -> (res .-= ($factor) .* (Kk * u))
		@eval _arch_linC(::Val{$kk}, Cc) =
			(res, v, $(ext_args...)) -> (res .-= ($factor) .* (Cc * v))
		@eval _arch_linM(::Val{$kk}, Mk) =
			(res, a, $(ext_args...)) -> (res .-= ($factor) .* (Mk * a))
		@eval _arch_quad(::Val{$kk}, pgn, k, buf) =
			(res, u₁, u₂, $(ext_args...)) -> begin
				evaluate_kth_quadratic!(buf, pgn, k, u₁, u₂)
				res .-= ($factor) .* buf
			end
		@eval _arch_cube(::Val{$kk}, pgn, k, buf) =
			(res, u₁, u₂, u₃, $(ext_args...)) -> begin
				evaluate_kth_cubic!(buf, pgn, k, u₁, u₂, u₃)
				res .-= ($factor) .* buf
			end
	end
end

# ===========================================================================
# 6.  multilinear_maps — wrap all θ-powers as MORFE.MultilinearMap
# ===========================================================================

function multilinear_maps(pgn::ArchGeometricNonlinearity{2})
	maps = MultilinearMap[]
	for k in 0:pgn.N_θ
		buf = zeros(ComplexF64, pgn.n_free)
		cl = _arch_quad(Val(k), pgn, k, buf)
		push!(maps, MultilinearMap(cl, (2, 0, 0), k))
	end
	return maps
end

function multilinear_maps(pgn::ArchGeometricNonlinearity{3})
	maps = MultilinearMap[]
	for k in 0:pgn.N_θ
		buf = zeros(ComplexF64, pgn.n_free)
		cl = _arch_cube(Val(k), pgn, k, buf)
		push!(maps, MultilinearMap(cl, (3, 0, 0), k))
	end
	return maps
end

# ===========================================================================
# 7.  Correction builders (K, C, M)
# ===========================================================================

"""
	build_arch_K_corrections(K_arr, N_θ) -> Vector{MultilinearMap}

Wrap K_arr[k+1] for k = 1..N_θ as linear stiffness corrections
`−θ^k · K_k · u` (modal arity (1,0,0), external arity k).
"""
function build_arch_K_corrections(K_arr::Vector, N_θ::Int)
	corr = MultilinearMap[]
	for k in 1:N_θ
		Kk = K_arr[k+1]
		nnz(Kk) > 0 || continue
		push!(corr, MultilinearMap(_arch_linK(Val(k), Kk), (1, 0, 0), k))
	end
	return corr
end

"""
	build_arch_C_corrections(K_arr, M_arr, α, β, N_θ) -> Vector{MultilinearMap}

Parametric Rayleigh damping C(θ) = α M(θ) + β K(θ).  Builds C_k = α M_k + β K_k
for k = 1..N_θ (modal arity (0,1,0), external arity k).
"""
function build_arch_C_corrections(K_arr::Vector, M_arr::Vector,
	α::Float64, β::Float64, N_θ::Int)
	corr = MultilinearMap[]
	for k in 1:N_θ
		Ck = β != 0 ? β * K_arr[k+1] : nothing
		if α != 0
			Ck = Ck === nothing ? α * M_arr[k+1] : Ck + α * M_arr[k+1]
		end
		Ck === nothing && continue
		nnz(Ck) > 0 || continue
		push!(corr, MultilinearMap(_arch_linC(Val(k), Ck), (0, 1, 0), k))
	end
	return corr
end

"""
	build_arch_M_corrections(M_arr, N_θ) -> Vector{MultilinearMap}

Wrap M_arr[k+1] for k = 1..N_θ as inertial corrections
`−θ^k · M_k · ü` (modal arity (0,0,1), external arity k).
Requires ORD=3 in the NDOrderModel.
"""
function build_arch_M_corrections(M_arr::Vector, N_θ::Int)
	corr = MultilinearMap[]
	for k in 1:N_θ
		Mk = M_arr[k+1]
		nnz(Mk) > 0 || continue
		push!(corr, MultilinearMap(_arch_linM(Val(k), Mk), (0, 0, 1), k))
	end
	return corr
end
