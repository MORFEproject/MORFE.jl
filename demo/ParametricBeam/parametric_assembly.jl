"""
	parametric_assembly.jl

Polynomial-in-θ versions of the linear and nonlinear weak forms of the
benchmark demo.  At each quadrature point of the *reference* mesh we
work with truncated power series:

	  ∇_θ u  := ∇₀ u · J⁻¹(θ)
	  ε_θ u  := sym(∇_θ u)
	  σ_θ ε  := λ tr(ε) I + 2μ ε
	  E_nl(u, w; θ) := ¼ (∇_θ uᵀ · ∇_θ w + ∇_θ wᵀ · ∇_θ u)
	  Δvol(θ) := det J(θ)

Then the linear, quadratic and cubic Galerkin forms are

	  a(u, v; θ) =  ∫ ε_θ v ⊡ σ_θ(ε_θ u)               · Δvol dV₀
	  m(u, v; θ) =  ∫ ρ (u · v)                         · Δvol dV₀

	  g(u₁, u₂, v; θ) = ∫ [  ε_θ v             ⊡ σ_θ(E_nl(u₁, u₂; θ))
							+ ½ sym(∇_θ u₁ᵀ · ∇_θ v) ⊡ σ_θ(ε_θ u₂)
							+ ½ sym(∇_θ u₂ᵀ · ∇_θ v) ⊡ σ_θ(ε_θ u₁) ] Δvol dV₀

	  h(u₁, u₂, u₃, v; θ) = (1/3) ∫ [  sym(∇_θ u₁ᵀ · ∇_θ v) ⊡ σ_θ(E_nl(u₂, u₃; θ))
									   + sym(∇_θ u₂ᵀ · ∇_θ v) ⊡ σ_θ(E_nl(u₁, u₃; θ))
									   + sym(∇_θ u₃ᵀ · ∇_θ v) ⊡ σ_θ(E_nl(u₁, u₂; θ)) ] Δvol dV₀

All of these become truncated polynomials in θ once `invJ_coeffs` and
`detJ_coeffs` are available.

This file exposes:

	assemble_K_M_polynomial!     — fills the K_k, M_k sequence

	ParametricGeometricNonlinearity{N_input}      — holds dh, cv, λ, μ,
													invJ_coeffs, detJ_coeffs

	evaluate_kth_quadratic!  /  evaluate_kth_cubic!
		— evaluate g(·; θ) or h(·; θ) at θ-power k, given input modal
		  vectors; populate a free-DOF residual.

	multilinear_maps(pgn)        — wrap each θ-power coefficient as a
								   `MORFE.MultilinearMap` of external
								   arity `k`.

NOTE on the closure signature.  MORFE's `MultilinearMap` requires
the closure to take *exactly* `1 + a_pos + a_vel + a_ext` positional
arguments (one `res` plus the modal and external slots, no varargs).
The benchmark forcing example illustrates this with the
external-only signature

	term_forcing = MultilinearMap(
		(res, r) -> (@. res += F_ext * r),
		(0, 0), 1,
	)

i.e. each external slot is its own scalar positional argument.  To
support arbitrary external multiplicity `k`, this file metaprograms
one closure factory per `k` (see `_wrap_linK`, `_wrap_linC`,
`_wrap_quad`, `_wrap_cube` below), each with the right fixed arity.
"""

using Ferrite
using Tensors
using LinearAlgebra
using SparseArrays
using MORFE       # for MultilinearMap

# `Tens3 = Tensor{2, 3, Float64, 9}` is defined in parametric_geometry.jl
# (which must be included before this file).  We rely on that single
# definition rather than redeclaring it, so that all tensor algebra in
# the demo stays unambiguous.

# Assumes theta_polynomials.jl has been included earlier (poly_mul,
# poly_contract, ZERO_TOL).

# ============================================================
# 1.  Linear K(θ), M(θ) coefficient-matrix assembly
# ============================================================

"""
	assemble_K_M_polynomial!(K_coeffs, M_coeffs, dh, cv, λ, μ, ρ,
							 invJ_coeffs, detJ_coeffs)

Fill the θ-power-coefficient matrices:

	K_coeffs[k+1] = K_k    (θ^k-coefficient of the linear stiffness form)
	M_coeffs[k+1] = M_k    (θ^k-coefficient of the linear mass form)

Both vectors must be pre-allocated with sparse matrices of the same
sparsity pattern (the standard `allocate_matrix(dh)`).  The truncation
levels are inferred from `length(K_coeffs) - 1` and `length(M_coeffs) - 1`;
typical choices:

	length(K_coeffs) = N_θ + 1        (truncate K series at the DPIM order)
	length(M_coeffs) = length(detJ_coeffs)
									   (M(θ) has the same finite degree as
										det J(θ), since the integrand
										does not depend on J⁻¹.)
"""
function assemble_K_M_polynomial!(K_coeffs::Vector,
	M_coeffs::Vector,
	dh::DofHandler,
	cv::CellValues,
	λ::Float64, μ::Float64, ρ::Float64,
	invJ_coeffs::Vector,
	detJ_coeffs::Vector)
	N_K    = length(K_coeffs) - 1
	N_M    = length(M_coeffs) - 1
	N_invJ = length(invJ_coeffs) - 1
	N_detJ = length(detJ_coeffs) - 1

	@assert all(eachindex(K_coeffs) .|> i -> size(K_coeffs[i]) == size(K_coeffs[1])) "K_coeffs must share a sparsity pattern"
	@assert all(eachindex(M_coeffs) .|> i -> size(M_coeffs[i]) == size(M_coeffs[1])) "M_coeffs must share a sparsity pattern"

	assemblers_K = [start_assemble(K) for K in K_coeffs]
	assemblers_M = [start_assemble(M) for M in M_coeffs]

	n_basefuncs = getnbasefunctions(cv)
	ke = [zeros(n_basefuncs, n_basefuncs) for _ in 0:N_K]
	me = [zeros(n_basefuncs, n_basefuncs) for _ in 0:N_M]

	σ_of(ε) = λ * tr(ε) * one(ε) + 2μ * ε

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

		for q_point in 1:getnquadpoints(cv)
			dΩ₀ = getdetJdV(cv, q_point)

			for i in 1:n_basefuncs
				∇Ni = shape_gradient(cv, q_point, i)
				Ni = shape_value(cv, q_point, i)
				ε_i = [symmetric(∇Ni ⋅ invJ_coeffs[a+1]) for a in 0:N_invJ]

				for j in 1:n_basefuncs
					∇Nj = shape_gradient(cv, q_point, j)
					Nj = shape_value(cv, q_point, j)
					ε_j = [symmetric(∇Nj ⋅ invJ_coeffs[a+1]) for a in 0:N_invJ]
					σ_j = [σ_of(ε_j[a+1]) for a in 0:N_invJ]

					# series for ε_i ⊡ σ_j  (then × det J)
					εσ_ser = poly_contract(ε_i, σ_j, N_K)
					K_ser = poly_mul(εσ_ser, detJ_coeffs, N_K)
					for k in 0:N_K
						ke[k+1][i, j] += K_ser[k+1] * dΩ₀
					end

					# mass: ρ (N_i · N_j) · det J
					NiNj = (Ni ⋅ Nj)
					for k in 0:min(N_M, N_detJ)
						me[k+1][i, j] += ρ * NiNj * detJ_coeffs[k+1] * dΩ₀
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

# ============================================================
# 2.  Quadratic and cubic parametric nonlinearities
# ============================================================

"""
	ParametricGeometricNonlinearity{N_input}

Holds the data needed to evaluate `g(·, θ)` (N_input = 2) or
`h(·, θ)` (N_input = 3) at any input vectors and any θ-power.

Fields
------
	dh, cv               — Ferrite DofHandler / CellValues
	λ, μ                 — Lamé parameters
	ρ                    — density (not used here, but cached for symmetry)
	invJ_coeffs          — Vector{Tensor{2,3,Float64,9}} of length N_invJ + 1
	detJ_coeffs          — Vector{Float64} of length N_detJ + 1
	free_to_local        — Dict mapping global DOF → free-vector index
	n_free               — length of the free-DOF vector
	N_θ                  — series truncation order in θ

The element type `N_input` is encoded in the type parameter for
dispatch on `evaluate_kth_*` and on `multilinear_maps`.
"""
struct ParametricGeometricNonlinearity{N_input}
	dh            :: DofHandler
	cv            :: CellValues
	λ            :: Float64
	μ            :: Float64
	invJ_coeffs   :: Vector{Tens3}
	detJ_coeffs   :: Vector{Float64}
	free_to_local :: Dict{Int, Int}
	n_free        :: Int
	N_θ          :: Int
end

function ParametricGeometricNonlinearity{N_input}(
	dh::DofHandler, cv::CellValues,
	free_to_local::Dict{Int, Int}, n_free::Int,
	λ::Float64, μ::Float64;
	invJ_coeffs::Vector,
	detJ_coeffs::Vector,
	N_θ::Int) where {N_input}
	return ParametricGeometricNonlinearity{N_input}(
		dh, cv, λ, μ, invJ_coeffs, detJ_coeffs,
		free_to_local, n_free, N_θ,
	)
end

# ------------------------------------------------------------
# 2.1  Gather / scatter helpers
# ------------------------------------------------------------
# Generic in `T` so that the same routines work for both real-valued
# unit tests (T = Float64) and DPIM's complex working type (T = ComplexF64).
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

# ------------------------------------------------------------
# 2.2  Quadrature-point series helpers
# ------------------------------------------------------------
@inline σ_lame(ε, λ::Float64, μ::Float64) = λ * tr(ε) * one(ε) + 2μ * ε

"""
	∇θ_series(∇u, invJ_coeffs) -> Vector

θ-series of  ∇₀u · J⁻¹(θ)  at a quadrature point, given the reference
gradient `∇u`.  Length matches `invJ_coeffs`.
"""
@inline function ∇θ_series(∇u, invJ_coeffs::Vector)
	return [∇u ⋅ invJ_coeffs[a+1] for a in 0:(length(invJ_coeffs)-1)]
end

"""
	E_nl_series(∇uA_ser, ∇uB_ser, N_θ) -> Vector{SymmetricTensor{2,3,Float64}}

θ-series of  E_nl(u_A, u_B; θ) := ¼ (∇_θ u_Aᵀ · ∇_θ u_B + ∇_θ u_Bᵀ · ∇_θ u_A)
truncated at order N_θ.
"""
function E_nl_series(∇uA_ser::Vector, ∇uB_ser::Vector, N_θ::Int)
	AB = poly_dot([transpose(g) for g in ∇uA_ser], ∇uB_ser, N_θ)
	BA = poly_dot([transpose(g) for g in ∇uB_ser], ∇uA_ser, N_θ)
	return [symmetric(0.25 * (AB[k+1] + BA[k+1])) for k in 0:N_θ]
end

# ------------------------------------------------------------
# 2.3  Quadratic residual  res = [g(u₁, u₂; θ)]_k
# ------------------------------------------------------------
"""
	evaluate_kth_quadratic!(res, pgn, k, u₁, u₂)

Compute the θ^k coefficient of the quadratic Galerkin form `g(u₁, u₂; θ)`,
written as a free-DOF residual vector `res`.  Overwrites `res`.

Generic in the element type: works for real or complex `u₁, u₂` (and
matches the eltype of `res`).  DPIM passes `ComplexF64` because the
parametrisation coefficients are complex; unit tests may pass `Float64`.
"""
function evaluate_kth_quadratic!(res::AbstractVector{T},
	pgn::ParametricGeometricNonlinearity{2},
	k::Int,
	u₁::AbstractVector{T},
	u₂::AbstractVector{T}) where {T}
	@assert 0 ≤ k ≤ pgn.N_θ "k=$k outside truncation range 0…$(pgn.N_θ)"
	fill!(res, zero(T))

	cv          = pgn.cv
	λ, μ      = pgn.λ, pgn.μ
	N_θ        = pgn.N_θ
	n_basefuncs = getnbasefunctions(cv)
	n_dofs_cell = ndofs_per_cell(pgn.dh)

	u₁e = zeros(T, n_dofs_cell)
	u₂e = zeros(T, n_dofs_cell)
	re = zeros(T, n_dofs_cell)

	for cell in CellIterator(pgn.dh)
		reinit!(cv, cell)
		dofs = celldofs(cell)
		gather_local!(u₁e, u₁, dofs, pgn.free_to_local)
		gather_local!(u₂e, u₂, dofs, pgn.free_to_local)
		fill!(re, zero(T))

		for q_point in 1:getnquadpoints(cv)
			dΩ₀ = getdetJdV(cv, q_point)

			∇u1 = function_gradient(cv, q_point, u₁e)
			∇u2 = function_gradient(cv, q_point, u₂e)

			∇u1_ser = ∇θ_series(∇u1, pgn.invJ_coeffs)
			∇u2_ser = ∇θ_series(∇u2, pgn.invJ_coeffs)

			ε_u1 = [symmetric(g) for g in ∇u1_ser]
			ε_u2 = [symmetric(g) for g in ∇u2_ser]
			σ_u1 = [σ_lame(ε, λ, μ) for ε in ε_u1]
			σ_u2 = [σ_lame(ε, λ, μ) for ε in ε_u2]

			E12 = E_nl_series(∇u1_ser, ∇u2_ser, N_θ)
			σE12 = [σ_lame(E, λ, μ) for E in E12]

			for I in 1:n_basefuncs
				∇NI     = shape_gradient(cv, q_point, I)
				∇NI_ser = ∇θ_series(∇NI, pgn.invJ_coeffs)
				ε_v      = [symmetric(g) for g in ∇NI_ser]

				# t1 :  ε_v ⊡ σ(E_nl(u₁, u₂))
				t1 = poly_contract(ε_v, σE12, N_θ)

				# t2 :  ½ sym(∇u₁ᵀ · ∇v) ⊡ σ(ε(u₂))
				M2 = poly_dot([transpose(g) for g in ∇u1_ser], ∇NI_ser, N_θ)
				S2 = [symmetric(A) for A in M2]
				t2 = poly_contract(S2, σ_u2, N_θ)

				# t3 :  ½ sym(∇u₂ᵀ · ∇v) ⊡ σ(ε(u₁))
				M3 = poly_dot([transpose(g) for g in ∇u2_ser], ∇NI_ser, N_θ)
				S3 = [symmetric(A) for A in M3]
				t3 = poly_contract(S3, σ_u1, N_θ)

				integ_ser = [t1[m+1] + 0.5 * (t2[m+1] + t3[m+1]) for m in 0:N_θ]
				with_detJ = poly_mul(integ_ser, pgn.detJ_coeffs, N_θ)
				re[I] += with_detJ[k+1] * dΩ₀
			end
		end
		scatter_local!(res, re, dofs, pgn.free_to_local)
	end
	return res
end

# ------------------------------------------------------------
# 2.4  Cubic residual  res = [h(u₁, u₂, u₃; θ)]_k
# ------------------------------------------------------------
"""
	evaluate_kth_cubic!(res, pgn, k, u₁, u₂, u₃)

θ^k coefficient of the cubic Galerkin form `h(u₁, u₂, u₃; θ)`, written
as a free-DOF residual vector.  Generic in element type T (see the
quadratic kernel for the rationale).
"""
function evaluate_kth_cubic!(res::AbstractVector{T},
	pgn::ParametricGeometricNonlinearity{3},
	k::Int,
	u₁::AbstractVector{T},
	u₂::AbstractVector{T},
	u₃::AbstractVector{T}) where {T}
	@assert 0 ≤ k ≤ pgn.N_θ "k=$k outside truncation range 0…$(pgn.N_θ)"
	fill!(res, zero(T))

	cv          = pgn.cv
	λ, μ      = pgn.λ, pgn.μ
	N_θ        = pgn.N_θ
	n_basefuncs = getnbasefunctions(cv)
	n_dofs_cell = ndofs_per_cell(pgn.dh)

	u₁e = zeros(T, n_dofs_cell)
	u₂e = zeros(T, n_dofs_cell)
	u₃e = zeros(T, n_dofs_cell)
	re = zeros(T, n_dofs_cell)

	for cell in CellIterator(pgn.dh)
		reinit!(cv, cell)
		dofs = celldofs(cell)
		gather_local!(u₁e, u₁, dofs, pgn.free_to_local)
		gather_local!(u₂e, u₂, dofs, pgn.free_to_local)
		gather_local!(u₃e, u₃, dofs, pgn.free_to_local)
		fill!(re, zero(T))

		for q_point in 1:getnquadpoints(cv)
			dΩ₀ = getdetJdV(cv, q_point)

			∇u1 = function_gradient(cv, q_point, u₁e)
			∇u2 = function_gradient(cv, q_point, u₂e)
			∇u3 = function_gradient(cv, q_point, u₃e)

			∇u1_ser = ∇θ_series(∇u1, pgn.invJ_coeffs)
			∇u2_ser = ∇θ_series(∇u2, pgn.invJ_coeffs)
			∇u3_ser = ∇θ_series(∇u3, pgn.invJ_coeffs)

			E12 = E_nl_series(∇u1_ser, ∇u2_ser, N_θ);
			σE12 = [σ_lame(E, λ, μ) for E in E12]
			E13 = E_nl_series(∇u1_ser, ∇u3_ser, N_θ);
			σE13 = [σ_lame(E, λ, μ) for E in E13]
			E23 = E_nl_series(∇u2_ser, ∇u3_ser, N_θ);
			σE23 = [σ_lame(E, λ, μ) for E in E23]

			for I in 1:n_basefuncs
				∇NI     = shape_gradient(cv, q_point, I)
				∇NI_ser = ∇θ_series(∇NI, pgn.invJ_coeffs)

				A1 = poly_dot([transpose(g) for g in ∇u1_ser], ∇NI_ser, N_θ)
				A2 = poly_dot([transpose(g) for g in ∇u2_ser], ∇NI_ser, N_θ)
				A3 = poly_dot([transpose(g) for g in ∇u3_ser], ∇NI_ser, N_θ)
				S1 = [symmetric(A) for A in A1]
				S2 = [symmetric(A) for A in A2]
				S3 = [symmetric(A) for A in A3]

				t1 = poly_contract(S1, σE23, N_θ)
				t2 = poly_contract(S2, σE13, N_θ)
				t3 = poly_contract(S3, σE12, N_θ)

				integ_ser = [(1/3) * (t1[m+1] + t2[m+1] + t3[m+1]) for m in 0:N_θ]
				with_detJ = poly_mul(integ_ser, pgn.detJ_coeffs, N_θ)
				re[I] += with_detJ[k+1] * dΩ₀
			end
		end
		scatter_local!(res, re, dofs, pgn.free_to_local)
	end
	return res
end

# ============================================================
# 3.  Fixed-arity closure factories for MORFE.MultilinearMap
# ============================================================
#
# MORFE checks that the closure passed to `MultilinearMap` has *exactly*
# `1 + a_pos + a_vel + a_ext` positional arguments (one `res` plus the
# modal and external slots).  We therefore can't use a variadic
# `(res, u, externals...)`-style closure; the arity must be fixed at
# construction time.
#
# To support arbitrary external multiplicity `k` (up to a sensible cap
# `_MAX_EXT_ARITY`), we metaprogram one explicit closure factory per
# `k`, all generated at file-load time below.  Each factory returns a
# closure with arity matching its `Val{k}` parameter.
#
# Sign convention.  MORFE writes every multilinear term on the
# right-hand side of  `M ẍ + C ẋ + K x = (multilinear terms)`,  so
# internal forces (linear corrections, quadratic / cubic elastic
# forces) are added with a *minus* sign — exactly as in the
# benchmark's `term_cubic = (res, x1, x2, x3) -> res .+= -β x1 x2 x3`.
#
# Note on parametric mass.  MultilinearMap modal arity is
# `(a_pos, a_vel)` — there is no acceleration slot, so a term of the
# form  `θ^k M_k ẍ`  cannot be expressed.  Parametric mass therefore
# enters only indirectly, via the parametric damping
# `C(θ) = α M(θ) + β K(θ)`.  The inertial part `θ^k M_k ẍ` is dropped
# at this level; see README.md for the rationale.

const _MAX_EXT_ARITY = 8        # support θ-multiplicity up to k = 8

for kk in 0:_MAX_EXT_ARITY
	ext_args = [Symbol("r$i") for i in 1:kk]
	if kk == 0
		# No external slots: closures have arity 1 + a_pos + a_vel only.
		@eval _wrap_linK(::Val{0}, Kk) = (res, u) -> (res .-= (Kk * u))
		@eval _wrap_linC(::Val{0}, Cc) = (res, v) -> (res .-= (Cc * v))
		@eval _wrap_quad(::Val{0}, pgn, k, buf) = (res, u₁, u₂) -> begin
			evaluate_kth_quadratic!(buf, pgn, k, u₁, u₂)
			res .-= buf
		end
		@eval _wrap_cube(::Val{0}, pgn, k, buf) = (res, u₁, u₂, u₃) -> begin
			evaluate_kth_cubic!(buf, pgn, k, u₁, u₂, u₃)
			res .-= buf
		end
	else
		factor = Expr(:call, :*, ext_args...)         # r1 * r2 * … * rk
		@eval _wrap_linK(::Val{$kk}, Kk) =
			(res, u, $(ext_args...)) -> (res .-= $factor .* (Kk * u))
		@eval _wrap_linC(::Val{$kk}, Cc) =
			(res, v, $(ext_args...)) -> (res .-= $factor .* (Cc * v))
		@eval _wrap_quad(::Val{$kk}, pgn, k, buf) =
			(res, u₁, u₂, $(ext_args...)) -> begin
				evaluate_kth_quadratic!(buf, pgn, k, u₁, u₂)
				res .-= $factor .* buf
			end
		@eval _wrap_cube(::Val{$kk}, pgn, k, buf) =
			(res, u₁, u₂, u₃, $(ext_args...)) -> begin
				evaluate_kth_cubic!(buf, pgn, k, u₁, u₂, u₃)
				res .-= $factor .* buf
			end
	end
end

# Convenience dispatch helpers (these are what callers use).
"""
	linear_K_correction_closure(Kk, k::Int)

Closure for the linear stiffness correction  `−θ^k · Kk · u`, with
arity `k + 2`.  `Kk` is captured by reference; no allocation is
performed by the closure itself.
"""
linear_K_correction_closure(Kk, k::Int) = _wrap_linK(Val(k), Kk)

"""
	linear_C_correction_closure(Ck, k::Int)

Closure for the linear damping correction  `−θ^k · Ck · u̇`,
arity `k + 2`.  Used to inject parametric Rayleigh damping
`C(θ) = α M(θ) + β K(θ)`.
"""
linear_C_correction_closure(Ck, k::Int) = _wrap_linC(Val(k), Ck)

"""
	quad_external_closure(pgn, k::Int, buf)

Closure for the θ^k-coefficient of the quadratic St-Venant–Kirchhoff
internal force, arity `k + 3`.  `buf` is a free-DOF scratch vector
(typically `Vector{ComplexF64}` to match DPIM's working type); it is
re-used across calls, so the closure is *not* concurrency-safe unless
each `MultilinearMap` gets its own `buf`.
"""
quad_external_closure(pgn, k::Int, buf) = _wrap_quad(Val(k), pgn, k, buf)

"""
	cube_external_closure(pgn, k::Int, buf)

Closure for the θ^k-coefficient of the cubic St-Venant–Kirchhoff
internal force, arity `k + 4`.  See `quad_external_closure` for the
buffer conventions.
"""
cube_external_closure(pgn, k::Int, buf) = _wrap_cube(Val(k), pgn, k, buf)

# ------------------------------------------------------------
# Convenience constructors for the full list of MultilinearMaps
# ------------------------------------------------------------

"""
	build_linear_K_corrections(K_k::Vector, N_K_used::Int) -> Vector{MultilinearMap}

Wrap each non-trivial θ^k coefficient `K_k[k+1]` (k = 1 … N_K_used) as a
`MultilinearMap` of modal arity `(1, 0)` and external arity `k`.
"""
function build_linear_K_corrections(K_k::Vector, N_K_used::Int)
	corr = MultilinearMap[]
	for k in 1:N_K_used
		Kk = K_k[k+1]
		nnz(Kk) > 0 || continue
		push!(corr, MultilinearMap(linear_K_correction_closure(Kk, k), (1, 0), k))
	end
	return corr
end

"""
	build_linear_C_corrections(K_k, M_k, α, β, N_K_used, N_M_used) -> Vector{MultilinearMap}

Parametric Rayleigh damping  `C(θ) = α M(θ) + β K(θ)`.  Constructs
`C_k = α M_k + β K_k` for k = 1 … max(N_K_used, N_M_used) and wraps
each non-trivial coefficient as a `MultilinearMap` of modal arity
`(0, 1)` and external arity `k`.
"""
function build_linear_C_corrections(K_k::Vector, M_k::Vector,
	α::Float64, β::Float64,
	N_K_used::Int, N_M_used::Int)
	corr = MultilinearMap[]
	for k in 1:max(N_K_used, N_M_used)
		Ck = β != 0 && k ≤ N_K_used ? β * K_k[k+1] : nothing
		if α != 0 && k ≤ N_M_used
			Ck = Ck === nothing ? α * M_k[k+1] : Ck + α * M_k[k+1]
		end
		Ck === nothing && continue
		nnz(Ck) > 0 || continue
		push!(corr, MultilinearMap(linear_C_correction_closure(Ck, k), (0, 1), k))
	end
	return corr
end

# ------------------------------------------------------------
# Wrap each θ-power of the quadratic / cubic forms
# ------------------------------------------------------------
"""
	multilinear_maps(pgn::ParametricGeometricNonlinearity{2}) -> Vector{MultilinearMap}
	multilinear_maps(pgn::ParametricGeometricNonlinearity{3}) -> Vector{MultilinearMap}

Wrap each θ-power coefficient `k = 0 … N_θ` of the quadratic (resp.
cubic) St-Venant–Kirchhoff internal force as a `MultilinearMap` of
modal arity `(N_input, 0)` and external arity `k`.

Each map allocates its own free-DOF scratch buffer to keep the
closures self-contained; this avoids cross-talk if DPIM evaluates
maps concurrently.

NOTE.  We do *not* cache the QP series across `k` for the same input
tuple — each call recomputes the QP series and reads out coefficient
k.  This is wasteful (work scales as N_θ² rather than N_θ if every
power is evaluated for the same inputs), but it keeps the code
stateless and concurrency-safe.  Profile-driven caching is left to
follow-up work.
"""
function multilinear_maps(pgn::ParametricGeometricNonlinearity{2})
	maps = MultilinearMap[]
	for k in 0:pgn.N_θ
		# ComplexF64 buffer: DPIM evaluates the multilinear forms at
		# complex modal inputs (parametrisation coefficients are complex).
		buf = zeros(ComplexF64, pgn.n_free)
		push!(maps, MultilinearMap(quad_external_closure(pgn, k, buf), (2, 0), k))
	end
	return maps
end

function multilinear_maps(pgn::ParametricGeometricNonlinearity{3})
	maps = MultilinearMap[]
	for k in 0:pgn.N_θ
		buf = zeros(ComplexF64, pgn.n_free)
		push!(maps, MultilinearMap(cube_external_closure(pgn, k, buf), (3, 0), k))
	end
	return maps
end
