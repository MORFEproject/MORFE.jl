"""
	parametric_assembly.jl

Polynomial-in-θ versions of the linear and nonlinear weak forms of the
benchmark demo, written **without ever materialising a series for
`J⁻¹(θ)`**.  At every quadrature point of the reference mesh we
substitute the cofactor identity

	J⁻¹  =  adj(J) / det(J)

into the pulled-back weak form and let one factor of `1/det(J)` cancel
the `det(J)` from the volume differential `dV = det(J) · dV₀`.  What
remains is the *minimal* number of reciprocal-series factors:

	  a(u, v; θ)  :  bracket_K(adj)   ·  (1/det)¹          (linear K)
	  m(u, v; θ)  :  ρ (u · v)        ·  det¹              (linear M)
	  g(u₁,u₂,v;θ): bracket_G(adj)    ·  (1/det)²          (quadratic)
	  h(u₁,u₂,u₃,v;θ): bracket_H(adj) ·  (1/det)³          (cubic)

where the brackets are polynomials in θ obtained from products of
`adj(J(θ))` (exact, degree ≤ 2 in 3D) and the reference gradients.

The pattern is uniform: an `N_input`-displacement elastic form carries
`(1/det)^{N_input}` after `dV` cancellation.  `N_input = 1` for the
linear stiffness, 2 for the quadratic, 3 for the cubic.  Mass is the
only form that *doesn't* have any reciprocal factor (because it has no
∇), and instead carries one positive power of `det(J)`.

This file exposes

	inv_detJ_power(inv_detJ_coeffs, n, N_θ)
		— n-th power of the reciprocal series, truncated to N_θ.

	assemble_K_M_polynomial!(K_coeffs, M_coeffs, dh, cv, λ, μ, ρ,
							 adjJ_coeffs, detJ_coeffs, inv_detJ_coeffs)
		— fill the K_k, M_k coefficient matrices.

	ParametricGeometricNonlinearity{N_input}
		— holds dh, cv, λ, μ, adjJ_coeffs, and `(1/det)^{N_input}`
		  (precomputed at construction time).

	evaluate_kth_quadratic!  /  evaluate_kth_cubic!
		— evaluate `g(·; θ)` or `h(·; θ)` at θ-power `k`, given input
		  modal vectors; populate a free-DOF residual.

	multilinear_maps(pgn)
		— wrap each θ-power coefficient as a `MORFE.MultilinearMap`
		  of external arity `k`.

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
							 adjJ_coeffs, detJ_coeffs, inv_detJ_coeffs)

Fill the θ-power-coefficient matrices:

	K_coeffs[k+1] = K_k     (θ^k-coefficient of the linear stiffness form)
	M_coeffs[k+1] = M_k     (θ^k-coefficient of the linear mass form)

The K integrand, **after the `1/det(J)` × `det(J) = 1` cancellation
between one J⁻¹ factor and the volume differential `dV`**, reads

	K_integrand × dV₀  =  ε_adj(v) ⊡ σ(ε_adj(u))  ·  (1/det(J))  · dV₀

with `ε_adj(u) := sym(∇₀u · adj(J(θ)))`.  No tensor-valued reciprocal
series is built; the only reciprocal that appears is the scalar
`inv_detJ_coeffs`, applied at the very end of the QP loop as a single
`poly_mul`.

The M integrand has no `J⁻¹` factor, only one positive power of
`det(J)`:

	M_integrand × dV₀  =  ρ (u · v) · det(J)  · dV₀.

Both vectors must be pre-allocated with sparse matrices of the same
sparsity pattern.  Truncation levels are inferred from
`length(K_coeffs) - 1` and `length(M_coeffs) - 1`; typical choices

	length(K_coeffs)   = N_θ + 1
	length(M_coeffs)   = length(detJ_coeffs)   (= 4 in 3D, exact)
"""
function assemble_K_M_polynomial!(K_coeffs::Vector,
	M_coeffs::Vector,
	dh::DofHandler,
	cv::CellValues,
	λ::Float64, μ::Float64, ρ::Float64,
	adjJ_coeffs::Vector,
	detJ_coeffs::Vector,
	inv_detJ_coeffs::Vector)
	N_K    = length(K_coeffs) - 1
	N_M    = length(M_coeffs) - 1
	N_adj  = length(adjJ_coeffs) - 1
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
				# ε_adj(N_i) := sym(∇N_i · adj(J(θ)))  — exact length-3 series
				ε_adj_i = [symmetric(∇Ni ⋅ adjJ_coeffs[a+1]) for a in 0:N_adj]

				for j in 1:n_basefuncs
					∇Nj = shape_gradient(cv, q_point, j)
					Nj = shape_value(cv, q_point, j)
					ε_adj_j = [symmetric(∇Nj ⋅ adjJ_coeffs[a+1]) for a in 0:N_adj]
					σ_adj_j = [σ_of(ε_adj_j[a+1]) for a in 0:N_adj]

					# bracket = ε_adj_i ⊡ σ_adj_j   (true degree ≤ 4 in θ)
					bracket = poly_contract(ε_adj_i, σ_adj_j, N_K)
					# Apply (1/det(J))¹ — the *only* reciprocal factor that
					# survives the dV cancellation in the linear stiffness.
					K_ser = poly_mul(bracket, inv_detJ_coeffs, N_K)
					for k in 0:N_K
						ke[k+1][i, j] += K_ser[k+1] * dΩ₀
					end

					# mass: ρ (N_i · N_j) · det J   — no reciprocal anywhere.
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

After the `dV` cancellation in the pulled-back weak form,
`N_input`-displacement elastic terms carry `(1/det(J(θ)))^{N_input}`
as their *only* reciprocal factor.  We precompute that power once at
construction time so the QP loop is a plain `poly_mul`.

Fields
------
	dh, cv                  — Ferrite DofHandler / CellValues
	λ, μ                    — Lamé parameters
	adjJ_coeffs             — Vector{Tens3} of length N_adj + 1
							  (length 3 in 3D; degree-2 exact polynomial)
	inv_detJ_power_coeffs   — `(1/det(J))^{N_input}` truncated at N_θ
	free_to_local           — Dict mapping global DOF → free-vector index
	n_free                  — length of the free-DOF vector
	N_θ                     — series truncation order in θ
"""
struct ParametricGeometricNonlinearity{N_input}
	dh                    :: DofHandler
	cv                    :: CellValues
	λ                    :: Float64
	μ                    :: Float64
	adjJ_coeffs           :: Vector{Tens3}
	inv_detJ_power_coeffs :: Vector{Float64}
	free_to_local         :: Dict{Int, Int}
	n_free                :: Int
	N_θ                  :: Int
end

"""
	inv_detJ_power(inv_detJ_coeffs, n, N_θ) -> Vector{Float64}

Compute the `n`-th power of the reciprocal scalar series `1/det(J)`,
truncated at order `N_θ`.  `n` must be a non-negative integer.

This is the convenience helper used both by the constructor of
`ParametricGeometricNonlinearity{N_input}` (with `n = N_input`) and by
callers that need the same power for K-assembly (n = 1) or M
(n = 0, returns `[1, 0, …, 0]`).
"""
function inv_detJ_power(inv_detJ_coeffs::Vector{Float64}, n::Int, N_θ::Int)
	@assert n ≥ 0 "n must be ≥ 0"
	out = [k == 0 ? 1.0 : 0.0 for k in 0:N_θ]    # the constant series 1
	for _ in 1:n
		out = poly_mul(out, inv_detJ_coeffs, N_θ)
	end
	return out
end

function ParametricGeometricNonlinearity{N_input}(
	dh::DofHandler, cv::CellValues,
	free_to_local::Dict{Int, Int}, n_free::Int,
	λ::Float64, μ::Float64;
	adjJ_coeffs::Vector,
	inv_detJ_coeffs::Vector,
	N_θ::Int) where {N_input}
	# Precompute (1/det(J))^{N_input} so the QP loop is one poly_mul
	# against this single series — no per-call exponentiation.
	inv_detJ_power_coeffs = inv_detJ_power(inv_detJ_coeffs, N_input, N_θ)
	return ParametricGeometricNonlinearity{N_input}(
		dh, cv, λ, μ, adjJ_coeffs, inv_detJ_power_coeffs,
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
	∇adj_series(∇u, adjJ_coeffs) -> Vector

θ-series of `∇₀u · adj(J(θ))` at a quadrature point, given the
reference gradient `∇u`.  Length = `length(adjJ_coeffs) = 3` in 3D
(degree-2 polynomial, exact).

Note that this is **not** `∇u · J⁻¹(θ)`; it is `∇u · adj(J(θ))`.
The missing factor of `1/det(J)` is restored *globally* on the
integrand at the end of each QP loop, where one factor cancels the
`det(J)` from `dV` and the rest are folded into one `poly_mul`.
"""
@inline function ∇adj_series(∇u, adjJ_coeffs::Vector)
	return [∇u ⋅ adjJ_coeffs[a+1] for a in 0:(length(adjJ_coeffs)-1)]
end

"""
	E_nl_adj_series(∇uA_adj_ser, ∇uB_adj_ser, N_θ) -> Vector{SymmetricTensor{2,3,T}}

θ-series of `¼ (∇_adj u_Aᵀ · ∇_adj u_B + ∇_adj u_Bᵀ · ∇_adj u_A)`,
truncated at order N_θ, with `∇_adj := ∇₀(·) · adj(J(θ))`.  Each
input series has length 3, so the output has effective degree ≤ 4;
we cap at N_θ.

This is the strain quantity that *would* be `E_nl(u_A, u_B; θ)` if
`adj(J)` were replaced by `J⁻¹`.  The missing factor `1/det²` is
reattached at the integrand level by the caller (one `1/det` from
each `∇_θ` factor; the `dV` cancellation has already absorbed one
power, so the quadratic form ends up with `(1/det)²`).
"""
function E_nl_adj_series(∇uA_adj_ser::Vector, ∇uB_adj_ser::Vector, N_θ::Int)
	AB = poly_dot([transpose(g) for g in ∇uA_adj_ser], ∇uB_adj_ser, N_θ)
	BA = poly_dot([transpose(g) for g in ∇uB_adj_ser], ∇uA_adj_ser, N_θ)
	return [symmetric(0.25 * (AB[k+1] + BA[k+1])) for k in 0:N_θ]
end

# ------------------------------------------------------------
# 2.3  Quadratic residual  res = [g(u₁, u₂; θ)]_k
# ------------------------------------------------------------
"""
	evaluate_kth_quadratic!(res, pgn, k, u₁, u₂)

Compute the θ^k coefficient of the quadratic Galerkin form `g(u₁, u₂; θ)`,
written as a free-DOF residual vector `res`.  Overwrites `res`.

Implementation
--------------
At each quadrature point we form three "bracket" series in θ,

	t1 = ε_adj(v)                   ⊡ σ(E_nl_adj(u₁, u₂))
	t2 = ½ sym(∇_adj u₁ᵀ · ∇_adj v) ⊡ σ(ε_adj(u₂))
	t3 = ½ sym(∇_adj u₂ᵀ · ∇_adj v) ⊡ σ(ε_adj(u₁))

using ONLY products with `adj(J(θ))` (no `J⁻¹`).  The factor
`(1/det(J))²` (= one `1/det` from each `∇_θ`, with one already absorbed
into `dV`) is applied at the end as a single `poly_mul` against
`pgn.inv_detJ_power_coeffs`, precomputed in the constructor.

Generic in the element type: works for real or complex `u₁, u₂` (and
matches the eltype of `res`).  DPIM passes `ComplexF64`.
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

			∇u1_adj = ∇adj_series(∇u1, pgn.adjJ_coeffs)
			∇u2_adj = ∇adj_series(∇u2, pgn.adjJ_coeffs)

			ε_u1 = [symmetric(g) for g in ∇u1_adj]
			ε_u2 = [symmetric(g) for g in ∇u2_adj]
			σ_u1 = [σ_lame(ε, λ, μ) for ε in ε_u1]
			σ_u2 = [σ_lame(ε, λ, μ) for ε in ε_u2]

			E12 = E_nl_adj_series(∇u1_adj, ∇u2_adj, N_θ)
			σE12 = [σ_lame(E, λ, μ) for E in E12]

			for I in 1:n_basefuncs
				∇NI     = shape_gradient(cv, q_point, I)
				∇NI_adj = ∇adj_series(∇NI, pgn.adjJ_coeffs)
				ε_v      = [symmetric(g) for g in ∇NI_adj]

				# t1 :  ε_v ⊡ σ(E_nl(u₁, u₂))
				t1 = poly_contract(ε_v, σE12, N_θ)

				# t2 :  ½ sym(∇u₁ᵀ · ∇v) ⊡ σ(ε(u₂))
				M2 = poly_dot([transpose(g) for g in ∇u1_adj], ∇NI_adj, N_θ)
				S2 = [symmetric(A) for A in M2]
				t2 = poly_contract(S2, σ_u2, N_θ)

				# t3 :  ½ sym(∇u₂ᵀ · ∇v) ⊡ σ(ε(u₁))
				M3 = poly_dot([transpose(g) for g in ∇u2_adj], ∇NI_adj, N_θ)
				S3 = [symmetric(A) for A in M3]
				t3 = poly_contract(S3, σ_u1, N_θ)

				integ_ser = [t1[m+1] + 0.5 * (t2[m+1] + t3[m+1]) for m in 0:N_θ]
				# Apply (1/det(J))² — residual reciprocal factor for a
				# 2-input elastic form after the dV cancellation.
				with_invdet2 = poly_mul(integ_ser, pgn.inv_detJ_power_coeffs, N_θ)
				re[I] += with_invdet2[k+1] * dΩ₀
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
as a free-DOF residual vector.

Implementation uses `adj(J(θ))` throughout and applies `(1/det(J))³`
as a single final `poly_mul` (= one `1/det` from each of the four
`∇_θ` factors, minus one absorbed into `dV`).  Generic in element
type T (see the quadratic kernel for the rationale).
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

			∇u1_adj = ∇adj_series(∇u1, pgn.adjJ_coeffs)
			∇u2_adj = ∇adj_series(∇u2, pgn.adjJ_coeffs)
			∇u3_adj = ∇adj_series(∇u3, pgn.adjJ_coeffs)

			E12 = E_nl_adj_series(∇u1_adj, ∇u2_adj, N_θ);
			σE12 = [σ_lame(E, λ, μ) for E in E12]
			E13 = E_nl_adj_series(∇u1_adj, ∇u3_adj, N_θ);
			σE13 = [σ_lame(E, λ, μ) for E in E13]
			E23 = E_nl_adj_series(∇u2_adj, ∇u3_adj, N_θ);
			σE23 = [σ_lame(E, λ, μ) for E in E23]

			for I in 1:n_basefuncs
				∇NI     = shape_gradient(cv, q_point, I)
				∇NI_adj = ∇adj_series(∇NI, pgn.adjJ_coeffs)

				A1 = poly_dot([transpose(g) for g in ∇u1_adj], ∇NI_adj, N_θ)
				A2 = poly_dot([transpose(g) for g in ∇u2_adj], ∇NI_adj, N_θ)
				A3 = poly_dot([transpose(g) for g in ∇u3_adj], ∇NI_adj, N_θ)
				S1 = [symmetric(A) for A in A1]
				S2 = [symmetric(A) for A in A2]
				S3 = [symmetric(A) for A in A3]

				t1 = poly_contract(S1, σE23, N_θ)
				t2 = poly_contract(S2, σE13, N_θ)
				t3 = poly_contract(S3, σE12, N_θ)

				integ_ser = [(1/3) * (t1[m+1] + t2[m+1] + t3[m+1]) for m in 0:N_θ]
				# Apply (1/det(J))³ — residual reciprocal factor for a
				# 3-input elastic form after the dV cancellation.
				with_invdet3 = poly_mul(integ_ser, pgn.inv_detJ_power_coeffs, N_θ)
				re[I] += with_invdet3[k+1] * dΩ₀
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

# MORFE passes each external slot as an `SVector{N_EXT, Int}` (a unit
# index-vector identifying which external state the slot points at),
# not as a scalar.  See the `_replay_split!` signature in
# `MORFE.MultilinearTerms`:  `unit_vectors::Vector{SVector{N_EXT, Int}}`.
#
# For a single external state (N_EXT = 1), each slot is therefore an
# `SVector{1, Int}` holding a single integer (typically `[1]`).  To
# get the scalar prefactor `r₁ · r₂ · … · r_k` we must index each
# slot's single component explicitly: `r_i[1]`.  This is the
# difference between this file and the benchmark's `term_forcing`
# example: the latter uses `@. F_ext * r1 * r2 * …`, where broadcast
# fortuitously lifts the elementwise product, masking the fact that
# `*` between two `SVector{1}`s is undefined.
#
# Note: this code therefore assumes N_EXT == 1.  For multi-parameter
# generalisations (N_EXT > 1) each `r_i[j]` would select a different
# external state, and the factory below must be rewritten to dispatch
# on which external axis each slot points at.

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
		# Each external slot rᵢ is SVector{1, Int}; rᵢ[1] is its scalar
		# component.  factor = (r₁[1] * r₂[1] * … * r_k[1])   (a scalar Int)
		factor = Expr(:call, :*, [:(($r)[1]) for r in ext_args]...)
		@eval _wrap_linK(::Val{$kk}, Kk) =
			(res, u, $(ext_args...)) -> (res .-= ($factor) .* (Kk * u))
		@eval _wrap_linC(::Val{$kk}, Cc) =
			(res, v, $(ext_args...)) -> (res .-= ($factor) .* (Cc * v))
		@eval _wrap_quad(::Val{$kk}, pgn, k, buf) =
			(res, u₁, u₂, $(ext_args...)) -> begin
				evaluate_kth_quadratic!(buf, pgn, k, u₁, u₂)
				res .-= ($factor) .* buf
			end
		@eval _wrap_cube(::Val{$kk}, pgn, k, buf) =
			(res, u₁, u₂, u₃, $(ext_args...)) -> begin
				evaluate_kth_cubic!(buf, pgn, k, u₁, u₂, u₃)
				res .-= ($factor) .* buf
			end
	end
end

# Acceleration-type closure factory: M_k · ü · θ^k  (one acceleration slot).
# These require ORD=3 in the NDOrderModel so that W has a third (acceleration)
# column; the multiindex (0, 0, 1) selects that column.
for kk in 0:_MAX_EXT_ARITY
	ext_args = [Symbol("r$i") for i in 1:kk]
	if kk == 0
		@eval _wrap_linM(::Val{0}, Mk) = (res, a) -> (res .-= (Mk * a))
	else
		factor = Expr(:call, :*, [:(($r)[1]) for r in ext_args]...)
		@eval _wrap_linM(::Val{$kk}, Mk) =
			(res, a, $(ext_args...)) -> (res .-= ($factor) .* (Mk * a))
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
	linear_M_correction_closure(Mk, k::Int)

Closure for the inertial correction  `−θ^k · Mk · ü`, arity `k + 2`.
Requires the NDOrderModel to use ORD=3 (multiindex `(0, 0, 1)`).
"""
linear_M_correction_closure(Mk, k::Int) = _wrap_linM(Val(k), Mk)

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
`MultilinearMap` of modal arity `(1, 0, 0)` and external arity `k`.
"""
function build_linear_K_corrections(K_k::Vector, N_K_used::Int)
	corr = MultilinearMap[]
	for k in 1:N_K_used
		Kk = K_k[k+1]
		nnz(Kk) > 0 || continue
		push!(corr, MultilinearMap(linear_K_correction_closure(Kk, k), (1, 0, 0), k))
	end
	return corr
end

"""
	build_linear_C_corrections(K_k, M_k, α, β, N_K_used, N_M_used) -> Vector{MultilinearMap}

Parametric Rayleigh damping  `C(θ) = α M(θ) + β K(θ)`.  Constructs
`C_k = α M_k + β K_k` for k = 1 … max(N_K_used, N_M_used) and wraps
each non-trivial coefficient as a `MultilinearMap` of modal arity
`(0, 1, 0)` and external arity `k`.
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
		push!(corr, MultilinearMap(linear_C_correction_closure(Ck, k), (0, 1, 0), k))
	end
	return corr
end

"""
	build_linear_M_corrections(M_k::Vector, N_M_used::Int) -> Vector{MultilinearMap}

Wrap each non-trivial θ^k coefficient `M_k[k+1]` (k = 1 … N_M_used) as a
`MultilinearMap` of modal arity `(0, 0, 1)` (acceleration slot) and external arity `k`.
Requires ORD=3 in the NDOrderModel so that W carries a third (acceleration) column.
"""
function build_linear_M_corrections(M_k::Vector, N_M_used::Int)
	corr = MultilinearMap[]
	for k in 1:N_M_used
		Mk = M_k[k+1]
		nnz(Mk) > 0 || continue
		push!(corr, MultilinearMap(linear_M_correction_closure(Mk, k), (0, 0, 1), k))
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
		push!(maps, MultilinearMap(quad_external_closure(pgn, k, buf), (2, 0, 0), k))
	end
	return maps
end

function multilinear_maps(pgn::ParametricGeometricNonlinearity{3})
	maps = MultilinearMap[]
	for k in 0:pgn.N_θ
		buf = zeros(ComplexF64, pgn.n_free)
		push!(maps, MultilinearMap(cube_external_closure(pgn, k, buf), (3, 0, 0), k))
	end
	return maps
end
