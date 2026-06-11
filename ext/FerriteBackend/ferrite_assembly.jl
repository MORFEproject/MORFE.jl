"""
Ferrite.jl FEM backend for MORFE.jl geometric nonlinearity.

Implements `FEMMultilinearMap` for the St. Venant-Kirchhoff material model,
which generates nonlinearity up to cubic order in the displacement field.

SVK internal virtual work:
	W_int = ∫ S:δE dΩ
where:
	E = ε(u) + ½∇u'∇u          (Green-Lagrange strain)
	S = λ tr(E) I + 2μ E        (2nd Piola-Kirchhoff stress)
	δE = δε + sym(∇u' δ∇u)

Expanding in powers of u gives quadratic and cubic contributions only.
Higher-order material models (e.g. polynomial hyperelastic) are needed for
quartic and higher terms.
"""

# -----------------------------------------------------------------------
# Concrete FEMMultilinearMap type
# -----------------------------------------------------------------------

"""
	FerriteGeometricNonlinearity{DEG, DH, CV} <: MORFE.FEMMultilinearMap{2}

FEM-backed multilinear term for St. Venant-Kirchhoff geometric nonlinearity.

- `DEG = 2` : quadratic g_quad term (two displacement inputs)
- `DEG = 3` : cubic   h_cube term (three displacement inputs)

Type parameter `ORD = 2` means the multiindex lives in an NDOrderModel of
order 2 (second-order ODE). The term only uses position-derivative inputs
(multiindex = (DEG, 0)).

# Fields
- `dh`             — DofHandler
- `cv`             — CellValues (quadrature + interpolation)
- `free_to_local`  — Dict: global DOF index → 1-based index in the free-DOF vector
- `n_free`         — number of free DOFs
- `λ`, `μ`         — Lamé constants
- `multiindex`     — NTuple{2, Int} = (DEG, 0)
- `multiplicity_external` — 0 (no external forcing)
- `deg`            — DEG
- `∇W_qp`         — pre-allocated qp gradient buffer, Matrix{Tensor{2,3,ComplexF64}}(DEG, n_qp)
- `Fe`             — pre-allocated element residual, Vector{ComplexF64}(ndofs_per_cell)
- `u_e`            — pre-allocated element DOF vector, Vector{ComplexF64}(ndofs_per_cell)
- `u_e_re`         — real part of u_e, Vector{Float64}(ndofs_per_cell)
- `u_e_im`         — imaginary part of u_e, Vector{Float64}(ndofs_per_cell)
"""
struct FerriteGeometricNonlinearity{DEG, DH, CV} <: MORFE.FEMMultilinearMap{2}
	dh::DH
	cv::CV
	free_to_local::Dict{Int, Int}
	n_free::Int
	λ::Float64
	μ::Float64
	multiindex::NTuple{2, Int}
	multiplicity_external::Int
	deg::Int
	fully_asymmetric::Union{Nothing, Bool}
	∇W_qp::Matrix{Tensor{2, 3, ComplexF64}}
	Fe::Vector{ComplexF64}
	u_e::Vector{ComplexF64}
	u_e_re::Vector{Float64}
	u_e_im::Vector{Float64}
end

"""
	FerriteGeometricNonlinearity{DEG}(dh, cv, free_to_local, n_free, λ, μ)

Construct with pre-allocated buffers sized from `cv`.
"""
function FerriteGeometricNonlinearity{DEG}(
	dh::DH, cv::CV,
	free_to_local::Dict{Int, Int}, n_free::Int,
	λ::Float64, μ::Float64;
	max_unique_cols::Int = DEG,
	fully_asymmetric::Union{Nothing, Bool} = false) where {DEG, DH, CV}
	n_qp = getnquadpoints(cv)
	n_dofs = ndofs_per_cell(dh)
	∇W_qp = Matrix{Tensor{2, 3, ComplexF64}}(undef, max_unique_cols, n_qp)
	Fe = Vector{ComplexF64}(undef, n_dofs)
	u_e = Vector{ComplexF64}(undef, n_dofs)
	u_e_re = zeros(Float64, n_dofs)
	u_e_im = zeros(Float64, n_dofs)
	return FerriteGeometricNonlinearity{DEG, DH, CV}(
		dh, cv, free_to_local, n_free, λ, μ,
		(DEG, 0), 0, DEG, fully_asymmetric, ∇W_qp, Fe, u_e, u_e_re, u_e_im)
end

# -----------------------------------------------------------------------
# FEMMultilinearMap interface
# -----------------------------------------------------------------------

MORFE.fem_elements(t::FerriteGeometricNonlinearity) = CellIterator(t.dh)

MORFE.fem_n_qp(t::FerriteGeometricNonlinearity) = getnquadpoints(t.cv)

MORFE.fem_ndofs_per_cell(t::FerriteGeometricNonlinearity) = ndofs_per_cell(t.dh)

MORFE.fem_qp_buffer(t::FerriteGeometricNonlinearity) = t.∇W_qp

MORFE.fem_getdetJdV(_element, q, t::FerriteGeometricNonlinearity) = getdetJdV(t.cv, q)

# Called once per element in _replay_fem_split! before the scatter loop (O1).
MORFE.fem_reinit!(element, t::FerriteGeometricNonlinearity) = reinit!(t.cv, element)

"""
	MORFE.scatter_qp!(∇W_col, W_free, element, t)

Scatter the free-DOF vector `W_free` to per-quadrature-point displacement gradients
∇W_col[q] = ∇u(ξ_q).  CellValues must already be reinit!-ed for `element` via
`fem_reinit!` before this call.
"""
function MORFE.scatter_qp!(∇W_col, W_free, element, t::FerriteGeometricNonlinearity)
	dofs = celldofs(element)
	u_e = t.u_e
	for (i, d) in enumerate(dofs)
		local_idx = get(t.free_to_local, d, 0)
		u_e[i] = local_idx == 0 ? zero(ComplexF64) : W_free[local_idx]
	end
	for q in eachindex(∇W_col)
		∇W_col[q] = function_gradient(t.cv, q, u_e)
	end
end

# Lamé stress from a strain tensor.
@inline _σ(E, λ, μ) = λ * tr(E) * one(E) + 2μ * E

# Symmetric Green-Lagrange cross term for two gradients.
@inline _E_nl(∇u1, ∇u2) = symmetric(
	Tensor{2, 3}(0.25 * (transpose(∇u1) ⋅ ∇u2 + transpose(∇u2) ⋅ ∇u1)))

# Extract real/imaginary Float64 tensors from a ComplexF64 gradient tensor.
# map on NTuple is compile-time-unrolled and avoids closure allocation.
@inline _re(G::Tensor{2, 3, ComplexF64}) = Tensor{2, 3, Float64}(map(real, G.data))
@inline _im(G::Tensor{2, 3, ComplexF64}) = Tensor{2, 3, Float64}(map(imag, G.data))

# Float64-specific E_nl: drops the Tensor{2,3}(...) passthrough wrapper, which
# can prevent full inlining of the arithmetic for non-complex inputs.
@inline _E_nl_f64(A::Tensor{2, 3, Float64}, B::Tensor{2, 3, Float64}) =
	symmetric(0.25 * (transpose(A) ⋅ B + transpose(B) ⋅ A))

"""
	MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{2}, mult, element, q, dΩ, t)

Quadratic geometric nonlinearity integrand at one quadrature point:

	fe_r += mult * [ε(φ_r) ⊡ σ(E_nl(∇u1,∇u2))
					+ 0.5*(sym(∇u1'⋅∇φ_r) ⊡ σ(ε(∇u2))
						 + sym(∇u2'⋅∇φ_r) ⊡ σ(ε(∇u1)))] * dΩ

Implemented by decomposing ∇u1 = A+iB, ∇u2 = C+iD into Float64 tensors and
expanding the bilinear form over Re/Im to avoid ComplexF64 tensor allocations.
"""
function MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{2}, mult, _element, q, dΩ,
	t::FerriteGeometricNonlinearity{2})
	∇u1, ∇u2 = ∇W_args
	A = _re(∇u1);
	B = _im(∇u1)
	C = _re(∇u2);
	D = _im(∇u2)

	E_nl_re = _E_nl_f64(A, C) - _E_nl_f64(B, D)
	E_nl_im = _E_nl_f64(A, D) + _E_nl_f64(B, C)
	σ_nl_re = _σ(E_nl_re, t.λ, t.μ)
	σ_nl_im = _σ(E_nl_im, t.λ, t.μ)

	ε1_re = symmetric(A);
	ε1_im = symmetric(B)
	ε2_re = symmetric(C);
	ε2_im = symmetric(D)
	σ_ε1_re = _σ(ε1_re, t.λ, t.μ);
	σ_ε1_im = _σ(ε1_im, t.λ, t.μ)
	σ_ε2_re = _σ(ε2_re, t.λ, t.μ);
	σ_ε2_im = _σ(ε2_im, t.λ, t.μ)

	n_dofs = ndofs_per_cell(t.dh)
	c = mult * dΩ

	for r in 1:n_dofs
		∂Nr = shape_gradient(t.cv, q, r)
		δε  = symmetric(∂Nr)

		cr1_re = symmetric(transpose(A) ⋅ ∂Nr)
		cr1_im = symmetric(transpose(B) ⋅ ∂Nr)
		cr2_re = symmetric(transpose(C) ⋅ ∂Nr)
		cr2_im = symmetric(transpose(D) ⋅ ∂Nr)

		re = δε ⊡ σ_nl_re +
			 0.5 * (cr1_re ⊡ σ_ε2_re - cr1_im ⊡ σ_ε2_im +
					cr2_re ⊡ σ_ε1_re - cr2_im ⊡ σ_ε1_im)

		im = δε ⊡ σ_nl_im +
			 0.5 * (cr1_re ⊡ σ_ε2_im + cr1_im ⊡ σ_ε2_re +
					cr2_re ⊡ σ_ε1_im + cr2_im ⊡ σ_ε1_re)

		Fe[r] -= complex(c * re, c * im)
	end
end

"""
	MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{3}, mult, element, q, dΩ, t)

Cubic geometric nonlinearity integrand at one quadrature point:

	fe_r += mult/3 * Σ_{(i,j,k) cyclic} sym(∇ui'⋅∇φ_r) ⊡ σ(E_nl(∇uj,∇uk)) * dΩ

Implemented by decomposing ∇u1=A+iB, ∇u2=C+iD, ∇u3=E+iF into Float64 tensors
and expanding the trilinear form over Re/Im to avoid ComplexF64 tensor allocations.
"""
function MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{3}, mult, _element, q, dΩ,
	t::FerriteGeometricNonlinearity{3})
	∇u1, ∇u2, ∇u3 = ∇W_args
	A = _re(∇u1);
	B = _im(∇u1)
	C = _re(∇u2);
	D = _im(∇u2)
	E = _re(∇u3);
	F = _im(∇u3)

	E23_re = _E_nl_f64(C, E) - _E_nl_f64(D, F);
	E23_im = _E_nl_f64(C, F) + _E_nl_f64(D, E)
	E13_re = _E_nl_f64(A, E) - _E_nl_f64(B, F);
	E13_im = _E_nl_f64(A, F) + _E_nl_f64(B, E)
	E12_re = _E_nl_f64(A, C) - _E_nl_f64(B, D);
	E12_im = _E_nl_f64(A, D) + _E_nl_f64(B, C)

	σ23_re = _σ(E23_re, t.λ, t.μ);
	σ23_im = _σ(E23_im, t.λ, t.μ)
	σ13_re = _σ(E13_re, t.λ, t.μ);
	σ13_im = _σ(E13_im, t.λ, t.μ)
	σ12_re = _σ(E12_re, t.λ, t.μ);
	σ12_im = _σ(E12_im, t.λ, t.μ)

	n_dofs = ndofs_per_cell(t.dh)
	c = mult * dΩ / 3

	for r in 1:n_dofs
		∂Nr = shape_gradient(t.cv, q, r)

		cr1_re = symmetric(transpose(A) ⋅ ∂Nr)
		cr1_im = symmetric(transpose(B) ⋅ ∂Nr)
		cr2_re = symmetric(transpose(C) ⋅ ∂Nr)
		cr2_im = symmetric(transpose(D) ⋅ ∂Nr)
		cr3_re = symmetric(transpose(E) ⋅ ∂Nr)
		cr3_im = symmetric(transpose(F) ⋅ ∂Nr)

		re = cr1_re ⊡ σ23_re - cr1_im ⊡ σ23_im +
			 cr2_re ⊡ σ13_re - cr2_im ⊡ σ13_im +
			 cr3_re ⊡ σ12_re - cr3_im ⊡ σ12_im

		im = cr1_re ⊡ σ23_im + cr1_im ⊡ σ23_re +
			 cr2_re ⊡ σ13_im + cr2_im ⊡ σ13_re +
			 cr3_re ⊡ σ12_im + cr3_im ⊡ σ12_re

		Fe[r] -= complex(c * re, c * im)
	end
end

"""
	MORFE.assemble_element!(accum, Fe, element, t)

Scatter element residual `Fe` (indexed by local DOF) into the global free-DOF
accumulator `accum` (indexed by free-DOF position).
"""
function MORFE.assemble_element!(accum, Fe, element, t::FerriteGeometricNonlinearity)
	dofs = celldofs(element)
	for (r, d) in enumerate(dofs)
		local_idx = get(t.free_to_local, d, 0)
		local_idx != 0 && (accum[local_idx] += Fe[r])
	end
end

# -----------------------------------------------------------------------
# Linear matrix assembly
# -----------------------------------------------------------------------

"""
	assemble_KM!(K, M, dh, cv, λ, μ, ρ)

Assemble the global stiffness matrix `K` and mass matrix `M` into pre-allocated
sparse matrices using standard Galerkin FEM.

	K_rs = ∫ ε(φ_r) ⊡ (λ tr(ε(φ_s)) I + 2μ ε(φ_s)) dΩ
	M_rs = ∫ ρ φ_r · φ_s dΩ
"""
function assemble_KM!(K, M, dh, cv, λ::Float64, μ::Float64, ρ::Float64)
	n_dpc = ndofs_per_cell(dh)
	Ke = zeros(n_dpc, n_dpc)
	Me = zeros(n_dpc, n_dpc)
	asm_K = start_assemble(K)
	asm_M = start_assemble(M)

	for element in CellIterator(dh)
		fill!(Ke, 0.0)
		fill!(Me, 0.0)
		reinit!(cv, element)
		for q in 1:getnquadpoints(cv)
			dΩ = getdetJdV(cv, q)
			for r in 1:n_dpc
				δε = shape_symmetric_gradient(cv, q, r)
				Nr = shape_value(cv, q, r)
				for s in 1:n_dpc
					ε = shape_symmetric_gradient(cv, q, s)
					σ = λ * tr(ε) * one(ε) + 2μ * ε
					Ke[r, s] += (δε ⊡ σ) * dΩ
					Ns = shape_value(cv, q, s)
					Me[r, s] += ρ * (Nr ⋅ Ns) * dΩ
				end
			end
		end
		assemble!(asm_K, celldofs(element), Ke)
		assemble!(asm_M, celldofs(element), Me)
	end
end
