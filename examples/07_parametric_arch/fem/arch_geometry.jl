"""
	arch_geometry.jl

Analytical sinusoidal arch map for the clamped-clamped beam:

	w(x₀) = h₀ · sin(π x₁ / L) · e₂

where x₁ is the axial coordinate along the beam span [0, L], h₀ is the arch
rise, and e₂ is the transverse unit vector.

The **parametric reference map** is:

	x(θ, x₀) = x₀ + (1 + θ) · w(x₀)

whose Jacobian decomposes as

	J(θ, x₀) = I + (1 + θ) · J_arch(x₀)
			 = J₀(x₀) + θ · J₁(x₀)

with

	J₀(x₀) = I + J_arch(x₀)     (base arch Jacobian, per QP)
	J₁(x₀) = J_arch(x₀)         (deviation Jacobian, per QP)

The gradient tensor is

	J_arch(x₀) = (π h₀ / L) · cos(π x₁ / L) · (e₂ ⊗ e₁)

a rank-1 tensor whose only non-zero entry is position (2, 1) in the standard
Cartesian basis.  Consequently det(J₀) = 1 everywhere (lower-triangular with
unit diagonal), which simplifies the adj/det series.

Scaling convention
------------------
  θ = −1 : straight beam (arch height → 0, J = I)
  θ = 0  : base arch     (height = h₀)
  θ = +1 : doubled arch  (height = 2 h₀)

h₀_L_ratio is the user-visible control knob; h₀ = h₀_L_ratio · L.

Requires Tensors.jl.  `Tens3` must be defined before this file is included
(it is defined in parametric_geometry.jl as `Tensor{2,3,Float64,9}`).
"""

using Tensors

"""
	arch_displacement(x₀, h₀, L) -> Vec{3,Float64}

Transverse displacement of the sinusoidal arch at reference position `x₀`.
Only the e₂ component is non-zero.
"""
function arch_displacement(x₀::Vec{3, Float64}, h₀::Float64, L::Float64)
	return Vec{3, Float64}((0.0, h₀ * sin(π * x₀[1] / L), 0.0))
end

"""
	arch_jacobian(x₀, h₀, L) -> Tens3

Gradient tensor of the arch displacement field:

	J_arch = ∂w/∂x₀ = (π h₀ / L) cos(π x₁ / L) · (e₂ ⊗ e₁)

Only the (2, 1) entry is non-zero.
"""
function arch_jacobian(x₀::Vec{3, Float64}, h₀::Float64, L::Float64)
	dw_dx1 = (π * h₀ / L) * cos(π * x₀[1] / L)
	return Tens3((i, j) -> (i == 2 && j == 1) ? dw_dx1 : 0.0)
end

"""
	arch_jacobian_pair(x₀, h₀, L) -> (J₀, J₁)

Return the base-arch Jacobian J₀ = I + J_arch and the perturbation Jacobian
J₁ = J_arch at reference position `x₀`.

These are the inputs to `det_and_adj_series(J₀, J₁)` from parametric_geometry.jl.
"""
function arch_jacobian_pair(x₀::Vec{3, Float64}, h₀::Float64, L::Float64)
	Ja = arch_jacobian(x₀, h₀, L)
	J₀ = one(Tens3) + Ja
	J₁ = Ja
	return J₀, J₁
end
