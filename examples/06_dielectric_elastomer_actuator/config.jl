"""
parameters.jl — nondimensional parameter set for the DEA cantilever example.

All quantities are nondimensional (length scale L, time scale 1/ω₁-ish, charge scale Q₀).
See `physics_description.md` and `implementation_plan_detailed.md` §1 for the model.
"""

"""
	dea_parameters(; kwargs...) -> NamedTuple

Default nondimensional parameters. Knobs:
- `ωτ`   : ω₁·RC ratio. `1.0` puts the electrical pole at the mechanical frequency —
		   the regime that motivates the third-order model.
- `ξ1`   : damping ratio of mode 1 (sets the Kelvin–Voigt modulus via D = η_over_E·K).
- `V0`   : DC bias voltage (keep well below pull-in; checked in `bias_checks`).
- `v_a`  : AC drive amplitude.
- `m_b`  : taper amplitude of the actuation couple m_b(x) = m_b·(1 − x/L) per unit Q².
- `α_c`  : capacitance strain sensitivity, gᵀx = 2α_c·w_tip/L².
"""
function dea_parameters(;
	n_elem    = 50,        # FE elements → FOM n = 2*n_elem free DOFs
	L         = 1.0,       # beam length
	EI        = 1.0,       # bending stiffness
	ρA        = 1.0,       # mass per unit length  → ω₁ ≈ 3.516
	ξ1        = 0.005,     # target damping ratio of mode 1
	ω1_target = 3.516015087, # analytic cantilever ω₁ = (1.8751041)²·√(EI/(ρA L⁴))
	R         = 1.0,       # electrical resistance
	ωτ        = 1.0,       # ≈ ω₁·R/ĉ  → sets c₀
	α_c       = 1.0,       # capacitance strain sensitivity
	m_b       = 1.0,       # actuation couple taper amplitude per unit Q²
	V0        = 1.0,       # DC bias voltage (→ Q₀ ≈ 0.29, bias strain ≈ 0.021; pull-in ≈ 2.7)
	v_a       = 0.02,      # harmonic voltage amplitude
)
	η_over_E = 2 * ξ1 / ω1_target      # Kelvin–Voigt: D = η_over_E * K
	c0 = ω1_target * R / ωτ            # inverse rest capacitance
	return (; n_elem, L, EI, ρA, η_over_E, R, c0, α_c, m_b, V0, v_a, ω1_target)
end
