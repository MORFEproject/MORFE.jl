(
	label = "mode_1_order_5_cnf",
	check = true, # true → pause after Phase 1 (resonance check) and ask [y/N]
	neig = 10,
	phys_modes = [1],
	max_degree = 5,
	forces = [
	# (frequency_mode = 1, shape_mode = 1, amplitude = 0.03),
	# (frequency_mode = 2, shape_mode = 4, amplitude = 20.0),
	],
	rayleigh_α = 0.0,
	rayleigh_β = 0.0,
	resonance = (
		style = :cnf,   # :cnf | :rnf | :graph
		tolerance_rel = 0.05,   # relative: resonant if |λⱼ - s| < tol_rel·|λⱼ|
	),
)
