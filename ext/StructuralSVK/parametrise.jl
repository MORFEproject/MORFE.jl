"""
	parametrise(m::AssembledMechanicalModel; master = [1], order,
				forcing = nothing, resonance_tol = 0.05, nev = ..., eigensolver = nothing)

Compute the DPIM invariant-manifold ROM. `master` lists physical mode PAIRS
(`[1]` → first conjugate pair, ROM = 2). With `forcing::HarmonicForcing`,
two external states with eigenvalues ±iΩ are appended (N_EXT = 2).
"""
function parametrise(m::AssembledMechanicalModel;
	master::Vector{Int} = [1],
	order::Int,
	forcing::Union{Nothing, HarmonicForcing} = nothing,
	resonance_tol::Real = 0.05,
	nev::Int = max(10,
		2 * max(maximum(master), forcing === nothing ? 0 : forcing.mode) + 4),
	eigensolver = nothing)
	@assert master == collect(1:length(master)) "only contiguous leading mode pairs are supported (master = [1], [1, 2], …); got master = $master"
	if forcing !== nothing
		@assert forcing.mode in master "forcing.mode = $(forcing.mode) must be a master mode pair (master = $master): the near-resonant reduction requires the forced mode on the manifold"
	end

	ROM = 2 * length(master)
	N_EXT = forcing === nothing ? 0 : 2
	NVAR = ROM + N_EXT
	mset = all_multiindices_up_to(NVAR, order; min_degree = 1)

	terms = Tuple(m.term_factory(d, length(mset)) for d in m.nonlinear_degrees)

	# ── Eigenproblem (autonomous operator; the forcing does not enter) ──────
	eig_model = NDOrderModel((m.K, m.C, m.M), terms)
	solver = eigensolver === nothing ?
			 RayleighEigenSolver(
		nothing, nothing, nev, Float64(m.damping.α), Float64(m.damping.β)) :
			 eigensolver
	t_eig = @elapsed eigenproblem =
		solver isa StructureModalDampingEigensolver ?
		solve_eigenproblem(m.K, m.M, solver; sorter! = (args...) -> nothing) :
		solve_eigenproblem(eig_model; solver = solver, sorter! = (args...) -> nothing)
	(eigenvalues, Y, X) = get_eigenpairs(eigenproblem)

	select_master_modes_by_sorting(eigenproblem, ROM)
	master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
	master_modes = Y[:, 1, 1:ROM]
	left_eigenmodes = X[:, 1:ROM]

	ORD_model = length(eig_model.linear_terms) - 1   # = 2
	FOM = m.info.n_dofs
	master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
	for r in 1:ROM, k in 1:(ORD_model-1)
		master_modes_derivatives[:, k, r] .= Y[:, k+1, r]
	end
	# Lower-order left eigenvector blocks (must be scale-consistent with the
	# physical slice above — both come from the same Eigenproblem storage).
	left_modes_derivatives = eigenproblem.left_eigenmodes_orders[:, 1:(ORD_model-1), 1:ROM]

	# ── Model with forcing (shape mode == frequency mode) ───────────────────
	Ω = nothing
	if forcing === nothing
		model = eig_model
		ext_eigs = ComplexF64[]
	else
		Ω = forcing.Ω === nothing ? abs(eigenvalues[2*forcing.mode-1]) : forcing.Ω
		fv = real((forcing.amplitude / 2) .* (m.M * Y[:, 1, 2*forcing.mode-1]))
		# Degree-1 term: multiindex (0, 0) in the state derivatives, multiplicity 1
		# in the external state ⇒ f! signature is exactly (res, r).
		force_term = MultilinearMap((res, r) -> (res .+= fv * sum(r)), (0, 0), 1)
		ext_eigs = ComplexF64[im*Ω, -im*Ω]
		model = NDOrderModel((m.K, m.C, m.M), (terms..., force_term),
			ExternalSystem(Tuple(ext_eigs)))
	end

	resonance_set = resonance_set_from_complex_normal_form_style(
		mset, Vector{ComplexF64}(master_eigenvalues), Float64(resonance_tol);
		external_eigenvalues = ext_eigs)

	conjugate_permutation = reduce(vcat, [[2p, 2p - 1] for p in 1:length(master)])
	for k in 1:(N_EXT ÷ 2)
		append!(conjugate_permutation, [ROM + 2k, ROM + 2k - 1])
	end

	t_solve = @elapsed W,
	R = solve_cohomological_problem(
		model, mset, master_eigenvalues, master_modes, left_eigenmodes, resonance_set;
		master_modes_derivatives = master_modes_derivatives,
		left_modes_derivatives = left_modes_derivatives,
		conjugate_permutation = conjugate_permutation)

	return InvariantManifoldROM(W, R, collect(eigenvalues), master, order, forcing,
		(; m.info..., n_monomials = length(mset), N_EXT = N_EXT, Ω = Ω,
			eig_time_s = t_eig, solve_time_s = t_solve))
end
