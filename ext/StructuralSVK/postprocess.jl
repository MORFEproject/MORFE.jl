"""
	real_dynamics(rom) -> DensePolynomial

Realified master equation ż₁ in the real pair z₁ = x₁ + i y₁:
Re(c) is the ẋ₁-equation, Im(c) the ẏ₁-equation (the imaginary parts carry
the frequency content; do not discard them). External (forcing) states also
come in conjugate ±iΩ pairs, so the same pairwise conjugation map applies.
"""
function real_dynamics(rom::InvariantManifoldROM)
	NVAR = 2 * length(rom.master) + rom.info.N_EXT
	conj_map = [isodd(i) ? i + 1 : i - 1 for i in 1:NVAR]
	return realify(extract_component(rom.R.poly, 1), conj_map)
end

"""
	print_equations(rom; tol = 1e-12, io = stdout)

Print the realified ż₁ monomial table. In the forced case the trailing two
exponents belong to the external states (e^{+iΩt}, e^{-iΩt}).
"""
function print_equations(rom::InvariantManifoldROM; tol = 1e-12, io = stdout)
	Rr = real_dynamics(rom)
	header = rom.info.N_EXT == 0 ?
			 "(x₁,y₁) exponents : ẋ₁-coeff, ẏ₁-coeff" :
			 "(x₁,y₁,ext₊,ext₋) exponents : ẋ₁-coeff, ẏ₁-coeff"
	println(io, "Reduced dynamics ż₁ = ẋ₁ + i·ẏ₁ in real variables:")
	println(io, "  " * header)
	for (m, mi) in enumerate(Rr.multiindex_set.exponents)
		c = Rr.coefficients[m]
		abs(c) > tol && println(io,
			"  $(Tuple(mi)) : " *
			"$(round(real(c); sigdigits = 6)), $(round(imag(c); sigdigits = 6))")
	end
end

"""
	save_rom(rom, dir)

Write `dir/data/{W.jls, R.jls, R_coefficients.csv}` and `dir/summary.txt`.
The CSV format is identical to `examples/common/results_io.jl` (the
equivalence gates diff the two files).
"""
function save_rom(rom::InvariantManifoldROM, dir::AbstractString)
	data = joinpath(dir, "data")
	mkpath(data)
	mkpath(joinpath(dir, "figures"))
	serialize(joinpath(data, "W.jls"), rom.W)
	serialize(joinpath(data, "R.jls"), rom.R)
	open(joinpath(data, "R_coefficients.csv"), "w") do io
		exps = rom.R.poly.multiindex_set.exponents
		coeffs = rom.R.poly.coefficients        # (NVAR, L) matrix
		NVAR_R = size(coeffs, 1)
		header = join(["exp_$i" for i in 1:length(exps[1])], ",") * "," *
				 join(["R$(i)_re,R$(i)_im" for i in 1:NVAR_R], ",")
		println(io, header)
		for (m, ex) in enumerate(exps)
			c = coeffs[:, m]
			any(abs.(c) .> 1e-14) || continue
			row = join(string.(Int.(ex)), ",") * "," *
				  join(["$(real(c[i])),$(imag(c[i]))" for i in 1:NVAR_R], ",")
			println(io, row)
		end
	end
	open(joinpath(dir, "summary.txt"), "w") do io
		println(io, "model: SVK + Ferrite (MORFEStructuralSVK)")
		println(io, "n_dofs: $(rom.info.n_dofs)")
		println(io, "master_pairs: $(rom.master)")
		println(io, "master_eigenvalues: $(rom.eigenvalues[1:(2 * length(rom.master))])")
		println(io, "parametrisation_order: $(rom.order)")
		println(io, "n_monomials: $(rom.info.n_monomials)")
		if rom.forcing === nothing
			println(io, "forcing: none")
		else
			println(io,
				"forcing: mode=$(rom.forcing.mode) amplitude=$(rom.forcing.amplitude) Omega=$(rom.info.Ω)")
		end
		println(io, "eigenproblem_time_s: $(rom.info.eig_time_s)")
		println(io, "cohomological_solve_time_s: $(rom.info.solve_time_s)")
		println(io, "julia_version: $(VERSION)")
		commit = try
			readchomp(`git rev-parse --short HEAD`)
		catch
			"unknown"
		end
		println(io, "morfe_commit: $commit")
		println(io, "timestamp: $(time())")
	end
	return nothing
end
