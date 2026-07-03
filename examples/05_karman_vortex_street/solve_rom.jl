"""
	solve_rom.jl — trace the ROM limit-cycle branch for each DPIM run (STEP 2).

For the NF-style ROM the limit cycle is exactly z₁ = ρ e^{iΩt}, so the branch is the
curve F(ρ, η′) = Re(R₁(ρ, ρ, η′)) = 0, traced by the PALC toolkit in solver/rom_palc.jl.

Usage:
	julia --project=. solve_rom.jl                  # all results/Re*_ord*/ with data/R.jls
	julia --project=. solve_rom.jl results/Re49.03_ord3 [more dirs...]

Writes per run dir:  data/rom_branch.csv  with columns  eta,Re,rho,omega,T
(ρ is in the 1e-2-scaled master coordinates used by W/R — downstream observables go
through W-derived polynomials in the same coordinates, so no rescaling is ever needed).
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using MORFE
using MORFE.Polynomials: evaluate
using LinearAlgebra
using StaticArrays
using Serialization
using Printf

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "solver", "rom_palc.jl"))

function trace_branch(R; re_max = BRANCH_RE_MAX, ds0 = BRANCH_DS0,
		max_steps = BRANCH_MAX_STEPS)
	η_c = rom_hopf_eta(R)
	re_c = 1 / (η_c + 1 / Re₀)
	@printf("  Hopf point: η′_c = %+.6e  (Re_c = %.4f)\n", η_c, re_c)

	# start slightly along the branch
	ρ = 1e-4
	η = η_c
	τ = rom_palc_tangent(ρ, η, R, [1.0, 0.0])
	τ[1] < 0 && (τ .*= -1.0)          # orient: ρ increasing (supercritical, Re growing)

	rows = Vector{NTuple{5, Float64}}()
	Ω0 = rom_po_frequency(ρ, η, R)
	push!(rows, (η, 1 / (η + 1 / Re₀), ρ, Ω0, 2π / abs(Ω0)))

	Δs = ds0
	for _ in 1:max_steps
		ρn, ηn, Tn, τn, n_iter, ok = rom_palc_step(ρ, η, τ, Δs, R)
		if !ok
			Δs /= 2
			Δs < 1e-12 && (@warn "PALC stalled (Δs < 1e-12)"; break)
			continue
		end
		ρ, η, τ = ρn, ηn, τn
		Ω = rom_po_frequency(ρ, η, R)
		re = 1 / (η + 1 / Re₀)
		push!(rows, (η, re, ρ, Ω, Tn))
		re > re_max && break
		# A truncated ROM can fold the branch back; once it exits the window below
		# Re_c the orbit is far outside the manifold's validity — stop there.
		re < re_c - 0.5 && break
		ρ < 1e-8 && break              # folded back to the trivial branch
		n_iter <= 4 && (Δs = min(Δs * 1.5, 100 * ds0))
		n_iter >= 10 && (Δs /= 2)
	end
	return rows
end

function process_dir(run_dir::AbstractString)
	r_path = joinpath(run_dir, "data", "R.jls")
	isfile(r_path) || (println("  skip (no data/R.jls): $run_dir"); return)
	println("── $run_dir")
	R = deserialize(r_path)
	rows = trace_branch(R)
	out_csv = joinpath(run_dir, "data", "rom_branch.csv")
	open(out_csv, "w") do io
		println(io, "eta,Re,rho,omega,T")
		for (η, re, ρ, Ω, T) in rows
			@printf(io, "%.12e,%.8f,%.12e,%.10f,%.10f\n", η, re, ρ, Ω, T)
		end
	end
	@printf("  %d branch points → %s  (Re %.3f → %.3f)\n",
		length(rows), out_csv, rows[1][2], rows[end][2])
end

run_dirs = if isempty(ARGS)
	base = joinpath(@__DIR__, "results")
	sort(filter(d -> occursin(r"^Re[\d.]+_ord\d+$", basename(d)) && isdir(d),
		readdir(base; join = true)))
else
	[abspath(a) for a in ARGS]
end
isempty(run_dirs) && error("no run directories found — run main.jl first")
foreach(process_dir, run_dirs)
