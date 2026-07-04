"""
	solve_rom.jl — trace the ROM limit-cycle branches for a DPIM run (STEP 2).

For the NF-style ROM the limit cycle is exactly z₁ = ρ e^{iΩt}, so the branch is the
curve F(ρ, η′) = Re(R₁(ρ, ρ, η′)) = 0, traced by the PALC toolkit in solver/rom_palc.jl.

The cohomological solve is graded, so the order-N reduced dynamics is the EXACT
truncation of the MAX_ORD run (verified bit-exact). One branch is therefore traced per
truncation order in TRUNC_ORDERS, each with R truncated to degree ≤ N.

Usage:
	julia --project=. solve_rom.jl                  # all results/Re*_ord*/ with data/R.jls
	julia --project=. solve_rom.jl results/Re49.03_ord9 [more dirs...]

Writes per run dir:  data/rom_branch_ord<N>.csv  with columns  eta,Re,rho,omega,T
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

"""
	truncate_dynamics(R, N) -> ReducedDynamics

Zero all coefficients of monomials with total degree > N. Because the cohomological
solve is graded, this IS the order-N reduced dynamics (bit-exact).
"""
function truncate_dynamics(R, N::Int)
	Rt = deepcopy(R)
	exps = Rt.poly.multiindex_set.exponents
	for (m, e) in enumerate(exps)
		sum(e) > N && (Rt.poly.coefficients[:, m] .= 0)
	end
	return Rt
end

function process_dir(run_dir::AbstractString)
	r_path = joinpath(run_dir, "data", "R.jls")
	isfile(r_path) || (println("  skip (no data/R.jls): $run_dir"); return)
	println("── $run_dir")
	R = deserialize(r_path)
	max_deg = maximum(sum, R.poly.multiindex_set.exponents)
	for N in TRUNC_ORDERS
		N > max_deg && (println("  skip order $N (> run order $max_deg)"); continue)
		println("  ── truncation order $N")
		rows = trace_branch(truncate_dynamics(R, N))
		out_csv = joinpath(run_dir, "data", "rom_branch_ord$(N).csv")
		open(out_csv, "w") do io
			println(io, "eta,Re,rho,omega,T")
			for (η, re, ρ, Ω, T) in rows
				@printf(io, "%.12e,%.8f,%.12e,%.10f,%.10f\n", η, re, ρ, Ω, T)
			end
		end
		@printf("  %d branch points → %s  (Re %.3f → %.3f)\n",
			length(rows), out_csv, rows[1][2], rows[end][2])
	end
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
