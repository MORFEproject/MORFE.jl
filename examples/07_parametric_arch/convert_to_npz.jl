"""
	convert_to_npz.jl

Converts all W.jls / R.jls pairs under results/ into rom.npz files
readable directly by numpy.

Usage:
  julia convert_to_npz.jl                   # converts all found pairs
  julia convert_to_npz.jl results/data/arch_h0.100   # single directory
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.add("NPZ"; io = devnull)
Pkg.instantiate(; io = devnull)

using Serialization
using NPZ
using MORFE

function convert_dir(dir::AbstractString)
	w_path = joinpath(dir, "W.jls")
	r_path = joinpath(dir, "R.jls")
	isfile(w_path) && isfile(r_path) || return false

	W = deserialize(w_path)
	R = deserialize(r_path)

	exps = W.poly.multiindex_set.exponents
	npzwrite(joinpath(dir, "rom.npz"), Dict(
		"W"         => W.poly.coefficients,                         # ComplexF64 (FOM, ORD, L)
		"R"         => R.poly.coefficients,                         # ComplexF64 (NVAR, L)
		"exponents" => Int32.(hcat([collect(e) for e in exps]...)), # Int32 (NVAR, L)
	))
	return true
end

# Collect target directories: CLI args or scan results/
if !isempty(ARGS)
	dirs = ARGS
else
	results_root = joinpath(@__DIR__, "results")
	dirs = String[]
	for (root, _, files) in walkdir(results_root)
		if "W.jls" in files && "R.jls" in files
			push!(dirs, root)
		end
	end
end

for dir in dirs
	print("  $dir … ")
	if Base.invokelatest(convert_dir, dir)
		println("done → rom.npz")
	else
		println("skipped (W.jls or R.jls missing)")
	end
end

println("\nConversion complete.")
