# Shared result-writing helpers for MORFE examples. Include with:
#   include(joinpath(@__DIR__, "..", "common", "results_io.jl"))
using Serialization, Printf, LinearAlgebra

"Create and return results dirs. `run_name=nothing` → flat results/{data,figures}."
function results_dirs(example_dir; run_name = nothing)
    base = run_name === nothing ? joinpath(example_dir, "results") :
                                  joinpath(example_dir, "results", run_name)
    data = joinpath(base, "data")
    figs = joinpath(base, "figures")
    mkpath(data)
    mkpath(figs)
    return (; base, data, figs)
end

"Serialize W and R, write R coefficients as CSV matching example 04 format."
function save_rom(dirs, W, R)
    serialize(joinpath(dirs.data, "W.jls"), W)
    serialize(joinpath(dirs.data, "R.jls"), R)
    open(joinpath(dirs.data, "R_coefficients.csv"), "w") do io
        exps   = R.poly.multiindex_set.exponents
        coeffs = R.poly.coefficients        # (NVAR, L) matrix
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
end

"Write summary.txt from a vector of `\"key: value\"` strings plus environment info."
function write_summary(dirs, lines::Vector{String})
    open(joinpath(dirs.base, "summary.txt"), "w") do io
        for l in lines
            println(io, l)
        end
        println(io, "julia_version: $(VERSION)")
        commit = try
            readchomp(`git rev-parse --short HEAD`)
        catch
            "unknown"
        end
        println(io, "morfe_commit: $commit")
        println(io, "timestamp: $(time())")
    end
end
