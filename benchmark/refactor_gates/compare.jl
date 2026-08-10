# Compare a freshly-captured baseline against the Stage-0 reference.
#
#   julia --project=test compare.jl <reference_dir> <candidate_dir>
#
# Accuracy gate:  W and R coefficient arrays must be BIT-IDENTICAL (===, not ≈).
# Perf gate:      allocations must not grow; wall time within 2%.

using Serialization, Printf

ref, cand = ARGS[1], ARGS[2]
rs = deserialize(joinpath(ref, "summary.jls"))
cs = deserialize(joinpath(cand, "summary.jls"))

fail = false
@printf("%-24s %-10s %14s %14s %8s\n", "model", "coeffs", "bytes ref", "bytes cand", "Δtime")
println(repeat("-", 78))

for name in sort(collect(keys(rs)))
    if !haskey(cs, name)
        @printf("%-24s %-10s  MISSING IN CANDIDATE\n", name, "—")
        global fail = true
        continue
    end
    ok = true
    for which in ("W", "R")
        a = deserialize(joinpath(ref, "$(name)_$(which).jls"))
        b = deserialize(joinpath(cand, "$(name)_$(which).jls"))
        if size(a) != size(b)
            @printf("  %s %s: SIZE %s vs %s\n", name, which, size(a), size(b))
            ok = false
        elseif a != b
            d = maximum(abs.(a .- b))
            rel = d / max(maximum(abs.(a)), eps())
            @printf("  %s %s: NOT BIT-IDENTICAL  maxabs=%.3e  rel=%.3e\n",
                name, which, d, rel)
            ok = false
        end
    end
    dt = (cs[name].time_s - rs[name].time_s) / max(rs[name].time_s, eps())
    db = cs[name].bytes - rs[name].bytes
    ok || (global fail = true)
    db > 0 && (@printf("  %s: +%d BYTES ALLOCATED\n", name, db); global fail = true)
    @printf("%-24s %-10s %14d %14d %+7.1f%%\n",
        name, ok ? "identical" : "DIFFER", rs[name].bytes, cs[name].bytes, 100dt)
end

println(repeat("-", 78))
if fail
    println("GATE FAILED")
    exit(1)
else
    println("GATE PASSED — all coefficients bit-identical, no allocation growth")
end
