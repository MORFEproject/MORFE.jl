# Compare fresh R_coefficients.csv against results/reference/. Exit nonzero on mismatch.
using DelimitedFiles

let
    fresh_path = joinpath(@__DIR__, "results", "data", "R_coefficients.csv")
    ref_path   = joinpath(@__DIR__, "results", "reference", "R_coefficients.csv")

    isfile(fresh_path) || error("Fresh results not found. Run main.jl first.")

    if !isfile(ref_path)
        println("Reference not yet blessed; skipping validation.")
        println("To bless: cp results/data/R_coefficients.csv results/reference/")
    else
        fresh = Float64.(readdlm(fresh_path, ',', skipstart = 1))
        ref   = Float64.(readdlm(ref_path,   ',', skipstart = 1))
        @assert size(fresh) == size(ref) "coefficient table size changed: fresh=$(size(fresh)), ref=$(size(ref))"
        maxrel = maximum(abs.(fresh .- ref) ./ max.(abs.(ref), 1e-12))
        println("max relative deviation vs reference: $maxrel")
        maxrel < 1e-6 || error("Results deviate from reference beyond tolerance 1e-6 (got $maxrel)")
        println("Validation passed.")
    end
end
