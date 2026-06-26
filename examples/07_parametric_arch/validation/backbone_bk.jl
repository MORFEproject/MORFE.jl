"""
    backbone_bk.jl

Numerical backbone curves for the parametric arch ROM via BifurcationKit PALC.

For each arch height ratio in `H_RATIOS` and each θ-truncation order the backbone
is traced as the zero set of

    G(r, Ω; θ) = Ω - Im[R₁(r, 0, θ)] / r = 0

using BifurcationKit pseudo-arc-length continuation (PALC).

Why not collocation?  For the DPIM conservative reduced dynamics ż = iΩ(|z|²)z the
periodic orbits are exactly circular.  The collocation bordered system is rank-1
deficient at every orbit (both Floquet multipliers are 1 for a 2D area-preserving
map), so Newton cannot distinguish amplitudes and collapses to r = 0.  The G = 0
algebraic formulation correctly encodes the backbone and admits a unique solution
for each Ω (or r), while still using genuine PALC numerics via BifurcationKit.

Outputs (in results/backbone_bk/):
  backbones.csv  — long-format (h_ratio, theta, model, theta_order, r, omega, amplitude)
                   amplitude = NaN (physical mapping via W_param omitted here)
  metrics.csv    — ω₀ per (h_ratio, model, theta_order)
"""

# -----------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------
using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, "backbone_bk_env"))
if !haskey(Pkg.project().dependencies, "MORFE")
    Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "..")))
end
Pkg.instantiate()

using BifurcationKit
using MORFE
using MORFE.Polynomials: DensePolynomial, evaluate, extract_component, each_term, similar_poly
using MORFE.Realification: realify
using Serialization, Printf
using StaticArrays: SVector

# -----------------------------------------------------------------------
# Constants  (must match main.jl and backbone.jl)
# -----------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "config.jl"))   # h0_L_ratio, N_INCREMENTS

const h₀_L_ratio = h0_L_ratio
const H_RATIOS   = collect(range(0.0, 2 * h0_L_ratio; length = N_INCREMENTS + 1))
const R_MAX      = 60.0    # modal amplitude sweep upper bound (same as backbone.jl)
const N_R        = 500     # number of backbone points (consistent with backbone.jl)

const _param_dir = joinpath(@__DIR__, "..", "results", "data",
    @sprintf("arch_h%.3f", h0_L_ratio))
const _out_dir = joinpath(@__DIR__, "..", "results", "backbone_bk")

isfile(joinpath(_param_dir, "R.jls")) ||
    error("Parametric ROM not found at $_param_dir.  Run main.jl first.")

# -----------------------------------------------------------------------
# Polynomial helpers (identical to backbone.jl)
# -----------------------------------------------------------------------

function poly_deriv(p::DensePolynomial{T, NVAR, 1}, var_idx::Int) where {T, NVAR}
    dict = Dict{SVector{NVAR, Int}, T}()
    for (α, c) in each_term(p)
        n = α[var_idx]
        iszero(n) && continue
        new_α = SVector{NVAR, Int}(ntuple(j -> j == var_idx ? α[j] - 1 : α[j], Val(NVAR)))
        dict[new_α] = get(dict, new_α, zero(T)) + T(n) * c
    end
    return similar_poly(dict)
end

function truncate_theta_order(poly::DensePolynomial, k_max::Int)
    coeffs = copy(poly.coefficients)
    for (l, α) in enumerate(collect(poly.multiindex_set))
        if α[3] > k_max
            selectdim(coeffs, ndims(coeffs), l) .= 0
        end
    end
    return DensePolynomial(coeffs, poly.multiindex_set)
end

# -----------------------------------------------------------------------
# Load parametric ROM
# -----------------------------------------------------------------------

println("Loading parametric ROM …")
R_param  = deserialize(joinpath(_param_dir, "R.jls"))

# R1_cplx(x, y, θ) : real inputs, complex output
# conservative backbone: Ω(r, θ) = Im[R1_cplx(r, 0, θ)] / r
R1_cplx = extract_component(realify(R_param.poly, [2, 1, 3]), 1)

θ_max_order  = maximum(α[3] for α in collect(R1_cplx.multiindex_set))
theta_orders = max(0, θ_max_order - 6):θ_max_order
@printf "  θ_max_order = %d   truncation orders: %s\n" θ_max_order string(collect(theta_orders))

# -----------------------------------------------------------------------
# BifurcationKit continuation for one (R1_trunc, θ_val)
# -----------------------------------------------------------------------

"""
    run_bk_backbone(R1_trunc, dR1_∂x, θ_val, ω₀)

Trace the backbone of the conservative 2-D ROM via PALC on G(r, Ω) = 0 where

    G(r, Ω) = Ω - Im[R₁(r, 0, θ)] / r

The Jacobian dG/dr is computed analytically from the precomputed derivative
polynomial `dR1_∂x = ∂R₁/∂x`.

Returns the BK branch (or `nothing` on failure).
"""
function run_bk_backbone(R1_trunc, dR1_∂x, θ_val::Float64, ω₀::Float64)
    r_seed = R_MAX / N_R   # first non-trivial amplitude on the backbone
    Ω_seed = ω₀            # starting frequency ≈ linear eigenfrequency

    # G(x=[r], p=[Ω]) = Ω - Im[R1(r,0,θ)] / r = 0
    function G(x, p)
        r  = max(x[1], 1e-12)
        f  = imag(evaluate(R1_trunc, [r, 0.0, θ_val]))
        return [f / r - p[1]]
    end

    # dG/dr = (Im[∂R1/∂x](r,0,θ) * r - Im[R1](r,0,θ)) / r²
    function J_G(x, _)
        r  = max(x[1], 1e-10)
        f  = imag(evaluate(R1_trunc, [r, 0.0, θ_val]))
        df = imag(evaluate(dR1_∂x,   [r, 0.0, θ_val]))
        return reshape([(df * r - f) / r^2], 1, 1)
    end

    prob = BifurcationProblem(G, [r_seed], [Ω_seed], @optic(_[1]);
        J                   = J_G,
        record_from_solution = (x, p; k...) -> (r = x[1], Ω = p),
    )

    opts = ContinuationPar(
        p_min     = ω₀ * 0.4,
        p_max     = ω₀ * 2.0,
        ds        = 1e-2 * ω₀,
        dsmax     = 0.2  * ω₀,
        max_steps = 2000,
        detect_bifurcation = 0,
    )

    return try
        continuation(prob, PALC(), opts; verbosity = 0)
    catch e
        @warn "BK continuation failed (θ=$θ_val, θ-order=??): $e"
        nothing
    end
end

# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------

mkpath(_out_dir)
curves = NamedTuple[]

println()
for h_ratio in H_RATIOS
    θ_fixed = h_ratio / h₀_L_ratio - 1.0
    println("── h₀/L = $h_ratio  (θ = $θ_fixed) ──")

    for theta_order in theta_orders
        R1_trunc  = truncate_theta_order(R1_cplx, theta_order)
        dR1_∂x    = poly_deriv(R1_trunc, 1)   # ∂R1/∂x  (used in J_G)

        ω₀ = imag(evaluate(dR1_∂x, [0.0, 0.0, θ_fixed]))
        @printf "  [θ-order %d]  ω₀ = %.6f  (T₀ = %.6f)\n" theta_order ω₀ 2π / ω₀

        br = run_bk_backbone(R1_trunc, dR1_∂x, θ_fixed, ω₀)
        isnothing(br) && continue

        r_vec = [s.r   for s in br.branch]
        Ω_vec = [s.Ω   for s in br.branch]
        @printf "    ✓ %d branch points  r ∈ [%.3g, %.3g]  Ω ∈ [%.4f, %.4f]\n" length(br.branch) extrema(r_vec)... extrema(Ω_vec)...

        push!(curves, (;
            h_ratio,
            θ_fixed,
            model       = "bk_palc",
            theta_order,
            r           = r_vec,
            Ω           = Ω_vec,
            ω₀,
        ))
    end
end

# -----------------------------------------------------------------------
# backbones.csv  — same column layout as backbone.jl for direct overlay
# -----------------------------------------------------------------------

open(joinpath(_out_dir, "backbones.csv"), "w") do io
    println(io, "h_ratio,theta,model,theta_order,r,omega,amplitude")
    for c in curves
        for k in eachindex(c.r)
            @printf io "%.6f,%.4f,%s,%d,%.8f,%.8f,NaN\n" c.h_ratio c.θ_fixed c.model c.theta_order c.r[k] c.Ω[k]
        end
    end
end
println("Saved → $(joinpath(_out_dir, "backbones.csv"))")

# -----------------------------------------------------------------------
# metrics.csv
# -----------------------------------------------------------------------

open(joinpath(_out_dir, "metrics.csv"), "w") do io
    println(io, "h_ratio,theta,model,theta_order,omega0,delta_omega0_rel,modal_proj")
    for c in curves
        @printf io "%.6f,%.4f,%s,%d,%.8f,NaN,NaN\n" c.h_ratio c.θ_fixed c.model c.theta_order c.ω₀
    end
end
println("Saved → $(joinpath(_out_dir, "metrics.csv"))")

println("\nDone. BK backbone data in $_out_dir/")
