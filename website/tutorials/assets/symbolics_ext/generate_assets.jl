# generate_assets.jl — the embedded figures for the "Symbolic full-order model" tutorial.
#
# Every driver drawn here is built by `externalsystem_from_symbolics` from the same
# expressions the tutorial page shows, then integrated through `evaluate` on the resulting
# object — so the curves come out of the real library code, never from a formula written a
# second time.
#
# Rendering reuses the site's dependency-free renderer,
# `examples/internals/full_order_model/viz.jl`: one standalone HTML file per figure, no CDN,
# no external stylesheet or script, so each opens from the file system and drops into an
# <iframe>.  The example notebook plots with Plots for its own `results/figures/`; these two
# paths are deliberately separate.
#
# Run from the repository root:
#
#     julia --project=website website/tutorials/assets/symbolics_ext/generate_assets.jl
#
# Writes into this directory by default; set MORFE_SYM_OUT to redirect.

using MORFE, Symbolics
using StaticArrays: SVector
using MORFE.Polynomials: evaluate

include(joinpath(@__DIR__, "..", "..", "..", "..",
    "examples", "internals", "full_order_model", "viz.jl"))

const OUT = get(ENV, "MORFE_SYM_OUT", @__DIR__)

# One RK4 step of ẏ = f(y) — the same ten lines the full-order-model example uses, kept here
# so this script needs no plotting or ODE dependency.
function rk4_step(f, y, h)
    k1 = f(y)
    k2 = f(y .+ (h / 2) .* k1)
    k3 = f(y .+ (h / 2) .* k2)
    k4 = f(y .+ h .* k3)
    return y .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

function orbit(f, y0, h, nsteps; project = identity)
    ys = Vector{typeof(y0)}(undef, nsteps + 1)
    ys[1] = project(y0)
    for i in 1:nsteps
        ys[i + 1] = project(rk4_step(f, ys[i], h))
    end
    return ys
end

# An ExternalSystem carries its dynamics as a polynomial, so this is all integrating one takes.
external_rhs(sys) = r -> evaluate(sys.first_order_dynamics, r)
component(ys, k) = [y[k] for y in ys]
thin(n, m) = n <= m ? (1:n) : range(1, n; length = m) .|> round .|> Int

# Physical initial state → the system's own (possibly re-based) coordinates.
function reduced_initial(sys, r0, ::Val{R}) where {R}
    Q = external_basis(sys)
    v = Q === nothing ? ComplexF64.(r0) : ComplexF64.(Q \ ComplexF64.(r0))
    return SVector{R, ComplexF64}(v)
end

const Ω = 1.3

# ── the drivers, built symbolically ───────────────────────────────────────────────────────

@variables rh[1:2]
rh = collect(rh)
harmonic = externalsystem_from_symbolics([im * Ω * rh[1], -im * Ω * rh[2]], rh)

@variables rq[1:4]
rq = collect(rq)
const Ω1, Ω2 = 1.0, sqrt(2)
quasi = externalsystem_from_symbolics(
    [im * Ω1 * rq[1], -im * Ω1 * rq[2], im * Ω2 * rq[3], -im * Ω2 * rq[4]], rq)

@variables rm[1:4]
rm = collect(rm)
multiharmonic = externalsystem_from_symbolics(
    [Ω * rm[3],
        -0.03 * Ω * rm[1] + 4 * Ω * rm[4],
        -Ω * rm[1],
        -4 * Ω * rm[2] + 0.12 * Ω * rm[3]], rm)

const σ, ρ, β = 10.0, 28.0, 8 / 3
const Cp = sqrt(β * (ρ - 1))
@variables X Y Z
lorenz = externalsystem_from_symbolics(
    [σ * (Y - X),
        X - Y - Z * (X + Cp),
        Cp * (X + Y) + X * Y - β * Z], [X, Y, Z])

# ── orbits ────────────────────────────────────────────────────────────────────────────────

h_h = 2π / Ω / 400
n_h = 2 * 400                                   # two forcing periods
o_h = orbit(external_rhs(harmonic), reduced_initial(harmonic, [1.0, 1.0], Val(2)), h_h, n_h)
t_h = h_h .* (0:n_h)
p_h = [to_physical_external(harmonic, v) for v in o_h]
i_h = thin(length(o_h), 900)
err_h = maximum(abs(real(v[1] + v[2]) - 2cos(Ω * t)) for (v, t) in zip(p_h, t_h))
println("harmonic       max |r₁+r₂ − 2cos Ωt| = ", round(err_h, sigdigits = 3))

h_q = 0.02
n_q = 8000
o_q = orbit(external_rhs(quasi),
    reduced_initial(quasi, [1.0, 1.0, 0.7, 0.7], Val(4)), h_q, n_q)
t_q = h_q .* (0:n_q)
q_re = real.(component(o_q, 1)) .+ real.(component(o_q, 3))
q_im = imag.(component(o_q, 1)) .+ imag.(component(o_q, 3))
i_q = thin(length(o_q), 2000)

h_m = 2π / Ω / 2000
n_m = 2 * 2000
o_m = orbit(external_rhs(multiharmonic),
    reduced_initial(multiharmonic, [-1.0, -1.0, 0.0, 0.0], Val(4)), h_m, n_m)
t_m = h_m .* (0:n_m)
p_m = [to_physical_external(multiharmonic, v) for v in o_m]
m_r1 = real.(component(p_m, 1))
m_r2 = real.(component(p_m, 2))
i_m = thin(length(o_m), 2000)
err_m = maximum(abs(m_r2[j] - (0.03sin(Ω * t_m[j]) - cos(4Ω * t_m[j])))
for j in eachindex(t_m))
println("multiharmonic  max |r₂ − (0.03 sin Ωt − cos 4Ωt)| = ", round(err_m, sigdigits = 3))

# Lorenz is stored in complex coordinates while the physical flow is real. One eigenvalue is
# real and the other two are a conjugate pair, and round-off leaves that subspace — the
# unstable directions then amplify the drift off the attractor. Which slot is which is read
# from the eigenvalues rather than assumed: re-basing keeps a pair adjacent but does not
# promise where it lands.
const λL = lorenz.eigenvalues
const L_REAL = findfirst(λ -> abs(imag(λ)) < 1e-8 * max(1, abs(λ)), λL)
const L_POS = findfirst(k -> k != L_REAL && imag(λL[k]) > 0, eachindex(λL))
const L_NEG = findfirst(k -> k != L_REAL && k != L_POS, eachindex(λL))
@assert λL[L_NEG]≈conj(λL[L_POS]) "slots $L_POS/$L_NEG are not a conjugate pair"

realify(v) = SVector{3, ComplexF64}(ntuple(
    k -> k == L_REAL ? real(v[L_REAL]) + 0im : k == L_POS ? v[L_POS] : conj(v[L_POS]), 3))

h_L = 1e-4
n_L = 400_000                                   # 40 s on the attractor
v0 = realify(reduced_initial(lorenz, [1.0, 1.0, 20.0] .- [Cp, Cp, ρ - 1], Val(3)))
o_L = orbit(external_rhs(lorenz), v0, h_L, n_L; project = realify)
t_L = h_L .* (0:n_L)
# read back physically, then undo the shift to the equilibrium C₊
p_L = [real.(to_physical_external(lorenz, v)) .+ [Cp, Cp, ρ - 1] for v in o_L]
L_x = component(p_L, 1)
L_z = component(p_L, 3)
i_L = thin(length(p_L), 4000)
println("lorenz         |x| ≤ ", round(maximum(abs, L_x), sigdigits = 4),
    "  (Lorenz stays within |x| ≲ 18)")

# ── figure 1: the drivers ─────────────────────────────────────────────────────────────────

write_pairs(joinpath(OUT, "fig1_drivers.html"),
    [
        PairPanel("harmonic",
            [Curve("Re r₁(t)", t_h[i_h], real.(component(p_h, 1))[i_h]; colour = 1),
                Curve("Im r₁(t)", t_h[i_h], imag.(component(p_h, 1))[i_h]; colour = 4)],
            [Curve("r₁", real.(component(p_h, 1))[i_h],
                imag.(component(p_h, 1))[i_h]; colour = 1)];
            tylabel = "r₁", pxlabel = "Re r₁", pylabel = "Im r₁",
            note = "ext_exprs = [im*Ω*r[1], -im*Ω*r[2]] with Ω = $(Ω). A purely imaginary " *
                   "pair: the orbit is a circle and r₁ + r₂ = 2cos Ωt."),
        PairPanel("quasi-periodic",
            [Curve("Re r₁ + Re r₃", t_q[i_q], q_re[i_q]; colour = 2)],
            [Curve("(Re, Im) of r₁ + r₃", q_re[i_q], q_im[i_q]; colour = 2)];
            tylabel = "signal", pxlabel = "Re", pylabel = "Im",
            note = "Two incommensurate frequencies, Ω₁ = $(Ω1) and Ω₂ = √2, over " *
                   "$(round(Int, h_q * n_q)) s: the signal beats and never repeats, and the " *
                   "orbit fills an annulus — a section of the invariant torus."),
        PairPanel("multiharmonic",
            [Curve("r₁(t) = −cos Ωt", t_m[i_m], m_r1[i_m]; colour = 3),
                Curve("r₂(t) = 0.03 sin Ωt − cos 4Ωt", t_m[i_m], m_r2[i_m]; colour = 4)],
            [Curve("(r₁, r₂)", m_r1[i_m], m_r2[i_m]; colour = 3)];
            tylabel = "r", pxlabel = "r₁", pylabel = "r₂",
            note = "Not diagonal, and not upper triangular either — " *
                   "externalsystem_from_symbolics re-bases it and says so. Read back with " *
                   "to_physical_external, the two harmonics Ω and 4Ω draw an M."),
        PairPanel("chaotic (Lorenz)",
            [Curve("X(t)", t_L[i_L], L_x[i_L]; colour = 1),
                Curve("Z(t)", t_L[i_L], L_z[i_L]; colour = 3)],
            [Curve("(X, Z)", L_x[i_L], L_z[i_L]; colour = 1)];
            tylabel = "state", pxlabel = "X", pylabel = "Z", equal_aspect = false,
            note = "A nonlinear driver, shifted to the equilibrium C₊ so the origin is one. " *
                   "Complex coordinates carry a reality condition, enforced after each step.")];
    title = "External systems, built symbolically",
    caption = "Time plot left, phase portrait right. Every driver is built by " *
              "`externalsystem_from_symbolics` and integrated through " *
              "`evaluate(sys.first_order_dynamics, r)` — no right-hand side is written twice.")

# ── figure 2: a damped oscillator driven by each ──────────────────────────────────────────
#
# ÿ + c ẏ + k y = forcing(r), integrated together with the driver as one state [y, ẏ, r…].

const cc, kk = 0.2, 1.0

function forced_response(sys, r0, forcing, h, nsteps, ::Val{R}; project = identity) where {R}
    rhs = function (s)
        r = SVector{R, ComplexF64}(ntuple(j -> s[2 + j], R))
        dr = evaluate(sys.first_order_dynamics, r)
        f = forcing(to_physical_external(sys, r))
        return SVector{2 + R, ComplexF64}(s[2], -cc * s[2] - kk * s[1] + f, dr...)
    end
    proj = function (s)
        r = project(SVector{R, ComplexF64}(ntuple(j -> s[2 + j], R)))
        return SVector{2 + R, ComplexF64}(real(s[1]) + 0im, real(s[2]) + 0im, r...)
    end
    s0 = SVector{2 + R, ComplexF64}(0, 0, reduced_initial(sys, r0, Val(R))...)
    ys = orbit(rhs, s0, h, nsteps; project = proj)
    t = h .* (0:nsteps)
    y = real.(component(ys, 1))
    F = [forcing(to_physical_external(sys, SVector{R, ComplexF64}(ntuple(j -> s[2 + j], R))))
         for s in ys]
    return t, y, real.(F)
end

t1, y1, f1 = forced_response(
    harmonic, [1.0, 1.0], p -> real(p[1] + p[2]), 0.01, 4000, Val(2))
t2, y2, f2 = forced_response(
    multiharmonic, [-1.0, -1.0, 0.0, 0.0], p -> real(p[2]), 0.002, 20000, Val(4))
t3, y3, f3 = forced_response(lorenz, [1.0, 1.0, 20.0] .- [Cp, Cp, ρ - 1],
    p -> 0.05 * real(p[1] + Cp), 1e-4, 400_000, Val(3); project = realify)

j1, j2, j3 = thin(length(t1), 1500), thin(length(t2), 2000), thin(length(t3), 3000)

write_charts(joinpath(OUT, "fig2_forced_response.html"),
    [
        ChartPanel("harmonic",
            [Curve("y(t)", t1[j1], y1[j1]; colour = 1),
                Curve("forcing(t)", t1[j1], f1[j1]; colour = 2, dashed = true)];
            xlabel = "t", ylabel = "y",
            note = "forcing(r) = r₁ + r₂ = 2cos Ωt. After the transient the response is a " *
                   "cosine at the driving frequency, lagging the force."),
        ChartPanel("multiharmonic",
            [Curve("y(t)", t2[j2], y2[j2]; colour = 1),
                Curve("forcing(t)", t2[j2], f2[j2]; colour = 3, dashed = true)];
            xlabel = "t", ylabel = "y",
            note = "forcing(r) = r₂ = 0.03 sin Ωt − cos 4Ωt. The oscillator is a low-pass " *
                   "filter: it follows the slow harmonic and barely registers 4Ω."),
        ChartPanel("chaotic (Lorenz)",
            [Curve("y(t)", t3[j3], y3[j3]; colour = 1),
                Curve("forcing(t)", t3[j3], f3[j3]; colour = 4, dashed = true)];
            xlabel = "t", ylabel = "y",
            note = "forcing(r) = 0.05 x(t) from the Lorenz attractor. A bounded but " *
                   "non-repeating drive, and a response that never settles.")];
    title = "One oscillator, three drivers",
    caption = "ÿ + cẏ + ky = forcing(r) with c = $(cc), k = $(kk), integrated together " *
              "with ṙ = E(r) as a single state. The forcing reads the physical external " *
              "state through `to_physical_external`.")

println("\nwrote:")
for f in ("fig1_drivers.html", "fig2_forced_response.html")
    println("  ", joinpath(OUT, f), "  (", filesize(joinpath(OUT, f)), " bytes)")
end
