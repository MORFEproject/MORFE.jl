"""
Building a full-order model

Every DPIM computation starts from the same three objects, and this tutorial builds all
three from scratch:

- `MultilinearMap` — one nonlinear term of the equation of motion. Its `multiindex` says
  which derivative fills each argument slot of `f!`.
- `ExternalSystem` — the autonomous dynamics `ṙ = f(r)` that drives the model. Its linear
  part **must be upper triangular** — a constraint the constructor enforces by changing
  coordinates when it has to.
- `NDOrderModel` — the assembled `B_ORD x^(ORD) + … + B_0 x = F(x, ẋ, …, r)`.

The arc is: nonlinear terms → the driver → the assembled model → its response.

 1. `MultilinearMap`: what a multiindex means, drawn as force–displacement curves
 2. `ExternalSystem`: harmonic, quasi-periodic and nonlinear drivers, drawn as orbits
 3. Lorenz: the coordinate change that makes a non-triangular driver usable
 4. `NDOrderModel`: assemble a forced Duffing oscillator and integrate it

Everything is evaluated through the real library code — the curves come from `evaluate_term!` and
`evaluate`, never from a formula written out a second time — and the two coordinate
changes are checked numerically rather than asserted.

Each section writes an interactive figure to `results/figures/`. Nothing is solved here:
the script runs in a couple of seconds and needs no FEM backend.

Run it from the repository root:

	julia --project -e 'include("examples/internals/full_order_model/main.jl")'
"""

using MORFE
using LinearAlgebra
using StaticArrays: SVector
using MORFE.Multiindices: all_multiindices_up_to
using MORFE.MultilinearMaps: evaluate_term!, _call_signature
using MORFE.Polynomials: DensePolynomial, evaluate

include(joinpath(@__DIR__, "viz.jl"))

const FOM_FIGDIR = get(ENV, "MORFE_FOM_OUT", joinpath(@__DIR__, "results", "figures"))

rule(title) = println("\n" * "="^92 * "\n  " * title * "\n" * "="^92)

# One RK4 step of ẏ = f(y). Ten lines, no dependency, and accurate enough that the
# coordinate changes below can be checked to ~1e-13.
function rk4_step(f, y, h)
    k1 = f(y)
    k2 = f(y .+ (h / 2) .* k1)
    k3 = f(y .+ (h / 2) .* k2)
    k4 = f(y .+ h .* k3)
    return y .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

"""
	orbit(f, y0, h, nsteps; project = identity) -> Vector

Integrate `ẏ = f(y)` from `y0` and return every state along the way.

`project` is applied after each step. It is the identity for every system here except the
modal Lorenz one, where it enforces a reality condition — see section 3.
"""
function orbit(f, y0, h, nsteps; project = identity)
    ys = Vector{typeof(y0)}(undef, nsteps + 1)
    ys[1] = project(y0)
    for i in 1:nsteps
        ys[i + 1] = project(rk4_step(f, ys[i], h))
    end
    return ys
end

# An `ExternalSystem` carries its dynamics as a polynomial, so this is all it takes to
# integrate one.
external_rhs(sys) = r -> evaluate(sys.first_order_dynamics, r)
function orbit(sys::ExternalSystem, r0, h, nsteps; kwargs...)
    orbit(external_rhs(sys), r0, h, nsteps; kwargs...)
end

component(ys, k) = [y[k] for y in ys]

# An external system's polynomial written from its non-zero terms alone: the pair
# `(i, exponent) => c` says the monomial r^exponent contributes c to ṙᵢ.  Everything else
# is zero, which spelling the coefficient vectors out in full does not make clearer — most
# of a driver's coefficients are zero, and only the handful that are not carry the physics.
function external_polynomial(nvar, degree, terms::Pair...)
    ms = all_multiindices_up_to(nvar, degree)
    deleteat!(ms.exponents, 1)              # an ExternalSystem carries no constant term
    coeffs = [zeros(ComplexF64, nvar) for _ in ms.exponents]
    for ((i, exponent), c) in terms
        k = findfirst(e -> Tuple(e) == exponent, ms.exponents)
        k === nothing && throw(ArgumentError("exponent $exponent is not in the set"))
        coeffs[k][i] += c
    end
    return DensePolynomial([SVector{nvar, ComplexF64}(c) for c in coeffs], ms)
end

# Orbits are integrated finely for accuracy but drawn coarsely: a screen cannot resolve
# 24 000 points, and shipping them would make the figure a megabyte instead of a page.
thin(len::Integer, n::Integer = 1500) = 1:cld(len, n):len

# ------------------------------------------------------------------------------
# 1. `MultilinearMap` — what a multiindex means
#
# `multiindex[k]` counts how many argument slots of `f!` take the derivative `x^(k-1)`.
# For a second-order system `multiindex = (3, 0)` means `f!(res, x, x, x)` and
# `(1, 1)` means `f!(res, x, ẋ)`. The terms below are all built for ORD = 2 and
# evaluated with `evaluate_term!`, so every curve comes out of the real object.
# ------------------------------------------------------------------------------
rule("1. MultilinearMap — the multiindex is the calling convention")

const K_LIN = 4.0     # linear stiffness
const K2 = 3.0        # quadratic coefficient
const K3 = 6.0        # cubic coefficient
const C_DRAG = 0.8    # drag coefficient
const Ω = 1.3         # forcing frequency; sections 2 and 4 drive the model with it too

# The equation of motion is written  M ẍ + C ẋ + K x = F(x, ẋ, r),  so a *hardening*
# spring — restoring force growing faster than linearly — contributes F = −k₃x³.
quadratic!(res, x1, x2) = (res .+= -K2 .* x1 .* x2)
hardening!(res, x1, x2, x3) = (res .+= -K3 .* x1 .* x2 .* x3)
softening!(res, x1, x2, x3) = (res .+= +K3 .* x1 .* x2 .* x3)
drag!(res, v1, v2) = (res .+= -C_DRAG .* v1 .* v2)
# A term that mixes the state with the external state: F = k·x·(r₁+r₂). `f!` receives the
# whole external vector in its last slot, and r₂ = r̄₁, so r₁ + r₂ = 2cos(Ωt) is already
# real along the physical orbit.  Taking `real` here instead would break multilinearity:
# Re is not complex-linear — it is not even complex-differentiable — and the whole method
# rests on each slot being linear.
const K_MIX = -1.5
mixed!(res, x, r) = (res .+= K_MIX .* x .* (r[1] + r[2]))

term_quad = MultilinearMap(quadratic!; multiindex = (2, 0),
    multiplicity_external = 0, fully_asymmetric = false)
term_hard = MultilinearMap(hardening!; multiindex = (3, 0),
    multiplicity_external = 0, fully_asymmetric = false)
term_soft = MultilinearMap(softening!; multiindex = (3, 0),
    multiplicity_external = 0, fully_asymmetric = false)
term_drag = MultilinearMap(drag!; multiindex = (0, 2),
    multiplicity_external = 0, fully_asymmetric = false)
# multiindex (1, 0) with one external factor: one slot takes x, the other takes r. This
# split is exactly the one the constructor refuses to guess — state it and it is fine.
term_mixed = MultilinearMap(mixed!; multiindex = (1, 0), multiplicity_external = 1)

for (name, t) in (("quadratic", term_quad), ("hardening cubic", term_hard),
    ("quadratic drag", term_drag), ("mixed x·r", term_mixed))
    println(rpad(name, 18), "multiindex = ", rpad(string(t.multiindex), 8),
        " deg = ", t.deg, "   ",
        _call_signature(t.multiindex, t.multiplicity_external))
end

# The keyword constructor reports anything it had to assume. Stating `multiindex` and
# `multiplicity_external` — as above — keeps it silent; here is what it says when they
# are left out and the shape has to be inferred from the arity of `f!`:
println("\nLeaving the shape to be inferred:")
_ = MultilinearMap(hardening!)

# Evaluate a term the way the solver does: one state vector per derivative order.
force(term, x, v) = (res = zeros(1);
    evaluate_term!(res, term, ([x], [v]), nothing);
    res[1])

xs = collect(range(-1.0, 1.0, length = 241))
vs = collect(range(-1.5, 1.5, length = 241))

# Total restoring force, i.e. the linear part *minus* the nonlinear contribution F,
# which is what a quasi-static pull test would measure.
restoring(term, x) = K_LIN * x - force(term, x, 0.0)

panel_force = ChartPanel("restoring force",
    [Curve("linear  K u", xs, K_LIN .* xs; colour = 4, dashed = true),
        Curve("hardening  K u + k₃ u³", xs, restoring.(Ref(term_hard), xs); colour = 1),
        Curve("softening  K u − k₃ u³", xs, restoring.(Ref(term_soft), xs); colour = 3),
        Curve("quadratic  K u + k₂ u²", xs, restoring.(Ref(term_quad), xs); colour = 2)];
    xlabel = "displacement u", ylabel = "restoring force",
    note = "A multiindex (3, 0) is for cubic f!(res, u, u, u); and (2, 0) is for quadratic f!(res, u, u). " *
           "Hardening stiffens with amplitude, softening goes the other way. K = 4, k₂ = 3, k₃ = 6.")

# A *multilinear* quadratic drag is v·v, which is even — it decelerates for v > 0 and
# accelerates for v < 0. Physical drag is |v|·v, which is odd but not multilinear, so it
# cannot be a `MultilinearMap` at all: it would have to be approximated by odd terms.
panel_drag = ChartPanel("quadratic drag",
    [
        Curve("smooth  −0.8 u̇²", vs, [force(term_drag, 0.0, v) for v in vs];
            colour = 1),
        Curve("not smooth  −0.8 |u̇| u̇", vs,
            -C_DRAG .* abs.(vs) .* vs; colour = 3, dashed = true)];
    xlabel = "velocity u̇", ylabel = "force F",
    note = "The multilinear map F(u̇₁,u̇₂) = u̇₁ u̇₂ is linear in each argument separately; and describes a quadratic function on repeated inputs: F(u̇,u̇) = u̇². " *
           "F(u̇,u̇) = |u̇| u̇ is not twice differentiable @ u̇ = 0.")

# F(x, t) = k·x·(r₁+r₂) with r = (e^{iΩt}, e^{−iΩt}), so r₁ + r₂ = 2cos(Ωt). It is a
# surface over the (x, t) plane, drawn as a wireframe: straight lines along x — the term is
# linear in the state — and cosines along t.
function force_ext(term, x, t)
    (res = zeros(1);
        evaluate_term!(res, term, ([x], [0.0]), SVector(cis(Ω * t), cis(-Ω * t)));
        res[1])
end

x_mix = collect(range(-1.0, 1.0, length = 33))
t_mix = collect(range(0, 2 * 2π / Ω, length = 65))       # two forcing periods
surf_mixed = Surface3D(x_mix, t_mix, (x, t) -> force_ext(term_mixed, x, t))

# The swept line is the law at one instant, F = 2k·cos(Ωt)·u, which is a straight line
# through the origin pivoting with the phase. It needs no grid: two endpoints and the
# closed form are exact. `offset` lifts it by ~1% of the force range, just enough to sit
# on the surface rather than inside it.
line_mixed = SweptLine([first(x_mix), last(x_mix)],
    [2K_MIX * first(x_mix), 2K_MIX * last(x_mix)];
    omega = Ω, offset = 0.01 * (maximum(surf_mixed.z) - minimum(surf_mixed.z)))

panel_mixed = ChartPanel("time-varying stiffness", surf_mixed;
    line = line_mixed,
    axes = ("displacement u", "time t", "force F"),
    note = "Periodically time-varying stiffness: F = k · u · (r₁ + r₂), with " *
           "r₁ + r₂ = 2 cos(Ωt). The swept line is the force–displacement law at one " *
           "instant. Drag to orbit.")

write_charts(joinpath(FOM_FIGDIR, "fig1_nonlinear_terms.html"),
    [panel_force, panel_drag, panel_mixed];
    title = "Nonlinear terms",
    caption = "Every curve is produced by `evaluate_term!` on a real `MultilinearMap`.")

# ------------------------------------------------------------------------------
# 2. `ExternalSystem` — the driver
#
# The external state r satisfies its own autonomous dynamics ṙ = f(r), and enters the
# model's nonlinear terms as an extra argument. The canonical MORFE forcing is a pair of
# conjugate eigenvalues ±iΩ, whose orbit r₁(t) = e^{iΩt} is a unit circle: harmonic
# forcing without ever writing cos(Ωt).
# ------------------------------------------------------------------------------
rule("2. ExternalSystem — harmonic, quasi-periodic, nonlinear")

harmonic = ExternalSystem((im * Ω, -im * Ω))
println("harmonic       eigenvalues = ", harmonic.eigenvalues)

h_step = 2π / Ω / 400
harm = orbit(harmonic, ComplexF64[1.0, 1.0], h_step, 1200)
t_harm = h_step .* (0:1200)
ih = thin(length(harm), 900)

# The diagonal case has the closed form r(t) = exp(λt) r₀, so RK4 can be checked rather
# than trusted.  `Base.exp` is spelled out because the examples smoke test includes every
# internals demo into one scope, and one of them binds a global named `exp`.
exact = [Base.exp(harmonic.eigenvalues[1] * t) for t in t_harm]
err_harm = maximum(abs.(component(harm, 1) .- exact))
println("RK4 vs closed form exp(λt)r₀: max error = ", round(err_harm, sigdigits = 3))

# Two incommensurate frequencies: the orbit never closes, and the envelope beats.
const Ω1, Ω2 = 1.0, sqrt(2)
quasi = ExternalSystem((im * Ω1, -im * Ω1, im * Ω2, -im * Ω2))
println("quasi-periodic eigenvalues = ", quasi.eigenvalues)
hq = 0.02
qorb = orbit(quasi, ComplexF64[1.0, 1.0, 0.7, 0.7], hq, 4000)
t_q = hq .* (0:4000)
q_sig = real.(component(qorb, 1)) .+ real.(component(qorb, 3))
q_im = imag.(component(qorb, 1)) .+ imag.(component(qorb, 3))
iq = thin(length(qorb), 2000)

# A *nonlinear* external system. Only the linear part must be upper triangular — the
# nonlinear part is unconstrained — so a one-way coupling is allowed:
#
#     ṙ₁ = iΩ₁ r₁ + i c r₁r₃        ṙ₃ = iΩ₂ r₃
#     ṙ₂ = −iΩ₁ r₂ − i c r₂r₄       ṙ₄ = −iΩ₂ r₄        (r₂ = r̄₁, r₄ = r̄₃)
#
# The slow pair (r₃, r₄) modulates the fast one and never the reverse — that asymmetry is
# the triangular structure, made visible. Integrating the r₃ term gives
# r₁ = exp(iΩ₁t + (c/Ω₂)(e^{iΩ₂t} − 1)), so |r₁| swings between exp(−2c/Ω₂) and 1: the
# modulation is *sustained*, not a transient that decays onto the linear system's circle.
const Ω_SLOW, C_MOD = 0.4, 0.25
cascade = ExternalSystem(external_polynomial(4, 2,
    (1, (1, 0, 0, 0)) => im * Ω,          # ṙ₁ ← iΩ₁ r₁
    (2, (0, 1, 0, 0)) => -im * Ω,         # ṙ₂ ← −iΩ₁ r₂
    (3, (0, 0, 1, 0)) => im * Ω_SLOW,     # ṙ₃ ← iΩ₂ r₃
    (4, (0, 0, 0, 1)) => -im * Ω_SLOW,    # ṙ₄ ← −iΩ₂ r₄
    (1, (1, 0, 1, 0)) => im * C_MOD,      # ṙ₁ ← i c r₁r₃
    (2, (0, 1, 0, 1)) => -im * C_MOD))    # ṙ₂ ← −i c r₂r₄
println("cascade        eigenvalues = ", cascade.eigenvalues)
println("               diagonal ⇒ upper triangular ⇒ accepted, nonlinear part free")
hc = 0.01
corb = orbit(cascade, ComplexF64[1.0, 1.0, 1.0, 1.0], hc, 12000)
t_c = hc .* (0:12000)
ic = thin(length(corb), 2400)
amp_c = abs.(component(corb, 1))
println("               |r₁| ∈ [", round(minimum(amp_c), digits = 3), ", ",
    round(maximum(amp_c), digits = 3), "]   predicted [",
    round(Base.exp(-2C_MOD / Ω_SLOW), digits = 3), ", 1.0]")

write_pairs(joinpath(FOM_FIGDIR, "fig2_external_systems.html"),
    [
        PairPanel("harmonic",
            [Curve("Re r₁(t)", t_harm[ih], real.(component(harm, 1))[ih]; colour = 1),
                Curve("Im r₁(t)", t_harm[ih], imag.(component(harm, 1))[ih]; colour = 2)],
            [Curve("r₁", real.(component(harm, 1))[ih],
                imag.(component(harm, 1))[ih]; colour = 1)];
            tylabel = "r₁", pxlabel = "Re r₁", pylabel = "Im r₁",
            note = "ExternalSystem((iΩ, −iΩ)) with Ω = $(Ω): a purely imaginary pair of " *
                   "eigenvalues, so the orbit is a circle and the signal a cosine. " *
                   "RK4 matches exp(λt)r₀ to $(round(err_harm, sigdigits = 2))."),
        PairPanel("quasi-periodic",
            [Curve("Re r₁ + Re r₃", t_q[iq], q_sig[iq]; colour = 2)],
            [Curve("(Re, Im) of r₁ + r₃", q_sig[iq], q_im[iq]; colour = 2)];
            tylabel = "signal", pxlabel = "Re", pylabel = "Im",
            note = "Two incommensurate frequencies, Ω₁ = $(Ω1) and Ω₂ = √2: the signal " *
                   "beats and never repeats, and the orbit fills an annulus — a section " *
                   "of the invariant torus."),
        PairPanel("nonlinear cascade",
            [Curve("Re r₁(t)", t_c[ic], real.(component(corb, 1))[ic]; colour = 1),
                Curve("|r₁(t)|", t_c[ic], amp_c[ic]; colour = 3)],
            [Curve("r₁", real.(component(corb, 1))[ic],
                imag.(component(corb, 1))[ic]; colour = 1)];
            tylabel = "r₁", pxlabel = "Re r₁", pylabel = "Im r₁",
            note = "ṙ₁ = iΩ₁r₁ + i c r₁r₃ and ṙ₃ = iΩ₂r₃ with r₂ = r̄₁ and r₄ = r̄₃: " *
                   "the slow pair (r₃, r₄) modulates the fast one (r₁, r₂).")];
    title = "External systems",
    caption = "Time plot left, phase portrait right. Orbits integrated with RK4 through " *
              "`evaluate(sys.first_order_dynamics, r)`.")

# ------------------------------------------------------------------------------
# 3. Lorenz — re-based automatically by a change of coordinates
#
# The linear part of an external system must be upper triangular. The reason is
# causality: the cohomological equations are solved monomial by monomial in GrLex order,
# and the |β| = 1 lower-order coupling needs W[α − eⱼ + eᵢ], a coefficient of the *same*
# degree, which precedes α only when i < j. So the solver reads only the strictly upper
# triangle of Λ, and a strictly-lower entry would be dropped without trace.
#
# That requirement is a property of the *coordinates*, not of the system, so a matrix that
# fails it is repaired rather than rejected: `ExternalSystem` finds a basis Q in which the
# linear part is triangular and re-expresses the whole polynomial in r′ = Q⁻¹r. Below, the
# automatic result is checked against the diagonalisation done by hand.
# ------------------------------------------------------------------------------
rule("3. Lorenz — re-based automatically")

const SIG, RHO, BET = 10.0, 28.0, 8 / 3
lorenz(r) = [SIG * (r[2] - r[1]), r[1] * (RHO - r[3]) - r[2], r[1] * r[2] - BET * r[3]]

# The quadratic part of Lorenz, as a symmetric bilinear form evaluated on one argument.
lorenz_quad(u) = [0.0 + 0im, -u[1] * u[3], u[1] * u[2]]

function lorenz_polynomial(A, quad_coeff)
    ms = all_multiindices_up_to(3, 2)
    deleteat!(ms.exponents, 1)
    coeffs = map(ms.exponents) do e
        ex = Tuple(e)
        if sum(ex) == 1
            k = findfirst(==(1), ex)
            return SVector{3, ComplexF64}(A[:, k])
        end
        slots = vcat((fill(i, ex[i]) for i in 1:3)...)
        return quad_coeff(slots[1], slots[end])
    end
    return DensePolynomial(coeffs, ms)
end

# --- 3a. the raw system, at the origin ---------------------------------------
A_origin = [-SIG SIG 0.0; RHO -1.0 0.0; 0.0 0.0 -BET]
function raw_quad(a, b)
    a == b ? SVector{3, ComplexF64}(lorenz_quad(1.0 .* (1:3 .== a))) :
    SVector{3, ComplexF64}(lorenz_quad(1.0 .* ((1:3 .== a) .+ (1:3 .== b))) .-
                           lorenz_quad(1.0 .* (1:3 .== a)) .-
                           lorenz_quad(1.0 .* (1:3 .== b)))
end
println("Lorenz linearised at the origin — linear matrix:")
println(repr("text/plain", A_origin))
println("\nNo reordering of the variables would help: both off-diagonal entries of the x–y")
println("block are non-zero, so one of them is always strictly below the diagonal. A change")
println("of basis is the only repair, and it is applied automatically:\n")

origin_ext = ExternalSystem(lorenz_polynomial(ComplexF64.(A_origin), raw_quad))
Q_origin = external_basis(origin_ext)
println("accepted; basis Q stored     = ", Q_origin !== nothing)
println("linear part now triangular   = ", istriu(origin_ext.linear_matrix))
println("eigenvalues (diag U)         = ", round.(origin_ext.eigenvalues, digits = 4))
println("A_origin is real ⇒ eigenvector route ⇒ U is diagonal: ",
    isdiag(round.(Matrix(origin_ext.linear_matrix), digits = 12)))

# --- 3b. shift to a non-trivial fixed point -----------------------------------
# The two non-trivial fixed points lost stability through a Hopf bifurcation at
# ρ_H = σ(σ+β+3)/(σ−β−1); past it there is no stable fixed point and no stable periodic
# orbit, which is what leaves the chaotic attractor. Stability is not what we need here:
# the fixed point is needed only to remove the constant term, since an ExternalSystem's
# polynomial has none.
rho_hopf = SIG * (SIG + BET + 3) / (SIG - BET - 1)
xe = sqrt(BET * (RHO - 1))
C_plus = [xe, xe, RHO - 1]
println(
    "\nHopf threshold ρ_H = ", round(rho_hopf, digits = 3), "   (running at ρ = ", RHO, ")")
println("fixed point C₊     = ", round.(C_plus, digits = 4),
    "   ‖f(C₊)‖ = ", round(norm(lorenz(C_plus)), sigdigits = 3))

# Because Lorenz is quadratic, u = r − C₊ gives exactly u̇ = J u + Q(u, u): the same
# quadratic part, and no constant because C₊ is an equilibrium. Writing
# u = (X, Y, Z) and C = √(β(ρ−1)), the centred system is
#
#     Ẋ = σ(Y − X)
#     Ẏ = X − Y − Z(X + C)     =  X − Y − C·Z  −  X·Z
#     Ż = C(X + Y) + X·Y − β·Z
#
# so the linear part is the Jacobian below and the quadratic part is (0, −X·Z, +X·Y),
# unchanged from the original system. J is still not triangular — the (2,1) entry is now
# 1 rather than ρ, but it is not zero — so a shift alone is not enough.
J = [-SIG SIG 0.0; RHO-C_plus[3] -1.0 -C_plus[1]; C_plus[2] C_plus[1] -BET]

# --- 3c. diagonalise by hand, then cross-check the automatic basis -------------
# This is the transformation `ExternalSystem` now performs internally. Doing it by hand
# once is what lets us check the automatic result against something independent.
F = eigen(J)
Λ, T = F.values, F.vectors
Ti = inv(T)
println("eigenvalues at C₊  = ", round.(Λ, digits = 4))
println("→ diagonal, hence upper triangular by construction.")

# v = T⁻¹u  ⇒  v̇ = Λ v + T⁻¹ Q(T v, T v).  The quadratic coefficients follow from the
# polarisation identity B(a, b) = Q(a+b) − Q(a) − Q(b).
function modal_quad(a, b)
    a == b ? SVector{3, ComplexF64}(Ti * lorenz_quad(T[:, a])) :
    SVector{3, ComplexF64}(Ti * (lorenz_quad(T[:, a] .+ T[:, b]) .-
                            lorenz_quad(T[:, a]) .- lorenz_quad(T[:, b])))
end
modal_poly = lorenz_polynomial(ComplexF64.(Diagonal(Λ)), modal_quad)
lorenz_ext = ExternalSystem(modal_poly)
println("ExternalSystem accepted; eigenvalues = ", round.(lorenz_ext.eigenvalues, digits = 4))
println("already diagonal ⇒ nothing to re-base: basis === nothing is ",
    external_basis(lorenz_ext) === nothing)

# Feed the *same* centred system in its raw, non-triangular coordinates and let the
# constructor find its own basis. It must describe the same dynamics as `modal_poly`.
raw_centred = lorenz_polynomial(ComplexF64.(J), raw_quad)
auto_ext = ExternalSystem(raw_centred)
Q_auto = external_basis(auto_ext)

# --- 3d. check the coordinate change, do not assert it ------------------------
rhs_err = 0.0
auto_err = 0.0
for _ in 1:200
    u = 3.0 .* randn(3)
    uc = SVector{3, ComplexF64}(u)
    back = T * evaluate(modal_poly, SVector{3, ComplexF64}(Ti * ComplexF64.(u)))
    global rhs_err = max(rhs_err, norm(real.(back) .- lorenz(C_plus .+ u)))
    # Same test for the automatic basis: Q·f_auto(Q⁻¹u) must reproduce the centred field.
    auto_back = Q_auto * evaluate(auto_ext.first_order_dynamics,
        SVector{3, ComplexF64}(Q_auto \ uc))
    global auto_err = max(auto_err, norm(auto_back .- evaluate(raw_centred, uc)))
end
println("\nmax ‖T·f_modal(T⁻¹u) − f_lorenz(C₊+u)‖ over 200 points = ",
    round(rhs_err, sigdigits = 3))
println("max ‖Q·f_auto(Q⁻¹u) − f_centred(u)‖   over 200 points = ",
    round(auto_err, sigdigits = 3))
println("→ the automatic re-basing reproduces the hand-derived change of coordinates.")

const hL = 1e-4
# Start inside the butterfly loops, at r₀ = (1, 1, 20) in physical coordinates: the orbit
# is on the attractor from the first step, so there is no transient to discard. Starting
# beside C₊ instead would be a poor choice — it is an unstable focus with Re λ = 0.094, so
# the orbit would spend tens of seconds spiralling slowly outwards and the picture would
# show that spiral rather than the attractor.
r0 = [1.0, 1.0, 20.0]
u0 = r0 .- C_plus                 # same point, measured from the fixed point
v0 = SVector{3, ComplexF64}(Ti * ComplexF64.(u0))

# The modal coordinates are complex, but a *physical* state is not an arbitrary point of
# ℂ³: λ₁ is real, so v₁ is real, and λ₃ = conj(λ₂), so v₃ = conj(v₂). Those reality
# conditions cut out a real 3-dimensional subspace that the exact flow preserves —
# but round-off does not, and here the unstable directions amplify the drift until the
# orbit leaves the attractor altogether (measured: ‖v₃ − conj(v₂)‖ reaches 10² by t ≈ 45 s
# without this, and x wanders to −79, far outside Lorenz's |x| ≲ 18).  Projecting back
# after every step costs nothing and is the same reality condition MORFE's `Realification`
# module imposes on a parametrisation.
realify(v) = SVector{3, ComplexF64}(real(v[1]) + 0im, v[2], conj(v[2]))

# Agreement is checked over a short window on purpose. The attractor is chaotic — its
# largest Lyapunov exponent is ≈ 0.9/s — so two integrations of the same orbit separate
# like e^{0.9 t}: by t = 12 s round-off has already been amplified by ~5·10⁴, and the
# comparison would be measuring the Lyapunov exponent rather than the change of
# coordinates. Five seconds keeps it a statement about the coordinates.
n_val = 5000
vorb_val = orbit(lorenz_ext, v0, hL, n_val; project = realify)
rorb_val = orbit(lorenz, r0, hL, n_val)
orbit_err = maximum(norm(real.(T * v) .+ C_plus .- r)
for (v, r) in zip(vorb_val, rorb_val))
println("max ‖modal orbit mapped back − direct Lorenz orbit‖ over ",
    round(hL * n_val, digits = 1), " s = ", round(orbit_err, sigdigits = 3))

n_rec = 300000
vorb = orbit(lorenz_ext, v0, hL, n_rec; project = realify)
mapped = [real.(T * v) .+ C_plus for v in vorb]
xs_rec = component(mapped, 1)
println("recorded ", round(hL * n_rec, digits = 1), " s on the attractor; x ∈ [",
    round(minimum(xs_rec), digits = 1), ", ", round(maximum(xs_rec), digits = 1),
    "], wings swapped ", count(i -> xs_rec[i] * xs_rec[i + 1] < 0, 1:(length(xs_rec) - 1)),
    " times")

t_L = hL .* (0:n_rec)
iL = thin(length(mapped), 4000)
iLt = thin(length(mapped), 1800)

# The modal coordinates are complex, but not independently so: v₁ belongs to the real
# eigenvalue and stays real, while v₃ = conj(v₂). Three real numbers therefore carry the
# whole state, and (Re v₁, Re v₂, Im v₂) is the phase space the ExternalSystem really
# moves in.
mv1 = real.(component(vorb, 1))
mv2r = real.(component(vorb, 2))
mv2i = imag.(component(vorb, 2))

write_split(joinpath(FOM_FIGDIR, "fig3_lorenz.html"),
    [
        SplitPanel("physical coordinates",
            # One curve, not two: over 90 s of a chaotic orbit a second integration would
            # trace the same attractor but a visibly different trajectory. The agreement
            # between the two is measured above, on a window short enough to mean something.
            [Orbit3D("modal, mapped back through T", component(mapped, 1)[iL],
                component(mapped, 2)[iL], component(mapped, 3)[iL]; colour = 1)],
            [Curve("x(t)", t_L[iLt], component(mapped, 1)[iLt]; colour = 1),
                Curve("y(t)", t_L[iLt], component(mapped, 2)[iLt]; colour = 2),
                Curve("z(t)", t_L[iLt], component(mapped, 3)[iLt]; colour = 3)];
            axes = ("x", "y", "z"), xlabel = "t", ylabel = "state",
            note = "The butterfly, drawn twice: integrated as an ExternalSystem in modal " *
                   "coordinates then mapped back through T, over a direct integration of " *
                   "Lorenz. Aperiodic switching between the two wings."),
        SplitPanel("modal coordinates",
            [Orbit3D("(Re v₁, Re v₂, Im v₂)", mv1[iL], mv2r[iL], mv2i[iL]; colour = 2)],
            [Curve("Re v₁(t)", t_L[iLt], mv1[iLt]; colour = 1),
                Curve("Re v₂(t)", t_L[iLt], mv2r[iLt]; colour = 2),
                Curve("Im v₂(t)", t_L[iLt], mv2i[iLt]; colour = 3)];
            axes = ("Re v₁", "Re v₂", "Im v₂"), xlabel = "t", ylabel = "modal state",
            note = "What the ExternalSystem actually integrates. v₁ follows the fast real " *
                   "eigenvalue −13.85; v₂ and its conjugate spiral out of the unstable " *
                   "focus left behind by the Hopf bifurcation.")];
    title = "Lorenz as an external system",
    caption = "Shift to a fixed point, diagonalise the Jacobian, and the full nonlinear " *
              "system becomes admissible — agreement with direct integration ≈ " *
              "$(round(orbit_err, sigdigits = 2)).")

println("\nNote: this driver's linear part is unstable (Re λ = ",
    round(maximum(real, Λ), digits = 3), "), unusual for a forcing term. ",
    "\nThe section is about the construction, not about a recommended forcing.")

# ------------------------------------------------------------------------------
# 4. `NDOrderModel` — assemble and drive it
#
# A forced Duffing oscillator:  M ẍ + C ẋ + K x = −k₃x³ + f·(r₁ + r₂),  with r driven by
# the harmonic system from section 2. One degree of freedom keeps the figure readable;
# nothing changes for a FEM-sized model beyond the size of the matrices.
# ------------------------------------------------------------------------------
rule("4. NDOrderModel — assemble the model and drive it")

const M_MAT = fill(1.0, 1, 1)
const C_MAT = fill(0.08, 1, 1)
const K_MAT = fill(K_LIN, 1, 1)
const F_AMP = 2.5

forcing!(res, r) = (res .+= F_AMP * (r[1] + r[2]))
term_forcing = MultilinearMap(forcing!; multiindex = (0, 0), multiplicity_external = 1)

model = NDOrderModel((K_MAT, C_MAT, M_MAT), (term_hard, term_forcing), harmonic)
println("model       = ", typeof(model).name.name, "{ORD=2}, n_fom = ", model.n_fom,
    ", terms = ", length(model.nonlinear_terms), ", max_nl_degree = ", model.max_nl_degree)

A_fo, B_fo = linear_first_order_matrices(model)
println("first-order companion pair from `linear_first_order_matrices`:")
println("  A = ", A_fo[1, :], " / ", A_fo[2, :])
println("  B = ", B_fo[1, :], " / ", B_fo[2, :])

# A term that reads the external state needs a model that has one — otherwise the
# mismatch would only surface much later, during evaluation.
try
    NDOrderModel((K_MAT, C_MAT, M_MAT), (term_forcing,))
catch e
    println("\nWithout an external system:\n", e.msg)
end

# Right-hand side of the coupled system (x, ẋ, r), assembled through the model itself.
function nl_force(model, x, v, r)
    (res = zeros(model.n_fom);
        for ord in 1:(model.max_nl_degree)
            evaluate_nonlinear_terms!(res, model, ord, ([x], [v]), r)
        end;
        res)
end

function duffing_rhs(y)
    x, v = y[1], y[2]
    r = SVector(y[3], y[4])
    F = nl_force(model, x, v, r)[1]
    acc = (F - K_MAT[1, 1] * x - C_MAT[1, 1] * v) / M_MAT[1, 1]
    dr = evaluate(harmonic.first_order_dynamics, r)
    return [v, acc, dr[1], dr[2]]
end

const STEPS_PER_PERIOD = 400
hD = 2π / Ω / STEPS_PER_PERIOD
nD = 25 * STEPS_PER_PERIOD                   # 25 forcing periods ≈ 120.8 s
dorb = orbit(duffing_rhs, ComplexF64[0.0, 0.0, 1.0, 1.0], hD, nD)
t_D = hD .* (0:nD)
xD = real.(component(dorb, 1))
vD = real.(component(dorb, 2))

# The steady state is the final forcing period, drawn as 1.01 of one so the curve laps
# itself slightly and the loop visibly closes rather than leaving a hairline gap. The
# transient is everything before it, and the two share their boundary point so the time
# trace reads as one continuous curve.
n_steady = round(Int, 1.01 * STEPS_PER_PERIOD)
steady = (length(xD) - n_steady):length(xD)
transient = 1:steady[1]
itr = transient[thin(length(transient), 1600)]
ist = steady

# A closed loop is the claim the phase portrait makes, so measure it: how far is the state
# from where it was exactly one period earlier, relative to the size of the loop?
closure = hypot(xD[end] - xD[end - STEPS_PER_PERIOD],
    vD[end] - vD[end - STEPS_PER_PERIOD])
loop = hypot(maximum(xD[steady]) - minimum(xD[steady]),
    maximum(vD[steady]) - minimum(vD[steady]))
println("\nintegrated ", round(hD * nD, digits = 1), " s = ", nD ÷ STEPS_PER_PERIOD,
    " forcing periods;  |x|max = ", round(maximum(abs, xD), digits = 4),
    ",  steady |x|max = ", round(maximum(abs, xD[steady]), digits = 4))
println("loop closure after one period: ", round(100 * closure / loop, sigdigits = 2),
    " % of the loop diameter")

write_charts(joinpath(FOM_FIGDIR, "fig4_forced_response.html"),
    [
        # One quantity, x(t), in two colours: purple while the transient is still dying,
        # red once it has. The same two colours carry over to the phase portrait, so a
        # feature can be traced from one panel to the other.
        ChartPanel("response — time",
            [Curve("transient", t_D[itr], xD[itr]; colour = 1),
                Curve("steady state — last period", t_D[ist], xD[ist]; colour = 3)];
            xlabel = "t", ylabel = "x",
            note = "Forced Duffing: M ẍ + C ẋ + K x = −k₃x³ + f·(r₁+r₂), with r from " *
                   "the harmonic ExternalSystem of section 2."),
        # No equal_aspect here: x and ẋ are different physical quantities whose ranges
        # differ by a factor of a few, so forcing one scale on both would letterbox the
        # orbit into a sliver. Scaling each axis to its own data fills the panel, and the
        # statement below survives it — a non-uniform scaling of an ellipse is still one.
        ChartPanel("phase portrait",
            [Curve("transient", xD[itr], vD[itr]; colour = 1),
                # The closed loop is the point of this panel, so draw it twice as heavy as
                # the spiral that leads into it.
                Curve("steady state — last period", xD[ist], vD[ist];
                    colour = 3, width = 2)];
            xlabel = "x", ylabel = "ẋ",
            note = "Damping pulls the orbit onto the periodic response, a closed loop; " *
                   "the hardening cubic is what bends it away from an ellipse.")];
    title = "Forced response of the assembled model",
    caption = "The nonlinear force at each step comes from " *
              "`evaluate_nonlinear_terms!` on the `NDOrderModel` itself.")

# The website card for this tutorial uses the phase portrait itself rather than a drawing
# of one, so it is generated from the same arrays as the figure above.
ith = itr[thin(length(itr), 420)]        # a card needs far fewer points than a figure
write_thumbnail(joinpath(FOM_FIGDIR, "thumb.svg"),
    [Curve("transient", xD[ith], vD[ith]; colour = 1, width = 0.55),
        Curve("steady state", xD[ist], vD[ist]; colour = 3, width = 2.1)])

# ------------------------------------------------------------------------------
println("\n" * "="^92)
println("Figures written to ", FOM_FIGDIR)
foreach(f -> println("  ", f), sort(readdir(FOM_FIGDIR)))
println("Demo finished successfully.")
