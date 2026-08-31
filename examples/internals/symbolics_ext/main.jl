# # Symbolic full-order model — examples
#
# A runnable walkthrough of the examples from the
# ["Symbolic full-order model"](@ref) tutorial: the two-mass Shaw–Pierre
# oscillator, and three `ExternalSystem` drivers (harmonic, quasi-periodic /
# multiharmonic, chaotic). For the general API and the full contract, see
# the tutorial itself — this script only walks the examples.

using MORFE, Symbolics

# ## Full model: two-mass oscillator
#
# The two-degree-of-freedom nonlinear oscillator introduced by Shaw and
# Pierre: two coupled masses with linear stiffness and damping, the first
# mass carrying an additional cubic restoring force and a harmonic forcing
# term.
#
# ```math
# \ddot u_1 + c\,\dot u_1 - c\,\dot u_2 + (1+k)\,u_1 - k\,u_2 + g\,u_1^3 - 2\cos(\Omega t) = 0
# ```
# ```math
# \ddot u_2 - c\,\dot u_1 + 2c\,\dot u_2 - k\,u_1 + (1+k)\,u_2 = 0
# ```

# ### Step 1 — without forcing
#
# Starting from the unforced/autonomous equation, dropping $-2\cos(\Omega t)$:

@variables u[1:2] du[1:2] ddu[1:2]
u, du, ddu = collect(u), collect(du), collect(ddu)

k, g, c = 1.0, 6.0, 0.1

exprs = [
	ddu[1] + c*du[1] - c*du[2] + (1 + k)*u[1] - k*u[2] + g*u[1]^3,
	ddu[2] - c*du[1] + 2*c*du[2] - k*u[1] + (1 + k)*u[2],
]

groups = (u, du, ddu)   # ascending order: (u, u̇, ü)

model = model_from_symbolics(exprs, groups)

# ### Step 2 — with harmonic forcing
#
# We substitute the non-autonomous term $-2\cos(\Omega t)$ with $-r_1-r_2$,
# where $\dot r_1=i\Omega r_1$ and $\dot r_2=-i\Omega r_2$, and add this ODE
# as an external system:

@variables u[1:2] du[1:2] ddu[1:2] r[1:2]
u, du, ddu = collect(u), collect(du), collect(ddu)
groups = (u, du, ddu)   # ascending order: (u, u̇, ü)

k, g, c = 1.0, 6.0, 0.1
Ω = 1.3

exprs = [
	ddu[1] + c*du[1] - c*du[2] + (1 + k)*u[1] - k*u[2] + g*u[1]^3 - r[1] - r[2],
	ddu[2] - c*du[1] + 2*c*du[2] - k*u[1] + (1 + k)*u[2],
]

ext_exprs = [
	im * Ω * r[1],
	-im * Ω * r[2],
]

ext_var = collect(r)
model = model_from_symbolics(exprs, groups, ext_var, ext_exprs)

# ## Examples for ExternalSystem
#
# ### Harmonic excitation
#
# The driver behind the forcing term above, on its own — the eigenvalue-tuple
# constructor `ExternalSystem((im*Ω, -im*Ω))` written as its own right-hand
# side instead:

@variables r[1:2]
r = collect(r)
Ω = 1.3

ext_exprs = [im*Ω*r[1], -im*Ω*r[2]]
harmonic = externalsystem_from_symbolics(ext_exprs, r)   # same object as ExternalSystem((im*Ω, -im*Ω))

# ### Quasi-periodic and multiharmonic drivers
#
# A pair of incommensurate frequencies closes onto a torus rather than a
# circle — still linear, still diagonal, just twice as many states:

@variables r[1:4]
r = collect(r)
Ω1, Ω2 = 1.0, sqrt(2)   # incommensurate

ext_exprs = [
	im*Ω1*r[1], -im*Ω1*r[2],
	im*Ω2*r[3], -im*Ω2*r[4],
]
quasi = externalsystem_from_symbolics(ext_exprs, r)   # same object as ExternalSystem((im*Ω1,-im*Ω1,im*Ω2,-im*Ω2))

# The eigenvalue-tuple constructor only reaches diagonal systems, though.
# Suppose the two signals you actually want are $r_1 = -\cos\Omega t$ and
# $r_2 = 0.03\sin\Omega t - \cos 4\Omega t$ — not diagonal in the obvious
# basis. Introducing auxiliary states $r_3=\sin\Omega t$, $r_4=\sin 4\Omega t$
# closes this as a first-order linear system $\dot{\mathbf r}=\mathbf A\mathbf r$:

Ω = 1.3
@variables r[1:4]
r = collect(r)

ext_exprs = [
	Ω*r[3],
	-0.03*Ω*r[1] + 4*Ω*r[4],
	-Ω*r[1],
	-4*Ω*r[2] + 0.12*Ω*r[3],
]
multiharmonic = externalsystem_from_symbolics(ext_exprs, r)   # same object as ExternalSystem(linear_polynomial(A))

# `externalsystem_from_symbolics` extracts the same linear matrix
# $\mathbf A$ from these expressions that you'd build by hand — it isn't
# upper triangular here, so MORFE re-bases it internally and reports the
# change of coordinates, exactly as it does for the matrix form.

# ### Chaotic nonlinear excitation
#
# A nonlinear driver works the same way — as long as it is polynomial and
# has an equilibrium at the origin. The Lorenz system has three equilibria,
# none at the origin, so we shift coordinates to the nontrivial equilibrium
# $C_+=(C,C,\rho-1)$, $C=\sqrt{\beta(\rho-1)}$, before defining it:
#
# ```math
# \dot X = \sigma(Y-X), \qquad \dot Y = X - Y - Z(X+C), \qquad \dot Z = C(X+Y)+XY-\beta Z
# ```

σ, ρ, β = 10.0, 28.0, 8/3
C = sqrt(β*(ρ-1))

@variables X Y Z
r = [X, Y, Z]

ext_exprs = [
	σ*(Y-X),
	X - Y - Z*(X+C),
	C*(X+Y) + X*Y - β*Z,
]
lorenz = externalsystem_from_symbolics(ext_exprs, r)

# ## Using the function layout
#
# Everything above only *builds* symbolic objects — it never runs anything.
# To see a driver actually do what it claims, we integrate a tiny,
# self-contained SDOF oscillator
#
# ```math
# \ddot u + c\,\dot u + k\,u = \mathrm{forcing}(\mathbf r), \qquad \dot{\mathbf r} = \mathbf E(\mathbf r)
# ```
#
# forced by whichever driver is uncommented below, and plot $u(t)$ next to
# the raw forcing signal. This needs `DifferentialEquations` and `Plots` in
# addition to `MORFE`/`Symbolics`.

using DifferentialEquations, Plots

# ### Pick exactly one forcing block
#
# Every block below defines the same three things: `ext_rhs!(dr, r, p, t)`
# (the driver's own ODE, numeric — not symbolic), `r0` (its initial
# condition), and `forcing(r)` (what actually enters the oscillator).
# Uncomment the one you want to try, leave the other two commented out —
# nothing else in this section needs to change to switch drivers.

## --- Option 1: harmonic excitation ------------------------------------
Ω = 1.3
ext_rhs!(dr, r, p, t) = (dr[1] = im*Ω*r[1]; dr[2] = -im*Ω*r[2])
r0 = ComplexF64[1.0, 1.0]
forcing(r) = real(r[1] + r[2])                 # = 2 cos(Ωt)

## --- Option 2: multiharmonic excitation -------------------------------
# Ω = 1.3
# function ext_rhs!(dr, r, p, t)
#     dr[1] = Ω*r[3]
#     dr[2] = -0.03*Ω*r[1] + 4*Ω*r[4]
#     dr[3] = -Ω*r[1]
#     dr[4] = -4*Ω*r[2] + 0.12*Ω*r[3]
# end
# r0 = ComplexF64[-1.0, -1.0, 0.0, 0.0]         # r₁(0)=-1, r₂(0)=-1, r₃(0)=r₄(0)=0
# forcing(r) = real(r[2])                        # = 0.03 sin(Ωt) - cos(4Ωt)

## --- Option 3: chaotic (Lorenz) excitation -----------------------------
# σ, ρ, β = 10.0, 28.0, 8/3
# C = sqrt(β*(ρ-1))
# function ext_rhs!(dr, r, p, t)
#     X, Y, Z = r[1], r[2], r[3]
#     dr[1] = σ*(Y-X)
#     dr[2] = X - Y - Z*(X+C)
#     dr[3] = C*(X+Y) + X*Y - β*Z
# end
# r0 = ComplexF64[1.0, 1.0, 1.0]                 # any small offset from the (unstable) origin
# forcing(r) = 0.05*real(r[1])                   # scaled down — Lorenz amplitudes are large

# ### Integrate and plot
#
# The oscillator and the driver are simulated together as one state vector
# `[u, du, r...]` — this part is the same regardless of which block above is active:

c, k = 0.2, 1.0   # a plain, lightly damped SDOF oscillator

function combined_rhs!(dstate, state, p, t)
	u, du = state[1], state[2]
	r = @view state[3:end]
	dr = @view dstate[3:end]
	ext_rhs!(dr, r, p, t)

	dstate[1] = du
	dstate[2] = -c*du - k*u + forcing(r)
end

state0 = vcat(0.0+0im, 0.0+0im, r0)   # u(0)=0, u̇(0)=0, r(0) from the active block
tspan = (0.0, 40.0)

prob = ODEProblem(combined_rhs!, state0, tspan)
sol = solve(prob, Tsit5())

forcing_t = real.(forcing.(eachcol(sol[3:end, :])))

plot(sol.t, real.(sol[1, :]); label = "u(t)", xlabel = "t")
plot!(sol.t, forcing_t; label = "forcing(t)", linestyle = :dash)

# If $u(t)$ visibly locks onto the shape of `forcing(t)` — a clean cosine
# for Option 1, the two-frequency wave for Option 2, or a chaotic-looking
# trace for Option 3 — the driver is doing exactly what its symbolic
# definition above says it should.
