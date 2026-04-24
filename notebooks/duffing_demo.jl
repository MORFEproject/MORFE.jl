### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# This Pluto notebook is part of MORFE.jl.
# It demonstrates the Direct Parametrisation of Invariant Manifolds (DPIM)
# on a 2-DOF Duffing oscillator with harmonic forcing.
#
# Running on Binder:
#   https://binder.plutojl.org/v2/gh/MORFEproject/MORFE.jl/main?path=notebooks%2Fduffing_demo.jl
#
# Running locally (requires Julia + Pluto):
#   julia> import Pkg; Pkg.add("Pluto")
#   julia> import Pluto; Pluto.run()
#   Then open this file from the Pluto interface.

# ╔═╡ 00000000-0000-0000-0000-000000000001
begin
	import Pkg
	# Install MORFE from the GitHub repository.
	# On Binder the repo is already cloned, so we develop from the local path.
	# Locally you can replace this with Pkg.develop(PackageSpec(path=".."))
	if isdir(joinpath(@__DIR__, "..", "src"))
		Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
	else
		Pkg.add(url = "https://github.com/MORFEproject/MORFE.jl.git")
	end
	Pkg.add(["Plots", "PlutoUI", "StaticArrays", "HDF5", "LinearAlgebra"])
	Pkg.instantiate()
end

# ╔═╡ 00000000-0000-0000-0000-000000000002
begin
	using MORFE
	using MORFE.Eigensolvers: generalised_eigenpairs
	using MORFE.Multiindices: MultiindexSet, all_multiindices_up_to
	using MORFE.Polynomials: DensePolynomial
	using MORFE.Resonance: resonance_set_from_graph_style
	using MORFE.FullOrderModel: NDOrderModel, MultilinearMap, linear_first_order_matrices
	using MORFE.ExternalSystems: ExternalSystem
	using MORFE.ParametrisationMethod: Parametrisation, ReducedDynamics
	using MORFE.CohomologicalEquations: solve_cohomological_problem

	using LinearAlgebra, StaticArrays
	using Plots, PlutoUI
end

# ╔═╡ 00000000-0000-0000-0000-000000000010
md"""
# MORFE.jl — DPIM Demo: 2-DOF Duffing Oscillator

**Direct Parametrisation of Invariant Manifolds** for a two-degree-of-freedom Duffing system
with harmonic excitation.

This notebook walks through the full MORFE.jl pipeline:

1. Define system matrices
2. Solve the generalised eigenproblem
3. Select master modes and build the multiindex set
4. Solve the cohomological equations → SSM parametrisation W and reduced dynamics R
5. Visualise the SSM and the frequency-response curve

> **Reference**: Jain & Haller (2022), *Nonlinear Dynamics* 107, 1417–1450.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000020
md"""
## 1. System Definition

We study the two coupled Duffing oscillators:

$$M\ddot{x} + C\dot{x} + Kx + \beta x_1^3 \mathbf{e}_1 = \hat{f}\,e^{i\Omega t}$$

with

$$M = I_2, \quad C = \varepsilon I_2, \quad K = \begin{pmatrix}2 & -1 \\ -1 & 2\end{pmatrix},
\quad \hat{f} = \begin{pmatrix}1 \\ 1\end{pmatrix}.$$

Use the sliders below to set the **damping** ε and **nonlinearity** β before solving.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000021
@bind epsilon Slider(0.001:0.001:0.05, default=0.01, show_value=true)

# ╔═╡ 00000000-0000-0000-0000-000000000022
@bind beta Slider(0.0:0.1:2.0, default=1.0, show_value=true)

# ╔═╡ 00000000-0000-0000-0000-000000000023
@bind max_degree Slider(3:1:7, default=5, show_value=true)

# ╔═╡ 00000000-0000-0000-0000-000000000024
md"""
*Damping coefficient* ε = **$(epsilon)** | *Cubic coefficient* β = **$(beta)** | *Polynomial degree* p = **$(max_degree)**
"""

# ╔═╡ 00000000-0000-0000-0000-000000000030
md"""
## 2. System Matrices
"""

# ╔═╡ 00000000-0000-0000-0000-000000000031
begin
	n_dof = 2   # degrees of freedom

	B0 = [2.0 -1.0; -1.0 2.0]           # stiffness K
	B1 = [epsilon 0.0; 0.0 epsilon]      # damping C  (ε·I)
	B2 = [1.0 0.0; 0.0 1.0]             # mass M = I

	md"""System matrices set: n = $n_dof DOF, ε = $epsilon, β = $beta."""
end

# ╔═╡ 00000000-0000-0000-0000-000000000040
md"""
## 3. Nonlinear Terms and External Forcing

The cubic stiffness $\beta x_1^3$ is a *fully symmetric* trilinear map acting on the
position slot of the first DOF.

Harmonic forcing $\hat{f}e^{i\Omega t}$ is encoded as an autonomous external variable
$r(t) = e^{i\Omega t}$ with $\dot r = i\Omega\,r$.  The forcing frequency $\Omega$ is a
parameter of the external system — the DPIM polynomial is computed *once* and is valid for
all $\Omega$.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000041
begin
	# Cubic stiffness  −β x₁³  (minus: force on right-hand side)
	term_cubic = MultilinearMap(
		(res, x1, x2, x3) -> (@. res += -beta * x1 * x2 * x3),
		(3, 0),
	)

	# External forcing:  F·r  (linear in r)
	F_ext = ComplexF64[1.0, 1.0]
	term_forcing = MultilinearMap(
		(res, r) -> (@. res += F_ext * r),
		(0, 0), 1,
	)

	# External system: ṙ = iΩ·r  (Ω is a symbol; set Ω = 1.0 as placeholder)
	Omega_placeholder = 1.0
	external_system = ExternalSystem(
		DensePolynomial(
			ComplexF64[1.0im * Omega_placeholder],
			MultiindexSet([[1]]),
		),
	)

	model = NDOrderModel((B0, B1, B2), (term_cubic, term_forcing), external_system)

	md"""Model assembled: cubic stiffness β=$(beta), harmonic forcing."""
end

# ╔═╡ 00000000-0000-0000-0000-000000000050
md"""
## 4. Spectral Decomposition

Solve the generalised eigenproblem $(A - \lambda B)\varphi = 0$ for the
first-order companion matrices.  The 2-DOF second-order system gives a $4 \times 4$
eigenproblem with two conjugate pairs.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000051
begin
	A_eig, B_eig = linear_first_order_matrices(model)
	eig_result   = eigen(A_eig, B_eig)

	sorted_idx  = sortperm(abs.(eig_result.values))
	sorted_vals = eig_result.values[sorted_idx]
	sorted_vecs = eig_result.vectors[:, sorted_idx]

	eigvec_pos  = sorted_vecs[1:n_dof, :]   # position part

	md"""
	Eigenvalues (sorted by magnitude):
	- λ₁ = $(round(sorted_vals[1], digits=6))
	- λ₂ = $(round(sorted_vals[2], digits=6))
	- λ₃ = $(round(sorted_vals[3], digits=6))
	- λ₄ = $(round(sorted_vals[4], digits=6))
	"""
end

# ╔═╡ 00000000-0000-0000-0000-000000000052
begin
	# Eigenvalue plot
	p_eig = scatter(
		real.(sorted_vals), imag.(sorted_vals);
		xlabel = "Re(λ)",
		ylabel = "Im(λ)",
		title  = "Eigenvalues in the complex plane",
		label  = "eigenvalues",
		markershape = :circle,
		markersize  = 8,
		markercolor = :purple,
		legend      = :topright,
		framestyle  = :box,
	)
	vline!([0.0]; linestyle = :dash, color = :gray, label = "Im axis")
	p_eig
end

# ╔═╡ 00000000-0000-0000-0000-000000000060
md"""
## 5. Master Mode Selection

We select the **first conjugate pair** (least damped) as master modes.  The external
forcing variable adds one more reduced variable, giving NVAR = ROM + 1 = 3.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000061
begin
	ROM   = 2
	N_EXT = 1
	NVAR  = ROM + N_EXT

	master_eigenvalues = SVector{ROM, ComplexF64}(sorted_vals[1:ROM])
	master_modes       = eigvec_pos[:, 1:ROM]          # n × ROM
	left_eigenmodes    = master_modes                  # placeholder (same as right)

	ORD_model = length(model.linear_terms) - 1
	master_modes_derivatives = zeros(ComplexF64, n_dof, ORD_model - 1, ROM)
	for r in 1:ROM
		orig_idx = sorted_idx[r]
		for k in 1:(ORD_model - 1)
			master_modes_derivatives[:, k, r] .=
				eig_result.vectors[(k * n_dof + 1):((k + 1) * n_dof), orig_idx]
		end
	end

	md"""
	Master eigenvalues:
	- λ₁ = $(round(master_eigenvalues[1], digits=6))
	- λ₂ = $(round(master_eigenvalues[2], digits=6))

	External eigenvalue: λ_ext = $(round(complex(1.0im * Omega_placeholder), digits=4))
	"""
end

# ╔═╡ 00000000-0000-0000-0000-000000000070
md"""
## 6. Multiindex Set and Resonance Set

Build all monomials in NVAR = $NVAR variables up to degree $max_degree.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000071
begin
	mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)

	super_eigenvalues = vcat(
		Vector{ComplexF64}(master_eigenvalues),
		Vector{ComplexF64}(external_system.eigenvalues),
	)
	outer_eigenvalues = sorted_vals[(ROM + 1):end]

	resonance_set = resonance_set_from_graph_style(
		ROM, mset, super_eigenvalues, outer_eigenvalues, 0.05,
	)

	n_resonant = sum(resonance_set.resonances)
	md"""
	Multiindex set: $(length(mset)) monomials (NVAR=$NVAR, degree ≤ $max_degree).
	Resonant monomials: $n_resonant.
	"""
end

# ╔═╡ 00000000-0000-0000-0000-000000000080
md"""
## 7. Solve the Cohomological Equations

This is the main DPIM computation.  For each monomial α in graded-lex order, MORFE.jl:
1. computes the superharmonic s = ⟨λ, α⟩
2. evaluates the nonlinear right-hand side Nα from lower-order coefficients
3. solves the stacked cohomological system for W[α] and (if resonant) R[α]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000081
begin
	W, R = solve_cohomological_problem(
		model, mset,
		master_eigenvalues,
		master_modes, left_eigenmodes,
		resonance_set;
		master_modes_derivatives = master_modes_derivatives,
	)

	md"""✓ Cohomological equations solved. Parametrisation W: shape $(size(W.poly.coefficients)), Reduced dynamics R: shape $(size(R.poly.coefficients))."""
end

# ╔═╡ 00000000-0000-0000-0000-000000000090
md"""
## 8. Visualise the SSM

Evaluate $W(z_1, \bar z_1, 0)$ over a grid of reduced coordinates $(z_1, \bar z_1)$ on the
unit circle to trace the manifold geometry in physical space.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000091
begin
	# Parametric evaluation: sweep amplitude a ∈ [0, 0.5] and phase θ ∈ [0, 2π)
	# Reduced coordinate: z₁ = a·e^{iθ},  z₂ = conj(z₁),  r = 0 (autonomous part)
	n_amp   = 8
	n_theta = 60
	amps    = LinRange(0.0, 0.4, n_amp)
	thetas  = LinRange(0.0, 2π, n_theta)

	# Build the exponents list once
	exponents_list = mset.exponents

	function eval_W(W_coeffs, z1, z2, r_val)
		# Evaluate the position component (k=1) of W at (z1, z2, r_val)
		result = zeros(ComplexF64, n_dof)
		for (l, alpha) in enumerate(exponents_list)
			mono_val = z1^alpha[1] * z2^alpha[2] * r_val^alpha[3]
			for i in 1:n_dof
				result[i] += W_coeffs[i, 1, l] * mono_val
			end
		end
		return result
	end

	# Collect SSM surface points
	surf_x1 = Float64[]
	surf_x2 = Float64[]
	surf_z1r = Float64[]

	for a in amps
		for θ in thetas
			z1 = a * exp(im * θ)
			z2 = conj(z1)
			pt = eval_W(W.poly.coefficients, z1, z2, 0.0 + 0.0im)
			push!(surf_x1,  real(pt[1]))
			push!(surf_x2,  real(pt[2]))
			push!(surf_z1r, real(z1))
		end
	end

	p_ssm = scatter3d(
		surf_z1r, surf_x1, surf_x2;
		xlabel     = "Re(z₁)",
		ylabel     = "x₁",
		zlabel     = "x₂",
		title      = "SSM: physical state vs reduced coordinate",
		markersize = 1.5,
		color      = :viridis,
		marker_z   = surf_z1r,
		colorbar   = false,
		label      = false,
	)
	p_ssm
end

# ╔═╡ 00000000-0000-0000-0000-000000000100
md"""
## 9. Amplitude-Frequency Response (Backbone Curve)

The **backbone curve** of the SSM is obtained by evaluating the autonomous reduced dynamics
$\dot z = R_{\text{auto}}(z)$ on the unit-amplitude circle $z_1 = a e^{i\theta}$ and reading
off the instantaneous frequency $\dot\theta = \text{Im}(\dot z_1 / z_1)$.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000101
begin
	function backbone(R_coeffs, exponents_list, amps_bb)
		freqs = Float64[]
		for a in amps_bb
			# average phase velocity over the circle
			freq_sum = 0.0
			for θ in LinRange(0.0, 2π - 2π / 64, 64)
				z1 = a * exp(im * θ)
				z2 = conj(z1)
				zdot1 = zero(ComplexF64)
				for (l, alpha) in enumerate(exponents_list)
					mono = z1^alpha[1] * z2^alpha[2]
					zdot1 += R_coeffs[1, l] * mono
				end
				if abs(z1) > 1e-12
					freq_sum += imag(zdot1 / z1)
				end
			end
			push!(freqs, freq_sum / 64)
		end
		return freqs
	end

	amps_bb = LinRange(0.001, 0.5, 80)
	freqs_bb = backbone(R.poly.coefficients, exponents_list, amps_bb)

	omega1_linear = sqrt(1.0)  # approximate first natural frequency for K=[[2,-1],[-1,2]]

	p_bb = plot(
		freqs_bb, amps_bb;
		xlabel    = "Frequency  ω / (rad s⁻¹)",
		ylabel    = "Amplitude  |z₁|",
		title     = "SSM backbone curve  (β=$(beta), ε=$(epsilon))",
		label     = "backbone",
		linewidth = 2,
		color     = :purple,
		framestyle = :box,
	)
	vline!([omega1_linear]; linestyle = :dash, color = :gray, label = "ω₁ (linear)")
	p_bb
end

# ╔═╡ 00000000-0000-0000-0000-000000000110
md"""
## 10. Key Takeaways

- The SSM parametrisation **W** maps the 2D reduced coordinates $(z_1, \bar z_1)$ to the
  4D full state $(x_1, x_2, \dot x_1, \dot x_2)$ with a polynomial of degree $max_degree.
- The backbone curve shows the frequency shift due to the cubic nonlinearity β = $(beta):
  - β > 0 (hardening): frequency increases with amplitude
  - β < 0 (softening): frequency decreases with amplitude
- Increasing the polynomial degree *p* improves accuracy at higher amplitudes.
- The reduced dynamics **R** is a 2D polynomial ODE that can be continued or integrated
  in microseconds, compared to hours for the full FOM.

### Next Steps

- Add quasi-periodic forcing with multiple frequencies
- Connect to a continuation package (e.g., BifurcationKit.jl) to trace forced-response curves
- Export W and R to MATLAB/Python for post-processing via the `HDF5` output in the demo scripts
"""

# ╔═╡ Cell order:
# ╟─00000000-0000-0000-0000-000000000010
# ╠═00000000-0000-0000-0000-000000000001
# ╠═00000000-0000-0000-0000-000000000002
# ╟─00000000-0000-0000-0000-000000000020
# ╠═00000000-0000-0000-0000-000000000021
# ╠═00000000-0000-0000-0000-000000000022
# ╠═00000000-0000-0000-0000-000000000023
# ╟─00000000-0000-0000-0000-000000000024
# ╟─00000000-0000-0000-0000-000000000030
# ╠═00000000-0000-0000-0000-000000000031
# ╟─00000000-0000-0000-0000-000000000040
# ╠═00000000-0000-0000-0000-000000000041
# ╟─00000000-0000-0000-0000-000000000050
# ╠═00000000-0000-0000-0000-000000000051
# ╠═00000000-0000-0000-0000-000000000052
# ╟─00000000-0000-0000-0000-000000000060
# ╠═00000000-0000-0000-0000-000000000061
# ╟─00000000-0000-0000-0000-000000000070
# ╠═00000000-0000-0000-0000-000000000071
# ╟─00000000-0000-0000-0000-000000000080
# ╠═00000000-0000-0000-0000-000000000081
# ╟─00000000-0000-0000-0000-000000000090
# ╠═00000000-0000-0000-0000-000000000091
# ╟─00000000-0000-0000-0000-000000000100
# ╠═00000000-0000-0000-0000-000000000101
# ╟─00000000-0000-0000-0000-000000000110
