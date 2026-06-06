### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# This Pluto notebook is part of MORFE.jl.
# It demonstrates the Direct Parametrisation of Invariant Manifolds (DPIM)
# on a 2-DOF Duffing oscillator with harmonic forcing.
#
# ── Running on Binder (no local install required) ──────────────────────────────
#   https://binder.plutojl.org/v2/gh/MORFEproject/MORFE.jl/main?path=notebooks%2Fduffing_demo.jl
#
# ── Running locally ────────────────────────────────────────────────────────────
#   julia> import Pkg; Pkg.add("Pluto")
#   julia> import Pluto; Pluto.run()
#   Then open this file from the Pluto interface.
#
# The notebook activates the notebooks/Project.toml environment, which lists
# MORFE as a local path dependency (resolves to the repo root automatically).

# ╔═╡ f3a1c2d4-1111-4abc-8def-aabbccddeeff
begin
    import Pkg
    # Activate the notebooks/ environment, which contains MORFE as a
    # local path source (see notebooks/Project.toml).
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

# ╔═╡ a2b3c4d5-2222-4bcd-9ef0-bbccddeeff00
begin
    using MORFE
    using MORFE.Multiindices: all_multiindices_up_to
    using MORFE.Polynomials: DensePolynomial
    using MORFE.Resonance: resonance_set_from_graph_style
    using MORFE.FullOrderModel: NDOrderModel, MultilinearMap, linear_first_order_matrices
    using MORFE.ExternalSystems: ExternalSystem
    using MORFE.CohomologicalEquations: solve_cohomological_problem

    using LinearAlgebra, StaticArrays
    using Plots, PlutoUI
end

# ╔═╡ e1f2a3b4-3333-4cde-a0f1-ccddeeff0011
md"""
# MORFE.jl — DPIM Demo: 2-DOF Duffing Oscillator

**Direct Parametrisation of Invariant Manifolds** applied to a two-degree-of-freedom
Duffing system with harmonic excitation.

Full pipeline:
1. Define system matrices (K, C, M) and nonlinear terms
2. Solve the generalised eigenproblem (4×4)
3. Select master modes and build the multiindex set
4. Solve the cohomological equations → SSM parametrisation **W** and reduced dynamics **R**
5. Visualise the SSM geometry and the backbone curve

> **Reference:** Opreni, Vizzaccaro, Touzé & Frangi (2021), *npj Comput. Mater.* 7, 161.
"""

# ╔═╡ b4c5d6e7-4444-4def-b102-ddeeff001122
md"""
## 1. System Parameters

$$M\ddot{x} + C\dot{x} + Kx + \beta x_1^3 \mathbf{e}_1 = \hat{f}\,e^{i\Omega t}$$

with $M = I_2$, $K = \begin{pmatrix}2&-1\\-1&2\end{pmatrix}$, $C = \varepsilon I_2$,
$\hat{f} = (1,\,1)^\top$.

Adjust the sliders — changing any parameter re-runs the entire pipeline automatically.
"""

# ╔═╡ c5d6e7f8-5555-4ef0-c213-eeff00112233
@bind epsilon Slider(0.005:0.005:0.05, default = 0.01, show_value = true)

# ╔═╡ d6e7f809-6666-4f01-d324-ff0011223344
@bind beta Slider(0.0:0.2:2.0, default = 1.0, show_value = true)

# ╔═╡ e7f80910-7777-4012-e435-001122334455
@bind max_degree Slider(3:1:7, default = 5, show_value = true)

# ╔═╡ f8091011-8888-4123-f546-112233445566
md"""
**ε** (damping) = $(epsilon) | **β** (cubic stiffness) = $(beta) | **p** (polynomial degree) = $(max_degree)
"""

# ╔═╡ 09101112-9999-4234-0657-223344556677
md"""## 2. Full-Order Model"""

# ╔═╡ 10111213-aaaa-4345-1768-334455667788
begin
    n_dof = 2

    B0 = [2.0 -1.0; -1.0 2.0]                       # stiffness K
    B1 = epsilon .* Matrix{Float64}(I, n_dof, n_dof) # damping C = ε I
    B2 = Matrix{Float64}(I, n_dof, n_dof)            # mass M = I

    # Cubic stiffness  −β x₁³  (minus sign: force is on the RHS)
    term_cubic = MultilinearMap(
        (res, x1, x2, x3) -> (@. res += -beta * x1 * x2 * x3),
        (3, 0)
    )

    # Harmonic forcing  F̂·r  (linear in the external variable r)
    F_ext = ComplexF64[1.0, 1.0]
    term_forcing = MultilinearMap(
        (res, r) -> (@. res += F_ext * r),
        (0, 0), 1
    )

    # External system: ṙ = iΩ₀·r  (Ω₀ is a symbolic placeholder;
    # the DPIM polynomial covers all forcing amplitudes at this frequency)
    Omega0 = 1.0
    external_system = ExternalSystem((ComplexF64(im * Omega0),))

    model = NDOrderModel((B0, B1, B2), (term_cubic, term_forcing), external_system)

    md"""Model assembled: n=$(n_dof) DOF, ε=$(epsilon), β=$(beta), Ω₀=$(Omega0)."""
end

# ╔═╡ 11121314-bbbb-4456-2879-445566778899
md"""
## 3. Spectral Decomposition

The second-order system with 2 DOF gives a 4×4 first-order generalised eigenproblem
$(A - \lambda B)\varphi = 0$, yielding two conjugate pairs.
"""

# ╔═╡ 12131415-cccc-4567-398a-5566778899aa
begin
    A_eig, B_eig = linear_first_order_matrices(model)
    eig_result = eigen(A_eig, B_eig)

    # Sort eigenvalues by magnitude (least-damped first)
    sorted_idx = sortperm(abs.(eig_result.values))
    sorted_vals = eig_result.values[sorted_idx]
    sorted_vecs = eig_result.vectors[:, sorted_idx]

    eigvec_pos = sorted_vecs[1:n_dof, :]   # position block of each eigenvector

    md"""
    Eigenvalues (sorted by |λ|):
    - λ₁ = $(round(sorted_vals[1], digits=5))
    - λ₂ = $(round(sorted_vals[2], digits=5))
    - λ₃ = $(round(sorted_vals[3], digits=5))
    - λ₄ = $(round(sorted_vals[4], digits=5))
    """
end

# ╔═╡ 13141516-dddd-4678-4a9b-6677889900bb
let
    p = scatter(
        real.(sorted_vals), imag.(sorted_vals);
        xlabel = "Re(λ)",
        ylabel = "Im(λ)",
        title = "Eigenvalues",
        label = "eigenvalues",
        markersize = 8,
        color = :purple,
        framestyle = :box
    )
    vline!([0.0]; linestyle = :dash, color = :gray, label = "Im axis")
    p
end

# ╔═╡ 14151617-eeee-4789-5bac-7788990011cc
md"""
## 4. Master Mode Selection

We take the first conjugate pair (least damped) as the **master modes** (ROM = 2).
One external variable (the forcing r) gives NVAR = 3.
"""

# ╔═╡ 15161718-ffff-4890-6cbd-8899001122dd
begin
    ROM = 2
    N_EXT = 1
    NVAR = ROM + N_EXT

    master_eigenvalues = SVector{ROM, ComplexF64}(sorted_vals[1:ROM])
    master_modes = eigvec_pos[:, 1:ROM]        # n_dof × ROM
    left_eigenmodes = master_modes                # placeholder (same as right)

    ORD_model = length(model.linear_terms) - 1       # = 2 for second-order system
    master_modes_derivatives = zeros(ComplexF64, n_dof, ORD_model - 1, ROM)
    for r in 1:ROM
        orig_idx = sorted_idx[r]
        for k in 1:(ORD_model - 1)
            master_modes_derivatives[:, k, r] .= eig_result.vectors[(k * n_dof + 1):((k + 1) * n_dof), orig_idx]
        end
    end

    md"""Master eigenvalues: λ₁ = $(round(master_eigenvalues[1], digits=5)), λ₂ = $(round(master_eigenvalues[2], digits=5))"""
end

# ╔═╡ 16171819-0000-4901-7dce-990011223300
md"""
## 5. Multiindex Set & Resonance Set

All monomials in NVAR = $(NVAR) variables up to degree **$(max_degree)**.
"""

# ╔═╡ 17181920-1111-4012-8edf-001122330011
begin
    mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)

    outer_eigenvalues = sorted_vals[(ROM + 1):end]

    resonance_set = resonance_set_from_graph_style(
        mset, Vector{ComplexF64}(master_eigenvalues),
        Vector{ComplexF64}(external_system.eigenvalues), outer_eigenvalues, 0.05)

    n_res = sum(any(resonant_targets(resonance_set, l)) for l in 1:length(mset))
    md"""Multiindex set: $(length(mset)) monomials. Resonant monomials: $(n_res)."""
end

# ╔═╡ 18192021-2222-4123-9ef0-1122334400aa
md"""
## 6. DPIM Solve

For each monomial α in GrLex order:
1. Compute superharmonic s = ⟨λ, α⟩
2. Evaluate nonlinear RHS Nα from already-solved lower-order coefficients
3. Solve the (augmented) cohomological system for **W[α]** and (if resonant) **R[α]**
"""

# ╔═╡ 19202122-3333-4234-af01-2233445511bb
begin
    W,
    R = solve_cohomological_problem(
        model, mset,
        master_eigenvalues,
        master_modes, left_eigenmodes,
        resonance_set;
        master_modes_derivatives = master_modes_derivatives
    )
    md"""✓ Done. W: $(size(W.poly.coefficients)), R: $(size(R.poly.coefficients))."""
end

# ╔═╡ 20212223-4444-4345-b012-3344556622cc
md"""
## 7. SSM Geometry

Evaluate **W(a e^{iθ}, a e^{-iθ}, 0)** over a polar grid of reduced coordinates
to trace the 2D manifold embedded in the 4D state space.
(r = 0 → autonomous part of the SSM, independent of forcing amplitude.)
"""

# ╔═╡ 21222324-5555-4456-c123-4455667733dd
begin
    exponents_list = mset.exponents   # Vector{SVector{NVAR, Int}}

    function eval_W_pos(W_coeffs, exps, z1::ComplexF64, z2::ComplexF64, r::ComplexF64)
        n = size(W_coeffs, 1)
        result = zeros(ComplexF64, n)
        for (l, alpha) in enumerate(exps)
            mono = z1^alpha[1] * z2^alpha[2] * r^alpha[3]
            for i in 1:n
                result[i] += W_coeffs[i, 1, l] * mono
            end
        end
        return result
    end

    n_amp, n_th = 7, 50
    amps_surf = LinRange(0.0, 0.4, n_amp)
    thetas_surf = LinRange(0.0, 2π, n_th)

    pts_z1r = Float64[]
    pts_x1 = Float64[]
    pts_x2 = Float64[]

    for a in amps_surf, θ in thetas_surf

        z1 = a * exp(im * θ)
        pt = eval_W_pos(W.poly.coefficients, exponents_list, z1, conj(z1), 0.0 + 0im)
        push!(pts_z1r, real(z1))
        push!(pts_x1, real(pt[1]))
        push!(pts_x2, real(pt[2]))
    end

    scatter3d(
        pts_z1r, pts_x1, pts_x2;
        xlabel = "Re(z₁)",
        ylabel = "x₁",
        zlabel = "x₂",
        title = "SSM geometry (autonomous, r=0)",
        markersize = 1.2,
        marker_z = pts_z1r,
        colorbar = false,
        label = false
    )
end

# ╔═╡ 22232425-6666-4567-d234-5566778844ee
md"""
## 8. Backbone Curve

The backbone curve tracks the instantaneous frequency
$\omega(a) = \operatorname{Im}(\dot z_1 / z_1)$
of the autonomous SSM dynamics as a function of amplitude $a = |z_1|$.

Only monomials with $\alpha_r = 0$ (the autonomous part of **R**) contribute here.
"""

# ╔═╡ 23242526-7777-4678-e345-6677889955ff
begin
    function backbone_curve(R_coeffs, exps, amps)
        freqs = Vector{Float64}(undef, length(amps))
        n_avg = 64
        thetas_bb = LinRange(0.0, 2π * (1 - 1 / n_avg), n_avg)

        for (k, a) in enumerate(amps)
            freq_sum = 0.0
            for θ in thetas_bb
                z1 = a * exp(im * θ)
                z2 = conj(z1)
                # Evaluate ż₁ = R₁(z₁, z₂, r=0) — only autonomous monomials contribute
                zdot1 = zero(ComplexF64)
                for (l, alpha) in enumerate(exps)
                    alpha[3] != 0 && continue           # skip forced monomials (r≠0 terms)
                    mono = z1^alpha[1] * z2^alpha[2]
                    zdot1 += R_coeffs[1, l] * mono
                end
                abs(z1) > 1e-14 && (freq_sum += imag(zdot1 / z1))
            end
            freqs[k] = freq_sum / n_avg
        end
        return freqs
    end

    amps_bb = LinRange(1e-3, 0.45, 100)
    freqs_bb = backbone_curve(R.poly.coefficients, exponents_list, amps_bb)

    # Linear natural frequency of the first mode: eigenvalues of K w.r.t. M
    omega1_lin = abs(imag(master_eigenvalues[1]))

    p_bb = plot(
        freqs_bb, collect(amps_bb);
        xlabel = "ω  (rad/s)",
        ylabel = "|z₁|  (amplitude)",
        title = "Backbone curve  (β=$(beta), ε=$(epsilon), p=$(max_degree))",
        label = "backbone",
        linewidth = 2,
        color = :purple,
        framestyle = :box
    )
    vline!([omega1_lin]; linestyle = :dash, color = :gray, label = "ω₁ (linear)")
    p_bb
end

# ╔═╡ 24252627-8888-4789-f456-7788990066aa
md"""
## Summary

| Symbol | Shape | Contents |
|--------|-------|----------|
| W.poly.coefficients | $(size(W.poly.coefficients)) | SSM embedding: full state vs reduced coordinates |
| R.poly.coefficients | $(size(R.poly.coefficients)) | Reduced dynamics on the SSM |

**Interpretation of the backbone curve:**
- β > 0 (hardening): frequency *increases* with amplitude
- β < 0 (softening): frequency *decreases* with amplitude
- Increasing polynomial degree p improves accuracy at larger amplitudes
- The forced response curve (FRC) can be traced by evaluating R with r ≠ 0

**Next steps** from this reduced-order model:
- Pass R to a continuation solver (e.g. BifurcationKit.jl) for the FRC
- Export W, R to HDF5 for post-processing in Python/MATLAB
- Increase p for higher accuracy near turning points
"""

# ╔═╡ Cell order:
# ╟─e1f2a3b4-3333-4cde-a0f1-ccddeeff0011
# ╟─b4c5d6e7-4444-4def-b102-ddeeff001122
# ╠═c5d6e7f8-5555-4ef0-c213-eeff00112233
# ╠═d6e7f809-6666-4f01-d324-ff0011223344
# ╠═e7f80910-7777-4012-e435-001122334455
# ╟─f8091011-8888-4123-f546-112233445566
# ╟─09101112-9999-4234-0657-223344556677
# ╠═10111213-aaaa-4345-1768-334455667788
# ╟─11121314-bbbb-4456-2879-445566778899
# ╠═12131415-cccc-4567-398a-5566778899aa
# ╠═13141516-dddd-4678-4a9b-6677889900bb
# ╟─14151617-eeee-4789-5bac-7788990011cc
# ╠═15161718-ffff-4890-6cbd-8899001122dd
# ╟─16171819-0000-4901-7dce-990011223300
# ╠═17181920-1111-4012-8edf-001122330011
# ╟─18192021-2222-4123-9ef0-1122334400aa
# ╠═19202122-3333-4234-af01-2233445511bb
# ╟─20212223-4444-4345-b012-3344556622cc
# ╠═21222324-5555-4456-c123-4455667733dd
# ╟─22232425-6666-4567-d234-5566778844ee
# ╠═23242526-7777-4678-e345-6677889955ff
# ╟─24252627-8888-4789-f456-7788990066aa
# ╠═f3a1c2d4-1111-4abc-8def-aabbccddeeff
# ╠═a2b3c4d5-2222-4bcd-9ef0-bbccddeeff00
