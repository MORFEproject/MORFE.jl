"""
    eigensolver.jl — Direct shift-invert ARPACK eigensolver for the NSE descriptor system.

The linearised system B₁ṡ = −B₀ s has a singular mass B₁ (descriptor system):
    B₁ = [M  0]   (no time derivative for pressure)
         [0  0]

We solve A y = λ B y  with  A = A_lin = −B₀,  B = B₁  via shift-invert ARPACK:
    σ = sigma_re + im*sigma_im
    F = klu(A − σB)
    T = F⁻¹ B                     (LinearMap)
    eigs(T; which=:LM) → μ
    λ = σ + 1/μ

sigma_re offsets the shift from the imaginary axis; sigma_im targets a frequency
band.  The Hopf mode is selected as the eigenvalue with the smallest |Re(λ)| among
those with Im(λ) > 0 — this is independent of the exact shift used.

Left eigenvectors are computed by per-eigenvalue adjoint shifts and then
biorthogonally normalised so that ψᵀ B φ = 1.
"""

using Arpack
using KLU
using LinearAlgebra
using LinearMaps
using Printf
using SparseArrays
using StaticArrays

"""
    solve_hopf_eigenproblem(A_lin, B_mass; nev, sigma_re, sigma_im)
        -> (master_eigenvalues, master_modes, left_eigenmodes)

Compute `nev` eigenvalues of the generalised problem A_lin y = λ B_mass y by
shift-invert ARPACK, print a formatted table of all results, then return the
Hopf complex-conjugate pair.

Arguments:
  A_lin       — linearised NSE operator  (n_free × n_free sparse)
  B_mass      — singular mass matrix     (n_free × n_free sparse)
  nev         — total number of eigenvalues to compute
  sigma_re    — real part of ARPACK shift
  sigma_im    — imaginary part of shift    (estimated Hopf frequency)
  target_freq — optional (rad/s): select the Hopf mode whose Im(λ) is closest to
                this frequency instead of the smallest-|Re(λ)| heuristic.  The
                default heuristic is only reliable near Re_c, where the shedding
                mode IS the least-damped one; away from Re_c (e.g. the paper's
                Re₀ = 20/70/80 cases) another oscillatory mode may sit closer to
                the imaginary axis and be picked silently — pin the frequency then.

Returns:
  master_eigenvalues — SVector{2,ComplexF64}: Hopf pair [λ₁, λ₂], Im(λ₁) > 0
  master_modes       — n_free × 2 right eigenvector matrix
  left_eigenmodes    — n_free × 2 left eigenvector matrix  (biorthonormal w.r.t. B)
"""
function solve_hopf_eigenproblem(
    A_lin::AbstractSparseMatrix,
    B_mass::AbstractSparseMatrix;
    nev::Int,
    sigma_re::Float64,
    sigma_im::Float64,
    target_freq::Union{Nothing, Float64} = nothing,
)
    n = size(A_lin, 1)
    sigma = complex(sigma_re, sigma_im)

    println("  Shift σ = $sigma,  nev = $nev,  n = $n")

    # ── Shift-invert factorisation ─────────────────────────────────────────────
    Ac = complex.(A_lin)
    Bc = complex.(B_mass)
    F  = klu(Ac - sigma * Bc)

    LM = LinearMap{ComplexF64}(n, n; ismutating = false) do x
        F \ (Bc * x)
    end

    mu, vecs, = eigs(LM; nev = nev, which = :LM, maxiter = 3000,
                            ncv = min(max(nev + 30, 120), n - 1))

    # guard against zero mu (spurious pressure modes)
    tiny = eps(Float64)
    mu_safe = map(m -> abs(m) < tiny ? complex(tiny) : m, mu)
    vals = sigma .+ inv.(mu_safe)

    # ── Print eigenvalue table ─────────────────────────────────────────────────
    println()
    @printf("  %3s   %-14s  %-14s  %-12s\n", "#", "Re(λ)", "Im(λ)", "|λ|")
    println("  " * "─"^52)
    for (i, λ) in enumerate(vals)
        @printf("  %3d   %+12.6f  %+12.6f  %12.6f\n",
                i, real(λ), imag(λ), abs(λ))
    end
    println()

    # ── Select Hopf mode: smallest |Re(λ)| among Im(λ) > 0 ──────────────────
    # A_lin and B_mass are real → eigenpairs come in conjugate pairs.
    # The Hopf mode is the one closest to the imaginary axis (smallest damping).
    # sigma_im is only used for the factorisation shift; selection is independent.
    hopf_tol = 0.1
    cand = findall(λ -> imag(λ) > hopf_tol, vals)
    isempty(cand) && error("No eigenvalue with Im(λ) > 0 found; increase nev.")

    i_best = if target_freq === nothing
        cand[argmin(abs(real(vals[i])) for i in cand)]
    else
        cand[argmin(abs(imag(vals[i]) - target_freq) for i in cand)]
    end
    λ₁     = vals[i_best]
    φ₁     = vecs[:, i_best]

    # Conjugate partner: exact for real-matrix systems
    λ₂ = conj(λ₁)
    φ₂ = conj(φ₁)

    @printf("  Selected Hopf mode:  λ₁ = %+.6f %+.6f·i  (ω_c = %.4f rad/s)\n",
            real(λ₁), imag(λ₁), imag(λ₁))

    # ── Left eigenvector for mode 1 (adjoint shift-invert) ────────────────────
    # Shift at λ₁ so A^T − λ₁·B^T is singular there → ARPACK returns ψ₁_bilinear.
    # Using conj(λ₁) = λ₂ would find ψ₂ instead, making J₁ᵀ·φ₁ = 0 (biorthogonality).
    Ac_adj = Ac'
    Bc_adj = Bc'
    F_adj  = klu(Ac_adj - λ₁ * Bc_adj)
    LM_adj = LinearMap{ComplexF64}(n, n; ismutating = false) do x
        F_adj \ (Bc_adj * x)
    end
    _, xl, = eigs(LM_adj; nev = 1, which = :LM, maxiter = 3000,
                  ncv = min(60, n - 1))
    ψ₁ = xl[:, 1]

    # ── Biorthogonal normalisation: ψ₁ᵀ B φ₁ = 1 (scale both by 1/√α) ─────────
    # Dividing both by √α keeps the normalisation symmetric (‖ψ‖ ~ ‖φ‖) and
    # avoids an ill-conditioned bordered system when ψ₁ is much larger than φ₁.
    α  = transpose(ψ₁) * (Bc * φ₁)
    sq_α = sqrt(α)
    φ₁ = φ₁ ./ sq_α
    ψ₁ = ψ₁ ./ sq_α
    φ₂ = conj(φ₁)
    ψ₂ = conj(ψ₁)

    master_eigenvalues = SVector{2, ComplexF64}(λ₁, λ₂)
    master_modes       = hcat(φ₁, φ₂)
    # ψ₁ is the BILINEAR left vector (ψ₁ᵀ(A − λ₁B) = 0). The orthogonality
    # equations use the sesquilinear convention φᴴ(λB − A) = 0, which for the
    # real matrices here is φ@λ₁ = conj(ψ₁), φ@λ₂ = conj(ψ₂). With these the
    # assembled row φᴴB₁W = 0 equals the bilinear ψ₁ᵀB₁W = 0 exactly; passing
    # ψ un-conjugated would pair each border row with the WRONG mode of the
    # conjugate pair (cross-mode pairing ≈ 0 → near-singular resonant solves).
    left_eigenmodes    = hcat(conj.(ψ₁), conj.(ψ₂))

    # ── All modes (positive-imaginary half, sorted by |Re(λ)|) ────────────────
    # Filter out conjugate duplicates; sort most-weakly-damped first so that
    # index 1 always matches the primary Hopf mode selected above.
    all_idx        = sort(findall(λ -> imag(λ) > hopf_tol, vals); by = i -> abs(real(vals[i])))
    all_eigenvalues = vals[all_idx]
    all_modes       = vecs[:, all_idx]

    println("  Returning $(length(all_idx)) modes (Im > $hopf_tol, sorted by |Re(λ)|):")
    for (k, i) in enumerate(all_idx)
        @printf("    mode %2d:  λ = %+.6f %+.6f·i\n", k, real(vals[i]), imag(vals[i]))
    end

    return master_eigenvalues, master_modes, left_eigenmodes, all_eigenvalues, all_modes
end
