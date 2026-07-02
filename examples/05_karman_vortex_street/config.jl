# config.jl — all tunable parameters for the Kármán vortex street DPIM demo.
# Edit this file to reproduce different figures from arXiv:2510.26542v1.

# ── Physics ───────────────────────────────────────────────────────────────────
const Re₀ = 49.03   # expansion Re; paper uses 20, Re_c≈49.03, 70, 80
const MAX_ORD = 9
const ROM = 2        # number of Hopf master modes
const N_EXT = 1        # external parameter dimensions (η′ = 1/Re − 1/Re₀)
const NVAR = ROM + N_EXT   # = 3

# ── Mesh ──────────────────────────────────────────────────────────────────────
const MESH_H_CYL = 0.005   # element size on cylinder surface
const MESH_H_WAKE = 0.015   # element size in the cylinder wake
const MESH_H_BULK = 0.04    # element size in the bulk channel

# ── Eigensolver ───────────────────────────────────────────────────────────────
const EIG_NEV = 40      # number of eigenvalues to compute and display
const EIG_SIGMA_RE = 3.0   # real part of ARPACK shift; offset from the imaginary
# axis avoids near-singularity when Re(λ_Hopf) ≈ 0
const EIG_SIGMA_IM = 8.0   # imag part ≈ Hopf freq (St≈0.2, D=0.1, U_mean=1)
const EIG_TARGET_FREQ = nothing   # rad/s: pin the Hopf mode by Im(λ) ≈ this frequency;
# nothing → smallest |Re(λ)| among Im(λ) > 0 (only reliable near Re_c ≈ 49)
