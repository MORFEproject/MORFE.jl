# evp_compare.py
# ============================================================
# Generalized EVP:  A x = λ B x   with B possibly singular
# Uses SciPy/ARPACK only (works on Windows).
#
# Solvers:
#   LM     : shift-invert around TARGET_SHIFT, then sort by largest Re(λ)
#   ZR     : scan imaginary shifts i*ω on a grid, pool+dedup, select smallest |Re(λ)|
#   CAYLEY : Cayley axis-targeting T=(A-αB)^{-1}(A+αB), select smallest |Re(λ)|
#   CAYLEY2: two-shift Cayley T=(A-σB)^{-1}(A-τB), prioritize largest |μ|
#   HYBRID : LM at small real shift + Cayley screening + targeted LM refinement
#
#
# ============================================================

import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import argparse


# ------------------ user inputs (edit these) ------------------
REQUESTED_EIGENVALUES = 100

# LM shift-invert target (can be complex)
TARGET_SHIFT = 1e-3 + 1j * 0.0

SOLVER_TOL = 1e-16
RANDOM_SEED = 0

# Which solver(s) to run: "LM" | "ZR" | "CAYLEY" | "CAYLEY2" | "HYBRID" | "ALL"
SOLVER = {"LM"}

# Dedup tolerance for pooling across multiple runs (relative)
CANDIDATE_DEDUP_TOL = 1e-2
# Absolute tolerance floor for very small eigenvalues
CANDIDATE_DEDUP_ABS = 1e-16

# ZR imaginary-axis scan controls
ZR_NUM_EIG_RUNS = 3
ZR_IMAG_SHIFT_MIN = 0.0
ZR_IMAG_SHIFT_MAX = 20.0
ZR_EIGS_PER_RUN = 10

# Cayley axis-targeting controls (real alpha)
CAYLEY_ALPHA = 1e-10         # try 0.1, 1, 10 if LU struggles
CAYLEY_BX_TOL = 1e-5      # filter: discard if ||B x|| / ||x|| < this (null(B)/infinite)
CAYLEY_LAM_MAX = 1e6       # discard absurdly large |λ|
CAYLEY_EIGS_FETCH = 50   # None -> fetch ~2*nev to survive filtering

# Two-shift Cayley controls: T=(A-σB)^(-1)(A-τB), μ=(λ-τ)/(λ-σ)
CAYLEY2_AUTO_PARAMS = True
CAYLEY2_SIGMA = 1
CAYLEY2_TAU = -20
CAYLEY2_SIGMA_MARGIN = 1.0
CAYLEY2_TAU_SPAN_MIN = 60.0
CAYLEY2_EIGS_FETCH = 120

# HYBRID controls: LM(seed) -> CAYLEY(screen) -> LM(targeted refine)
HYBRID_BASE_REAL_SHIFT = 1e-3
HYBRID_INITIAL_K = 20
HYBRID_REFINE_K = 8
HYBRID_MAX_TARGET_RUNS = 10
HYBRID_MIN_TARGET_RUNS = 3
HYBRID_SCREEN_NEV_MULT = 10
HYBRID_NOVEL_REL_TOL = 1e-4
HYBRID_TARGET_IMAG_FACTORS = (1.0, 0.9, 1.1)
HYBRID_TARGET_REAL_OFFSETS = (0.0, -0.5, 0.5)
HYBRID_RESIDUAL_CUT = 1e-1

LM_PLOT_FILE = "python_lm.png"
ZR_PLOT_FILE = "python_zr.png"
CAYLEY_PLOT_FILE = "python_cayley.png"
CAYLEY2_PLOT_FILE = "python_cayley2.png"
HYBRID_PLOT_FILE = "python_hybrid.png"
# --------------------------------------------------------------

VALID_SOLVERS = {"LM", "ZR", "CAYLEY", "CAYLEY2", "HYBRID"}


def parse_complex_arg(value: str) -> complex:
    txt = str(value).strip().lower().replace(" ", "")
    if txt.endswith("im"):
        txt = txt[:-2] + "j"
    return complex(txt)


def parse_solver_selection(value):
    if value is None:
        return set()

    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return set()
        txt = txt.strip("[](){}")
        requested = [s.strip().strip("\"'").upper() for s in txt.split(",") if s.strip()]
    elif isinstance(value, (set, list, tuple)):
        requested = [str(s).strip().strip("\"'").upper() for s in value if str(s).strip()]
    else:
        requested = [str(value).strip().upper()]

    if "ALL" in requested:
        return set(VALID_SOLVERS)

    selected = set(requested)
    unknown = sorted(selected - VALID_SOLVERS)
    if unknown:
        valid = ", ".join(sorted(VALID_SOLVERS))
        bad = ", ".join(unknown)
        raise ValueError(f"Unknown solver name(s): {bad}. Valid choices are: {valid}, or ALL.")
    return selected


def load_sparse_csv(path: str) -> sp.csr_matrix:
    df = pd.read_csv(path, header=None, dtype=str, low_memory=False).fillna("0")
    vals = df.values
    n_rows = int(float(vals[0, 1])); n_cols = int(float(vals[0, 2]))
    ijv = vals[1:, :]
    i = ijv[:, 0].astype(float).astype(np.int64) - 1
    j = ijv[:, 1].astype(float).astype(np.int64) - 1
    v = ijv[:, 2].astype(float)
    return sp.coo_matrix((v, (i, j)), shape=(n_rows, n_cols)).tocsr()


def rel_residual(A, B, lam, v):
    Av = A @ v
    Bv = B @ v
    r = Av - lam * Bv
    num = np.linalg.norm(r)
    den = max(1.0, np.linalg.norm(Av), abs(lam) * np.linalg.norm(Bv))
    return num / den


def sort_by_largest_real(lam, vec):
    idx = np.argsort(-np.real(lam))
    return lam[idx], vec[:, idx]


def sort_by_smallest_abs_real(lam, vec, rr=None):
    if rr is None:
        idx = np.argsort(np.abs(lam.real))
        return lam[idx], vec[:, idx]
    idx = np.lexsort((np.abs(lam.imag), rr, np.abs(lam.real)))
    return lam[idx], vec[:, idx], rr[idx]


def is_better_candidate(new_lam, new_rr, old_lam, old_rr):
    # Prefer smaller residual
    if new_rr < old_rr * (1.0 - 1e-12):
        return True
    if old_rr < new_rr * (1.0 - 1e-12):
        return False
    # Tie-break: prefer rightmost
    if np.real(new_lam) > np.real(old_lam) + 1e-12:
        return True
    if np.real(old_lam) > np.real(new_lam) + 1e-12:
        return False
    # Final tie-break: smaller magnitude
    return abs(new_lam) < abs(old_lam)


def same_eig(l1, l2, rel=CANDIDATE_DEDUP_TOL, abs_tol=CANDIDATE_DEDUP_ABS):
    scale = max(1.0, abs(l1), abs(l2))
    return abs(l1 - l2) <= max(abs_tol, rel * scale)


def dedup_pool(pool, rel=CANDIDATE_DEDUP_TOL, abs_tol=CANDIDATE_DEDUP_ABS):
    """
    Merge near-duplicate eigenvalues transitively.
    Keeps best representative by:
      1) smallest residual
      2) largest real part
      3) smallest |λ|
    """
    kept = []
    for lam, vec, rr in pool:
        merged = False
        for i, (l0, v0, r0) in enumerate(kept):
            if same_eig(lam, l0, rel=rel, abs_tol=abs_tol):
                if is_better_candidate(lam, rr, l0, r0):
                    kept[i] = (lam, vec, rr)
                merged = True
                break
        if not merged:
            kept.append((lam, vec, rr))
    return kept


# ------------------------- ZR (imag scan) -------------------------

def zr_style(
    A,
    B,
    nev=10,
    tol=1e-10,
    seed=0,
    num_runs=ZR_NUM_EIG_RUNS,
    imag_min=ZR_IMAG_SHIFT_MIN,
    imag_max=ZR_IMAG_SHIFT_MAX,
    eigs_per_run=ZR_EIGS_PER_RUN,
):
    """
    1) LM shift-invert on a user-defined imaginary-axis grid (sigma = i*ω),
    2) pool + strong dedup,
    3) impose conjugate symmetry in post-processing,
    4) select smallest |Re(λ)|.
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]

    v0 = rng.standard_normal(n)
    ncv = min(max(2 * max(nev, 8) + 10, 60), n - 1)

    def run_shift(sig, k):
        lam, vec = spla.eigs(
            A,
            M=B,
            k=k,
            sigma=sig,
            which="LM",
            tol=tol,
            maxiter=5000,
            ncv=ncv,
            v0=v0,
        )
        return np.array(lam), np.array(vec)

    pool = []

    def add_pool(lams, vecs, rr_cut=1e-6):
        nonlocal pool
        for j in range(len(lams)):
            rr = rel_residual(A, B, lams[j], vecs[:, j])
            if rr > rr_cut:
                continue

            merged = False
            for i, (l0, v0_old, r0) in enumerate(pool):
                if same_eig(lams[j], l0):
                    if is_better_candidate(lams[j], rr, l0, r0):
                        pool[i] = (lams[j], vecs[:, j].copy(), rr)
                    merged = True
                    break

            if not merged:
                pool.append((lams[j], vecs[:, j].copy(), rr))

        pool = dedup_pool(pool)

    k_run = int(max(2, min(n - 2, eigs_per_run if eigs_per_run is not None else max(nev, 12))))
    n_runs = int(max(1, num_runs))
    imag_lo = float(min(imag_min, imag_max))
    imag_hi = float(max(imag_min, imag_max))
    imag_grid = np.linspace(imag_lo, imag_hi, n_runs)

    used_shifts = []

    for wi in imag_grid:
        s = 1j * float(wi)
        used_shifts.append(s)
        try:
            lam_s, X_s = run_shift(s, k_run)
            add_pool(lam_s, X_s)
        except Exception:
            continue

    pool = dedup_pool(pool)
    if len(pool) == 0:
        raise RuntimeError("ZR style produced no candidates after filtering")

    # Impose conjugate symmetry for real-valued matrices
    sym_pool = list(pool)
    for lam, vec, rr in pool:
        if abs(lam.imag) <= 1e-12:
            continue
        lam_c = np.conjugate(lam)
        has_conj = any(same_eig(lam_c, l0) for (l0, _, _) in sym_pool)
        if not has_conj:
            sym_pool.append((lam_c, np.conjugate(vec), rr))

    pool = dedup_pool(sym_pool)

    # Rank: closest to imag axis, then residual, then |Im|
    pool.sort(key=lambda t: (abs(t[0].real), t[2], abs(t[0].imag)))

    out = pool[:min(nev, len(pool))]
    lam = np.array([t[0] for t in out])
    X = np.column_stack([t[1] for t in out])
    rr = np.array([t[2] for t in out])
    meta = {"runs": len(used_shifts), "imag_min": imag_lo, "imag_max": imag_hi, "pool_size": len(pool)}
    return lam, X, rr, meta


# ------------------------- LM (single shift) -------------------------

def run_lm(A, B, nev, sigma, tol, compute_residuals=True):
    t_solve = time.perf_counter()
    lam, vec = spla.eigs(A, M=B, k=nev, sigma=sigma, which="LM", tol=tol, maxiter=5000)
    solve_time = time.perf_counter() - t_solve
    lam = np.array(lam); vec = np.array(vec)
    lam, vec = sort_by_largest_real(lam, vec)
    rr = np.array([rel_residual(A, B, lam[i], vec[:, i]) for i in range(len(lam))]) if compute_residuals else np.array([])
    return lam, vec, rr, {"sigma": sigma, "eigsolve_time_s": solve_time}


# ------------------------- CAYLEY (axis targeting) -------------------------

def run_cayley(A, B, nev, alpha, tol, seed, bx_tol=CAYLEY_BX_TOL, lam_max=CAYLEY_LAM_MAX, ncv=None, fetch=None):
    """
    ARPACK on Cayley transform:
        T = (A - alpha B)^{-1} (A + alpha B)
    Map back:
        λ = alpha * (μ + 1)/(μ - 1)

    Filters out null(B)/infinite candidates by ||B x|| / ||x||.
    Sorts by smallest |Re(λ)|.
    """
    n = A.shape[0]
    if fetch is None:
        fetch = min(n - 2, max(nev, 10) * 2)
    fetch = int(min(n - 2, max(2, fetch)))

    if ncv is None:
        ncv = min(n - 1, max(3 * fetch + 10, 2 * fetch + 20, 3 * nev + 20, 80))
    ncv = int(min(n - 1, max(ncv, fetch + 20)))

    # Factor (A - alpha B)
    S = (A - alpha * B).tocsc()
    lu = spla.splu(S)

    Ap = (A + alpha * B).tocsr()

    def mv(v):
        y = Ap @ v
        return lu.solve(y)

    T = spla.LinearOperator((n, n), matvec=mv, dtype=np.float64)

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n)

    # Robust ARPACK solve: retry with larger subspace and/or smaller k on failure
    solve_tol = max(float(tol), 1e-12)
    k_try = int(fetch)
    ncv_try = int(ncv)
    last_err = None
    mu = X = None
    for _ in range(4):
        try:
            mu, X = spla.eigs(T, k=k_try, which="LM", tol=solve_tol, ncv=ncv_try, v0=v0, maxiter=8000)
            break
        except spla.ArpackError as e:
            last_err = e
            # Typical ARPACK error 3 fix: increase NCV or reduce requested eigenpairs
            ncv_try = int(min(n - 1, max(ncv_try + 20, 2 * k_try + 20)))
            k_try = int(max(2, min(k_try - max(2, k_try // 6), n - 2, ncv_try - 2)))
        except Exception as e:
            last_err = e
            break

    if mu is None or X is None:
        raise RuntimeError(
            f"CAYLEY ARPACK failed after retries (k={k_try}, ncv={ncv_try}, tol={solve_tol:.1e}): {last_err}"
        )

    lam = alpha * (mu + 1.0) / (mu - 1.0)

    AX = A @ X
    BX = B @ X

    keep_lam, keep_vec, keep_rr = [], [], []
    for i in range(len(lam)):
        x = X[:, i]
        nx = np.linalg.norm(x)
        if nx == 0:
            continue
        bx = BX[:, i]
        nbx = np.linalg.norm(bx)

        # filter null(B) / infinite-ish
        if (nbx / nx) < bx_tol:
            continue
        # filter absurd eigenvalues
        if (not np.isfinite(lam[i])) or (abs(lam[i]) > lam_max):
            continue

        r = AX[:, i] - lam[i] * bx
        den = max(1.0, np.linalg.norm(AX[:, i]), abs(lam[i]) * nbx)
        rr = np.linalg.norm(r) / den

        keep_lam.append(lam[i])
        keep_vec.append(x.copy())
        keep_rr.append(rr)

    if len(keep_lam) == 0:
        raise RuntimeError("CAYLEY: no finite candidates after filtering. Try alpha=0.1/1/10 or change bx_tol.")

    lam = np.array(keep_lam)
    vec = np.column_stack(keep_vec)
    rr = np.array(keep_rr)

    lam, vec, rr = sort_by_smallest_abs_real(lam, vec, rr)

    lam = lam[:nev]
    vec = vec[:, :nev]
    rr = rr[:nev]

    meta = {
        "alpha": float(alpha),
        "fetch": int(fetch),
        "fetch_eff": int(k_try),
        "ncv": int(ncv_try),
        "bx_tol": float(bx_tol),
    }
    return lam, vec, rr, meta


def suggest_cayley2_params(A, B, tol, seed):
    """
    Auto-pick (sigma, tau) for two-shift Cayley based on a quick LM probe.
    sigma: slightly right of rightmost estimated eigenvalue.
    tau: well left of estimated spectrum, with enough span for high-|Im| modes.
    """
    n = A.shape[0]
    k_probe = int(max(6, min(n - 2, 20)))
    probe_shift = complex(float(np.real(TARGET_SHIFT)), 0.0)

    try:
        lam_probe, _ = spla.eigs(A, M=B, k=k_probe, sigma=probe_shift, which="LM", tol=tol, maxiter=3000)
        lam_probe = np.array(lam_probe)
        rightmost = float(np.max(np.real(lam_probe)))
        max_abs_im = float(np.max(np.abs(np.imag(lam_probe))))
    except Exception:
        rightmost = -1.0
        max_abs_im = 20.0

    sigma = rightmost + float(CAYLEY2_SIGMA_MARGIN)
    tau_span = max(float(CAYLEY2_TAU_SPAN_MIN), 2.0 * max_abs_im + 20.0)
    tau = rightmost - tau_span
    return float(sigma), float(tau), {
        "rightmost_est": rightmost,
        "max_abs_im_est": max_abs_im,
        "tau_span": tau_span,
    }


def run_cayley2(
    A,
    B,
    nev,
    tol,
    seed,
    sigma=None,
    tau=None,
    auto_params=CAYLEY2_AUTO_PARAMS,
    bx_tol=CAYLEY_BX_TOL,
    lam_max=CAYLEY_LAM_MAX,
    ncv=None,
    fetch=None,
):
    """
    Two-shift Cayley transform:
        T = (A - sigma B)^{-1}(A - tau B)
        mu = (lambda - tau)/(lambda - sigma)
        lambda = (mu * sigma - tau)/(mu - 1)

    We solve for dominant |mu|, map back to lambda, filter, then keep rightmost lambdas.
    """
    n = A.shape[0]
    if ncv is None:
        ncv = min(max(3 * nev + 20, 60), n - 1)
    if fetch is None:
        fetch = min(n - 2, max(nev * 3, CAYLEY2_EIGS_FETCH))
    fetch = int(min(n - 2, max(2, fetch)))
    ncv = int(min(n - 1, max(ncv, fetch + 2)))

    auto_info = {}
    if auto_params:
        sigma_eff, tau_eff, auto_info = suggest_cayley2_params(A, B, tol, seed)
    else:
        sigma_eff = float(CAYLEY2_SIGMA if sigma is None else np.real(sigma))
        tau_eff = float(CAYLEY2_TAU if tau is None else np.real(tau))

    if abs(sigma_eff - tau_eff) < 1e-12:
        tau_eff = sigma_eff - 10.0

    S = (A - sigma_eff * B).astype(complex).tocsc()
    lu = spla.splu(S)
    At = (A - tau_eff * B).astype(complex).tocsr()

    def mv(v):
        return lu.solve(At @ v)

    T = spla.LinearOperator((n, n), matvec=mv, dtype=complex)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n).astype(complex)

    mu, X = spla.eigs(T, k=fetch, which="LM", tol=tol, ncv=ncv, v0=v0, maxiter=5000)
    lam = (mu * sigma_eff - tau_eff) / (mu - 1.0)

    AX = A @ X
    BX = B @ X

    keep = []
    for i in range(len(lam)):
        x = X[:, i]
        nx = np.linalg.norm(x)
        if nx == 0:
            continue
        bx = BX[:, i]
        nbx = np.linalg.norm(bx)
        if (nbx / nx) < bx_tol:
            continue
        if (not np.isfinite(lam[i])) or (abs(lam[i]) > lam_max):
            continue

        r = AX[:, i] - lam[i] * bx
        den = max(1.0, np.linalg.norm(AX[:, i]), abs(lam[i]) * nbx)
        rr = np.linalg.norm(r) / den
        keep.append((lam[i], x.copy(), rr, mu[i]))

    if len(keep) == 0:
        raise RuntimeError("CAYLEY2: no finite candidates after filtering")

    # Screening priority in transformed space: largest |mu| first.
    keep.sort(key=lambda t: (-abs(t[3]), t[2], -np.real(t[0])))
    keep = keep[:min(len(keep), max(nev * 2, nev))]

    # Final reporting order: rightmost first.
    keep.sort(key=lambda t: (-np.real(t[0]), t[2], abs(np.imag(t[0]))))
    out = keep[:min(nev, len(keep))]

    lam_out = np.array([t[0] for t in out])
    vec_out = np.column_stack([t[1] for t in out])
    rr_out = np.array([t[2] for t in out])

    meta = {
        "sigma": float(sigma_eff),
        "tau": float(tau_eff),
        "fetch": int(fetch),
        "auto": bool(auto_params),
        "bx_tol": float(bx_tol),
    }
    for k, v in auto_info.items():
        meta[k] = float(v)
    return lam_out, vec_out, rr_out, meta


def run_hybrid(A, B, nev, tol, seed):
    """
    Hybrid strategy requested by user:
      1) LM shift-invert at a small real shift (captures real-axis modes),
      2) CAYLEY used only to screen additional candidate targets,
      3) Targeted LM runs at screened shifts, taking largest-Re missing values first,
      4) Stop when nev unique eigenpairs are collected.
    """
    n = A.shape[0]
    k0 = int(max(2, min(n - 2, HYBRID_INITIAL_K, nev)))
    k_ref = int(max(2, min(n - 2, HYBRID_REFINE_K, nev)))
    max_target_runs = int(max(0, HYBRID_MAX_TARGET_RUNS))
    min_target_runs = int(max(0, min(HYBRID_MIN_TARGET_RUNS, max_target_runs)))

    pool = []
    targeted_new = []
    lm_runs = 0

    def in_entries(lam, entries):
        return any(
            same_eig(lam, l0, rel=HYBRID_NOVEL_REL_TOL, abs_tol=CANDIDATE_DEDUP_ABS)
            for (l0, _, _) in entries
        )

    def add_pool(lams, vecs, rr_cut=HYBRID_RESIDUAL_CUT, source="seed"):
        nonlocal pool, targeted_new
        for j in range(len(lams)):
            rr = rel_residual(A, B, lams[j], vecs[:, j])
            if rr > rr_cut:
                continue

            exists = in_entries(lams[j], pool)
            if exists:
                continue

            entry = (lams[j], vecs[:, j].copy(), rr)
            pool.append(entry)
            if source == "target":
                targeted_new.append(entry)

    def unique_stable(entries):
        kept = []
        for lam, vec, rr in entries:
            if in_entries(lam, kept):
                continue
            kept.append((lam, vec, rr))
        return kept

    def sigma_neighbors(z):
        zr = float(np.real(z))
        zi = float(np.imag(z))
        sigmas = []
        for f in HYBRID_TARGET_IMAG_FACTORS:
            for dr in HYBRID_TARGET_REAL_OFFSETS:
                sigmas.append(complex(zr + dr, zi * float(f)))
                if abs(zi) > 1e-12:
                    sigmas.append(complex(zr + dr, -zi * float(f)))

        unique = []
        for s in sigmas:
            if not any(abs(s - u) <= max(1e-10, 1e-8 * max(1.0, abs(s), abs(u))) for u in unique):
                unique.append(s)
        return unique

    # Step 1: baseline LM near real axis
    base_sigma = complex(float(HYBRID_BASE_REAL_SHIFT), 0.0)
    lam0, vec0 = spla.eigs(A, M=B, k=k0, sigma=base_sigma, which="LM", tol=tol, maxiter=5000)
    lm_runs += 1
    add_pool(np.array(lam0), np.array(vec0), source="seed")

    # Step 2: CAYLEY screening only (do not directly trust residuals as final)
    screen_nev = int(max(nev, min(n - 2, HYBRID_SCREEN_NEV_MULT * nev)))
    screen_fetch = int(max(screen_nev * 2, CAYLEY_EIGS_FETCH if CAYLEY_EIGS_FETCH is not None else screen_nev * 2))
    screened = []
    cayley_meta = {}
    try:
        lam_scr, _, _, cayley_meta = run_cayley(
            A,
            B,
            nev=screen_nev,
            alpha=float(CAYLEY_ALPHA),
            tol=tol,
            seed=seed,
            fetch=min(screen_fetch, n - 2),
        )
        screened = list(np.array(lam_scr))
    except Exception:
        screened = []

    # Step 3: target screened values, prioritizing rightmost candidates.
    screened.sort(key=lambda z: (np.real(z), abs(np.imag(z))), reverse=True)
    targeted = []
    for z in screened:
        if len(pool) >= nev and len(targeted) >= min_target_runs:
            break

        for sigma_t in sigma_neighbors(z):
            if len(pool) >= nev and len(targeted) >= min_target_runs:
                break
            if len(targeted) >= max_target_runs:
                break

            try:
                lam_t, vec_t = spla.eigs(A, M=B, k=k_ref, sigma=sigma_t, which="LM", tol=tol, maxiter=5000)
                lm_runs += 1
                targeted.append(complex(sigma_t))
                add_pool(np.array(lam_t), np.array(vec_t), source="target")
            except Exception:
                continue

        if len(targeted) >= max_target_runs:
            break

    if len(pool) == 0:
        raise RuntimeError("HYBRID: no valid candidates found")

    pool = unique_stable(pool)
    targeted_unique = unique_stable(targeted_new)

    if len(pool) == 0:
        raise RuntimeError("HYBRID: no output eigenpairs after uniqueness filtering")

    # Main criterion: largest real part.
    pool.sort(key=lambda t: (-np.real(t[0]), t[2], abs(np.imag(t[0]))))
    out = pool[:min(nev, len(pool))]
    lam = np.array([t[0] for t in out])
    vec = np.column_stack([t[1] for t in out])
    rr = np.array([t[2] for t in out])

    meta = {
        "base_sigma": float(np.real(base_sigma)),
        "lm_runs": lm_runs,
        "targeted": len(targeted),
        "targeted_new": len(targeted_unique),
        "min_targeted": min_target_runs,
        "screened": len(screened),
        "pool_size": len(pool),
        "alpha": float(cayley_meta.get("alpha", CAYLEY_ALPHA)),
    }
    return lam, vec, rr, meta


# ------------------------------ main ------------------------------

def main():
    # Parse command-line args for solver selection
    parser = argparse.ArgumentParser(description="Run selected eigensolvers")
    parser.add_argument(
        "--solvers",
        type=str,
        default=None,
        help="Comma-separated list of solvers to run: LM,ZR,CAYLEY,CAYLEY2,HYBRID or ALL. Overrides SOLVER constant.",
    )
    parser.add_argument("--benchmark-lm-only", action="store_true", help="Benchmark only LM eigensolve cost and skip non-solver work.")
    parser.add_argument("--nev", type=int, default=None, help="Override number of requested eigenvalues.")
    parser.add_argument("--sigma", type=str, default=None, help="Override LM shift (examples: 0.001, 10im, 0+10im).")
    args = parser.parse_args()

    print("Loading K.csv and M.csv ...")
    K = load_sparse_csv("K.csv")
    M = load_sparse_csv("M.csv")
    A = (-K).tocsr()
    B = (M).tocsr()

    nEV = int(args.nev) if args.nev is not None else int(REQUESTED_EIGENVALUES)
    sigma = parse_complex_arg(args.sigma) if args.sigma is not None else complex(TARGET_SHIFT)

    if args.benchmark_lm_only:
        t_total = time.perf_counter()
        lam, vec, rr, meta = run_lm(A, B, nEV, sigma, SOLVER_TOL, compute_residuals=False)
        total_time = time.perf_counter() - t_total
        print(f"LM_EIGENSOLVE_COST_S={meta['eigsolve_time_s']:.9f}")
        print(f"LM_BENCHMARK_TOTAL_S={total_time:.9f}")
        print(f"LM_BENCHMARK_NEV={len(lam)}")
        print("completed")
        return

    source = args.solvers if args.solvers is not None else SOLVER
    selected = parse_solver_selection(source)
    if not selected:
        valid = ", ".join(sorted(VALID_SOLVERS))
        raise ValueError(f"No solver selected. Use --solvers or set SOLVER to one of: {valid}, or ALL.")

    solvers = {}  # name -> (lam, vec, rr, meta, elapsed)

    # LM
    if "LM" in selected:
        print(f"Running LM  (sigma={sigma}) ...")
        t0 = time.perf_counter()
        lam, vec, rr, meta = run_lm(A, B, nEV, sigma, SOLVER_TOL, compute_residuals=True)
        solvers["LM"] = (lam, vec, rr, meta, time.perf_counter() - t0)

    # ZR
    if "ZR" in selected:
        print("Running ZR  (imag-axis scan) ...")
        t0 = time.perf_counter()
        lam, vec, rr, meta = zr_style(A, B, nev=nEV, tol=SOLVER_TOL, seed=RANDOM_SEED)
        solvers["ZR"] = (lam, vec, rr, meta, time.perf_counter() - t0)

    # CAYLEY
    if "CAYLEY" in selected:
        print(f"Running CAYLEY  (alpha={CAYLEY_ALPHA}) ...")
        t0 = time.perf_counter()
        lam, vec, rr, meta = run_cayley(
            A, B, nEV,
            alpha=float(CAYLEY_ALPHA),
            tol=SOLVER_TOL,
            seed=RANDOM_SEED,
            fetch=CAYLEY_EIGS_FETCH,
        )
        solvers["CAYLEY"] = (lam, vec, rr, meta, time.perf_counter() - t0)

    # CAYLEY2
    if "CAYLEY2" in selected:
        mode = "auto" if CAYLEY2_AUTO_PARAMS else f"manual sigma={CAYLEY2_SIGMA}, tau={CAYLEY2_TAU}"
        print(f"Running CAYLEY2  ({mode}) ...")
        t0 = time.perf_counter()
        lam, vec, rr, meta = run_cayley2(
            A,
            B,
            nEV,
            tol=SOLVER_TOL,
            seed=RANDOM_SEED,
            sigma=CAYLEY2_SIGMA,
            tau=CAYLEY2_TAU,
            auto_params=CAYLEY2_AUTO_PARAMS,
            fetch=CAYLEY2_EIGS_FETCH,
        )
        solvers["CAYLEY2"] = (lam, vec, rr, meta, time.perf_counter() - t0)

    # HYBRID
    if "HYBRID" in selected:
        print(f"Running HYBRID  (base_real_shift={HYBRID_BASE_REAL_SHIFT}, alpha={CAYLEY_ALPHA}) ...")
        t0 = time.perf_counter()
        lam, vec, rr, meta = run_hybrid(A, B, nEV, SOLVER_TOL, RANDOM_SEED)
        solvers["HYBRID"] = (lam, vec, rr, meta, time.perf_counter() - t0)

    # Print results
    print()
    for name, (lam, vec, rr, meta, elapsed) in solvers.items():
        extra = "  " + "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in meta.items()
        )
        print("-" * 70)
        print(f" Solver : {name}   elapsed={elapsed:.3f} s   count={len(lam)}{extra}")
        print(f" Residuals : min={rr.min():.2e}  max={rr.max():.2e}")
        print(f" {'#':>3}  {'Re(lam)':>12}  {'Im(lam)':>14}   res")
        for i in range(len(lam)):
            print(f" {i+1:3d}  {lam[i].real:+12.6f}  {lam[i].imag:+14.6f}j  {rr[i]:.2e}")
    print("-" * 70)

    # Plots
    if "LM" in solvers:
        lam, _, rr, meta, _ = solvers["LM"]
        plt.figure(figsize=(6, 4))
        sc = plt.scatter(lam.real, lam.imag, s=60, c=lam.real, cmap="plasma", alpha=0.85)
        plt.axvline(0); plt.axhline(0); plt.grid(True)
        plt.xlabel("Real(λ)"); plt.ylabel("Imag(λ)")
        plt.title(f"LM shift-invert (σ={meta.get('sigma')})")
        plt.colorbar(sc, label="Re(λ)")
        plt.tight_layout()
        plt.savefig(LM_PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {LM_PLOT_FILE}")

    if "ZR" in solvers:
        lam, _, rr, meta, _ = solvers["ZR"]
        plt.figure(figsize=(6, 4))
        sc = plt.scatter(lam.real, lam.imag, s=60, c=lam.real, cmap="viridis", alpha=0.85)
        plt.axvline(0); plt.axhline(0); plt.grid(True)
        plt.xlabel("Real(λ)"); plt.ylabel("Imag(λ)")
        plt.title(f"ZR imag-axis scan  runs={meta.get('runs', 0)}")
        plt.colorbar(sc, label="Re(λ)")
        plt.tight_layout()
        plt.savefig(ZR_PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {ZR_PLOT_FILE}")

    if "CAYLEY" in solvers:
        lam, _, rr, meta, _ = solvers["CAYLEY"]
        plt.figure(figsize=(6, 4))
        sc = plt.scatter(lam.real, lam.imag, s=60, c=lam.real, cmap="magma", alpha=0.85)
        plt.axvline(0); plt.axhline(0); plt.grid(True)
        plt.xlabel("Real(λ)"); plt.ylabel("Imag(λ)")
        plt.title(f"CAYLEY axis-targeting  alpha={meta.get('alpha')}")
        plt.colorbar(sc, label="Re(λ)")
        plt.tight_layout()
        plt.savefig(CAYLEY_PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {CAYLEY_PLOT_FILE}")

    if "CAYLEY2" in solvers:
        lam, _, rr, meta, _ = solvers["CAYLEY2"]
        plt.figure(figsize=(6, 4))
        sc = plt.scatter(lam.real, lam.imag, s=60, c=lam.real, cmap="magma", alpha=0.85)
        plt.axvline(0); plt.axhline(0); plt.grid(True)
        plt.xlabel("Real(λ)"); plt.ylabel("Imag(λ)")
        plt.title(f"CAYLEY2 two-shift  σ={meta.get('sigma'):.3g}, τ={meta.get('tau'):.3g}")
        plt.colorbar(sc, label="Re(λ)")
        plt.tight_layout()
        plt.savefig(CAYLEY2_PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {CAYLEY2_PLOT_FILE}")

    if "HYBRID" in solvers:
        lam, _, rr, meta, _ = solvers["HYBRID"]
        plt.figure(figsize=(6, 4))
        sc = plt.scatter(lam.real, lam.imag, s=60, c=lam.real, cmap="cividis", alpha=0.85)
        plt.axvline(0); plt.axhline(0); plt.grid(True)
        plt.xlabel("Real(λ)"); plt.ylabel("Imag(λ)")
        plt.title(f"HYBRID LM+CAYLEY  runs={meta.get('lm_runs', 0)}")
        plt.colorbar(sc, label="Re(λ)")
        plt.tight_layout()
        plt.savefig(HYBRID_PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {HYBRID_PLOT_FILE}")

    print("completed")


if __name__ == "__main__":
    main()