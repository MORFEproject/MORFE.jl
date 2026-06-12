#!/usr/bin/env python3
"""
numeric_mirror.py — language-independent cross-check of the DEA example numerics.

Replays the Phase 3–6 acceptance checks of implementation_plan_detailed.md in
NumPy/SciPy with the SAME formulas as the Julia sources (config.jl, fem/hermite_beam.jl,
model/bias.jl, model/coupling_terms.jl, solver/eigensolver.jl). Does NOT touch MORFE —
it verifies the model construction, the third-order closure, and the companion
eigenstructure only.

Run:  python3 numeric_mirror.py
"""
import numpy as np
from numpy.linalg import norm, solve
import scipy.linalg as sla

rng = np.random.default_rng(7)
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}")


# ──────────────────────────────────────────────────────────────────────────────
# parameters.jl
# ──────────────────────────────────────────────────────────────────────────────
n_elem, L, EI, rhoA = 50, 1.0, 1.0, 1.0
xi1, w1_target = 0.005, 3.516015087
R, omega_tau = 1.0, 1.0
alpha_c, m_b, V0, v_a = 1.0, 1.0, 1.0, 0.02
eta_over_E = 2 * xi1 / w1_target
c0 = w1_target * R / omega_tau

# ──────────────────────────────────────────────────────────────────────────────
# hermite_beam.jl
# ──────────────────────────────────────────────────────────────────────────────
h = L / n_elem
nd = 2 * (n_elem + 1)


def Ke(EI, h):
    return (EI / h**3) * np.array([
        [12, 6 * h, -12, 6 * h],
        [6 * h, 4 * h**2, -6 * h, 2 * h**2],
        [-12, -6 * h, 12, -6 * h],
        [6 * h, 2 * h**2, -6 * h, 4 * h**2]])


def Me(rhoA, h):
    return (rhoA * h / 420) * np.array([
        [156, 22 * h, 54, -13 * h],
        [22 * h, 4 * h**2, 13 * h, -3 * h**2],
        [54, 13 * h, 156, -22 * h],
        [-13 * h, -3 * h**2, -22 * h, 4 * h**2]])


Kf = np.zeros((nd, nd))
Mf = np.zeros((nd, nd))
bf = np.zeros(nd)
bwf = np.zeros(nd)
ke, me = Ke(EI, h), Me(rhoA, h)
fe_uni = (m_b / L) * np.array([h / 2, h**2 / 12, h / 2, -h**2 / 12])
xg = (0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3))
wq = (0.5, 0.5)
for e in range(n_elem):
    d = slice(2 * e, 2 * e + 4)
    Kf[d, d] += ke
    Mf[d, d] += me
    bf[d] += fe_uni
    xe = e * h
    for xi, w in zip(xg, wq):
        x = xe + xi * h
        mb = m_b * (1 - x / L)
        dN = np.array([(-6 * xi + 6 * xi**2) / h, 1 - 4 * xi + 3 * xi**2,
                       (6 * xi - 6 * xi**2) / h, -2 * xi + 3 * xi**2])
        bwf[d] += w * mb * dN * h

K = Kf[2:, 2:]
M = Mf[2:, 2:]
D = eta_over_E * K
b = bf[2:]
b_weak = bwf[2:]
n = K.shape[0]
g = np.zeros(n)
idx_wtip = n - 2          # Julia n-1 (1-based) ⇒ Python n-2 (0-based)
g[idx_wtip] = 2 * alpha_c / L**2

print("Phase 3 — FE assembly")
check("A1 symmetry/PD", norm(K - K.T) < 1e-12 * norm(K) and np.all(np.linalg.eigvalsh(K) > 0))
w1 = np.sqrt(min(sla.eigvalsh(K, M)))
check("A2 omega1", abs(w1 - w1_target) / w1_target < 1e-3, f"omega1={w1:.7f}")
f = np.zeros(n)
f[idx_wtip] = 1.0
check("A3 tip force statics", abs(solve(K, f)[idx_wtip] - L**3 / (3 * EI)) < 1e-6 * L**3 / 3)
u_st = solve(K, b)
u_ref = (m_b / L) * L**4 / (8 * EI)
check("A4 distributed-load statics", abs(u_st[idx_wtip] - u_ref) < 1e-6 * u_ref,
      f"tip={u_st[idx_wtip]:.8f} vs {u_ref:.8f}")
check("A4 sign convention", g @ u_st > 0)
check("A5 b assemblies agree", norm(b - b_weak) < 1e-12 * norm(b),
      f"diff={norm(b - b_weak):.2e}")
# All w-DOFs (even Julia indices 1,3,5,… ⇒ Python 0,2,4,…) must be loaded;
# interior θ-couples cancel between adjacent elements by design.
w_dofs_loaded = np.all(b[0::2] != 0)
check("A6 b distributed (all w-DOFs)", w_dofs_loaded and np.count_nonzero(b) / n >= 0.45,
      f"{100 * np.count_nonzero(b) / n:.0f}% DOFs")

# ──────────────────────────────────────────────────────────────────────────────
# bias.jl
# ──────────────────────────────────────────────────────────────────────────────
print("Phase 4 — bias point")
xb = solve(K, b)
a = c0 * (g @ xb)
Q0 = V0 / c0
for _ in range(100):
    phi = c0 * Q0 - a * Q0**3 - V0
    dphi = c0 - 3 * a * Q0**2
    assert dphi > 0, "beyond pull-in"
    Q0 -= phi / dphi
    if abs(c0 * Q0 - a * Q0**3 - V0) < 1e-14:
        break
x0 = Q0**2 * xb
chat = c0 * (1 - g @ x0)
btb = b @ b
l0 = K @ b / (2 * Q0 * btb)
l1 = D @ b / (2 * Q0 * btb)
l2 = M @ b / (2 * Q0 * btb)
beta = 2 * Q0 * b

check("A1 mech equilibrium", norm(K @ x0 - b * Q0**2) < 1e-8 * norm(b) * Q0**2)
check("A1 elec equilibrium", abs(c0 * Q0 * (1 - g @ x0) - V0) < 1e-12)
check("pull-in guard", chat > 0.5 * c0, f"chat={chat:.4f}, c0={c0:.4f}, Q0={Q0:.4f}")
pi_ok, worst = True, 0.0
for _ in range(5):
    u = 1e-3 * rng.standard_normal(n)
    ud = 1e-3 * rng.standard_normal(n)
    q = 1e-3 * rng.standard_normal()
    udd = solve(M, beta * q + b * q**2 - D @ ud - K @ u)
    sg = l0 @ u + l1 @ ud + l2 @ udd
    scale = abs(l0 @ u) + abs(l1 @ ud) + abs(l2 @ udd) + abs(q)
    rel = abs(sg - (q + q**2 / (2 * Q0))) / scale
    worst = max(worst, rel)
    pi_ok &= rel < 1e-12
check("A2 proxy identity exact (relative)", pi_ok, f"worst rel = {worst:.2e}")
check("A3 bias strain window", 0.005 <= g @ x0 <= 0.2, f"g.x0={g @ x0:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# coupling_terms.jl — closure checks (Phase 5 A3)
# ──────────────────────────────────────────────────────────────────────────────
print("Phase 5 — third-order closure")
B3 = R * M
B2 = R * D + chat * M
B1 = R * K + chat * D
B0 = chat * K - 2 * c0 * Q0**2 * np.outer(b, g)


def s_trunc(u, ud, udd, v):
    """Truncated RHS scalar (what the MultilinearMaps implement)."""
    sg = l0 @ u + l1 @ ud + l2 @ udd
    gam = g @ u
    return (2 * Q0 * v + 2 * sg * v + 4 * c0 * Q0 * sg * gam - chat * sg**2
            + (chat / Q0) * sg**3 - (1 / Q0) * sg**2 * v)


def s_exact(q, gam, v):
    """Exact RHS scalar of the closure, in terms of the true charge q."""
    return 2 * Q0 * v + 2 * q * v + 4 * c0 * Q0 * q * gam - chat * q**2 \
        + 2 * c0 * q**2 * gam


# (i) algebraic consistency of the closure at roundoff (uses exact q, untruncated F)
u_dir = rng.standard_normal(n)
v_dir = rng.standard_normal(n)
q_dir = rng.standard_normal()
eps = 1e-2
u, ud, q = eps * u_dir, eps * v_dir, eps * q_dir
v = v_a
udd = solve(M, beta * q + b * q**2 - D @ ud - K @ u)
qd = (v + c0 * Q0 * (g @ u) + c0 * q * (g @ u) - chat * q) / R
u3_ref = solve(M, beta * qd + 2 * b * q * qd - D @ udd - K @ ud)
u3_full = solve(B3, s_exact(q, g @ u, v) * b - B2 @ udd - B1 @ ud - B0 @ u)
check("closure exact pre-truncation", norm(u3_full - u3_ref) / norm(u3_ref) < 1e-11,
      f"res={norm(u3_full - u3_ref) / norm(u3_ref):.2e}")

# (ii) truncation error of the RHS scalar: must drop ≈8–16× per ε-halving.
# Scale = sum of magnitudes of the exact-RHS terms (robust to cancellation).
errs = []
for eps in (0.05, 0.025):
    u, ud, q = eps * u_dir, eps * v_dir, eps * q_dir
    udd = solve(M, beta * q + b * q**2 - D @ ud - K @ u)   # consistent state
    gam = g @ u
    e = abs(s_trunc(u, ud, udd, v_a) - s_exact(q, gam, v_a))
    scale = (abs(2 * Q0 * v_a) + abs(2 * q * v_a) + abs(4 * c0 * Q0 * q * gam)
             + abs(chat * q**2) + abs(2 * c0 * q**2 * gam))
    errs.append(e / scale)
ratio = errs[0] / max(errs[1], 1e-300)
check("A3 RHS truncation small", errs[0] < 5e-2, f"rel err={errs[0]:.3e} @ eps=0.05")
check("A3 RHS truncation order", 6 < ratio < 24, f"ratio={ratio:.2f} (expect 8–16)")

# ──────────────────────────────────────────────────────────────────────────────
# dea_demo.jl — companion eigenstructure (Phase 6)
# ──────────────────────────────────────────────────────────────────────────────
print("Phase 6 — cubic pencil spectrum (companion linearisation)")
Z = np.zeros((n, n))
Iden = np.eye(n)
A = np.block([[Z, Iden, Z], [Z, Z, Iden], [-B0, -B1, -B2]])
B = sla.block_diag(Iden, Iden, B3)
lam, V = sla.eig(A, B)
order = np.argsort(np.abs(lam))
lam, V = lam[order], V[:, order]

check("A4 all stable", np.all(lam.real < 0), f"max Re = {lam.real.max():.3e}")
osc = lam[np.abs(lam.imag) > 0.1]
real_br = lam[np.abs(lam.imag) <= 0.1]
lam1 = osc[np.argmin(np.abs(osc))]
if lam1.imag < 0:
    lam1 = np.conj(lam1)
rc_near = real_br[np.argmin(np.abs(real_br + chat / R))].real
print(f"    lambda1 = {lam1:.6f}   (undamped omega1 = {w1:.6f})")
print(f"    RC eigenvalue nearest -chat/R: {rc_near:.6f}  (-chat/R = {-chat / R:.6f})")
check("RC branch present", abs(rc_near + chat / R) < 0.05 * chat / R)
check("master is slowest oscillatory", abs(lam1) <= np.abs(osc).min() + 1e-12)
n_overdamped = len(real_br) - n   # high Kelvin–Voigt modes go overdamped (real pairs)
print(f"    oscillatory modes: {len(osc) // 2} pairs;  real eigenvalues: {len(real_br)} "
      f"(RC branch n={n} + {n_overdamped} from overdamped high modes)")

i1 = np.argmin(np.abs(lam - lam1))
phi = V[:n, i1]
phid1 = V[n:2 * n, i1]
phid2 = V[2 * n:, i1]


def P(lm):
    return lm**3 * B3 + lm**2 * B2 + lm * B1 + B0


def pencil_scale(lm):
    return (abs(lm)**3 * norm(B3) + abs(lm)**2 * norm(B2) + abs(lm) * norm(B1) + norm(B0))


resR = norm(P(lam1) @ phi) / (pencil_scale(lam1) * norm(phi))
check("A1 right pencil residual (normalised)", resR < 1e-10, f"res={resR:.2e}")
check("A3 derivative blocks", norm(phid1 - lam1 * phi) / norm(phid1) < 1e-8
      and norm(phid2 - lam1**2 * phi) / norm(phid2) < 1e-8)

lamL, VL = sla.eig(A.T, B.T)
j1 = np.argmin(np.abs(lamL - lam1))
j2 = np.argmin(np.abs(lamL - np.conj(lam1)))
cands = [VL[2 * n:, j1], np.conj(VL[2 * n:, j1]), VL[2 * n:, j2], np.conj(VL[2 * n:, j2])]
res = [norm(P(lam1).T @ l) / (pencil_scale(lam1) * norm(l)) for l in cands]
check("A2 left pencil residual (best candidate)", min(res) < 1e-10,
      f"res={min(res):.2e} (candidate {int(np.argmin(res))})")

osc_pos = np.sort(osc[osc.imag > 0].imag)
print(f"    bending frequencies: {osc_pos[:3].round(4)} …")

print()
print("ALL CHECKS PASSED" if all(ok) else f"{ok.count(False)} CHECK(S) FAILED")
raise SystemExit(0 if all(ok) else 1)
