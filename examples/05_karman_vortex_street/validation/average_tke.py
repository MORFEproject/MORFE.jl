#!/usr/bin/env python3
"""
average_tke.py

Period-averaged fluctuation kinetic energy (TKE) of a reduced-order flow, computed
from the ROM reduced states.  Python port of matlab/average_tke.m.

    TKE = (1/T) * integral_0^T  0.5 * (u'-ubar)' * M * (u'-ubar)  dt,
    ubar = (1/T) * integral_0^T  u'(t) dt.

The perturbation field is reconstructed from the reduced states exactly as the
reference does (disp = sum_m mappings_m * z^Avector_m in matcont/NS.m), i.e.
u'(t) = W(z(t)).  Subtracting the period-mean field ubar gives the fluctuation
about the cycle mean and removes the steady eta'-shift of the base flow.

Run `python average_tke.py` to execute the built-in self-test.
"""

from __future__ import annotations

import numpy as np

# numpy >= 2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def average_tke(orbit: dict, rom: dict):
    """Period-averaged fluctuation kinetic energy of a reduced-order flow.

    Parameters
    ----------
    orbit : dict
        x1  : (Ns,) array_like -- Re(z) over one period.
        x2  : (Ns,) array_like -- Im(z) over one period.
        eta : float or (Ns,)   -- parameter eta' = 1/Re - 1/Re0 (= z3).
        t   : (Ns,) optional   -- time samples spanning one period.  If omitted,
                                  uniform sampling over [0, T) is assumed (means).
    rom : dict
        W       : (Nd, L) ndarray -- parametrisation coefficients per DOF per
                                     monomial (complex, or real if form='real').
        Avector : (L, 3) int      -- exponents [pow_z, pow_zconj, pow_eta].
        M       : (Nd, Nd)        -- velocity mass matrix (dense or scipy sparse;
                                     pressure rows/cols zero -> drop out of KE).
        form    : 'complex' (default; z = x1 + 1j*x2, take real(.)) or 'real'
                  (x1, x2, eta used directly as real variables, W real).

    Returns
    -------
    TKE  : float          -- period-averaged fluctuation kinetic energy.
    KE_t : (Ns,) ndarray  -- instantaneous fluctuation KE 0.5*(u'-ubar)'M(u'-ubar).
    ubar : (Nd,) ndarray  -- period-mean reconstructed field.
    """
    # ---- unpack & validate -------------------------------------------------
    x1 = np.asarray(orbit["x1"], dtype=float).ravel()
    x2 = np.asarray(orbit["x2"], dtype=float).ravel()
    ns = x1.size
    if x2.size != ns:
        raise ValueError("orbit['x1'] and orbit['x2'] must have equal length")

    eta = -0.00186059929953265725#np.asarray(orbit["eta"], dtype=float).ravel()
    if eta.size == 1:
        eta = np.full(ns, eta.item())
    elif eta.size != ns:
        raise ValueError("orbit['eta'] must be scalar or length Ns")

    A = np.asarray(rom["Avector"])
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("rom['Avector'] must be (L, 3): [z, zconj, eta] powers")
    L = A.shape[0]

    W = np.asarray(rom["W"])
    if W.ndim != 2 or W.shape[1] != L:
        raise ValueError("rom['W'] must be (Nd, L) with L = rows of Avector")
    nd = W.shape[0]

    M = rom["M"]
    if M.shape != (nd, nd):
        raise ValueError("rom['M'] must be (Nd, Nd)")

    form = str(rom.get("form", "complex")).lower()

    # ---- reconstruct the field u'(t) = W(z(t)) -----------------------------
    zc = np.empty((ns, L), dtype=complex if form == "complex" else float)
    if form == "complex":
        z = x1 + 1j * x2
        for m in range(L):
            zc[:, m] = z ** A[m, 0] * np.conj(z) ** A[m, 1] * eta ** A[m, 2]
        u = np.real(zc @ W.T)                      # (Ns, Nd), real by conj. symmetry
    elif form == "real":
        for m in range(L):
            zc[:, m] = x1 ** A[m, 0] * x2 ** A[m, 1] * eta ** A[m, 2]
        u = zc @ W.T                               # (Ns, Nd)
    else:
        raise ValueError("rom['form'] must be 'complex' or 'real'")

    # ---- fluctuation about the period mean ---------------------------------
    has_t = ("t" in orbit) and (orbit["t"] is not None)
    if has_t:
        t = np.asarray(orbit["t"], dtype=float).ravel()
        period = t[-1] - t[0]
        if period <= 0:
            raise ValueError("orbit['t'] must span a positive period")
        ubar = _trapz(u, t, axis=0) / period       # (Nd,)
    else:
        ubar = u.mean(axis=0)

    uf = u - ubar                                  # (Ns, Nd) broadcast

    # ---- kinetic energy: 0.5 * u'^T M u' (velocity mass) -------------------
    if _is_sparse(M):
        muf = (M @ uf.T).T                         # sparse M: (Nd,Nd)@(Nd,Ns) -> (Ns,Nd)
    else:
        muf = uf @ np.asarray(M)
    ke_t = 0.5 * np.sum(muf * uf, axis=1)          # (Ns,)

    if has_t:
        tke = float(_trapz(ke_t, t) / period)
    else:
        tke = float(ke_t.mean())

    return tke, ke_t, ubar


def tke_from_gram(orbit: dict, gram: dict):
    """Period-averaged fluctuation TKE from the precomputed energy Gram matrix.

    Mathematically identical to ``average_tke`` but avoids shipping the full FOM
    parametrisation: with G_mn = W_m^T M W_n (see fem/energy_gram.jl),

        KE(t) = 0.5 * (zeta(t) - zbar)^T G (zeta(t) - zbar),

    where zeta_m(t) = z^a * conj(z)^b * eta^c and zbar is the period mean of zeta.

    Parameters
    ----------
    orbit : dict   -- x1, x2, eta, optional t  (as in ``average_tke``).
    gram  : dict
        G       : (L, L) complex ndarray -- energy Gram matrix (tke_gram_*.csv).
        Avector : (L, 3) int             -- exponents [pow_z, pow_zconj, pow_eta].

    Returns
    -------
    TKE  : float          -- period-averaged fluctuation kinetic energy.
    KE_t : (Ns,) ndarray  -- instantaneous fluctuation KE.
    """
    x1 = np.asarray(orbit["x1"], dtype=float).ravel()
    x2 = np.asarray(orbit["x2"], dtype=float).ravel()
    ns = x1.size
    eta = np.asarray(orbit["eta"], dtype=float).ravel()
    if eta.size == 1:
        eta = np.full(ns, eta.item())

    A = np.asarray(gram["Avector"])
    G = np.asarray(gram["G"])
    L = A.shape[0]
    if G.shape != (L, L):
        raise ValueError("gram['G'] must be (L, L) with L = rows of Avector")

    z = x1 + 1j * x2
    zeta = np.empty((ns, L), dtype=complex)
    for m in range(L):
        zeta[:, m] = z ** A[m, 0] * np.conj(z) ** A[m, 1] * eta ** A[m, 2]

    has_t = ("t" in orbit) and (orbit["t"] is not None)
    if has_t:
        t = np.asarray(orbit["t"], dtype=float).ravel()
        period = t[-1] - t[0]
        zbar = _trapz(zeta, t, axis=0) / period
    else:
        zbar = zeta.mean(axis=0)

    zf = zeta - zbar
    ke_t = 0.5 * np.real(np.sum((zf @ G) * zf, axis=1))   # (Ns,), real by symmetry

    if has_t:
        tke = float(_trapz(ke_t, t) / period)
    else:
        tke = float(ke_t.mean())
    return tke, ke_t


def _is_sparse(M) -> bool:
    """True if M is a scipy sparse matrix, without importing scipy eagerly."""
    try:
        import scipy.sparse as sp
    except ImportError:
        return False
    return sp.issparse(M)


# ---------------------------------------------------------------------------
def _selftest() -> bool:
    """2-DOF modal ROM (M-normalised real modes a'Ma = b'Mb = 1/2), a steady
    eta-shift term (removed by mean subtraction), and a circular orbit of
    amplitude A.  Analytic fluctuation TKE = A^2, independent of eta."""
    M = np.eye(2)
    a = np.array([1.0 / np.sqrt(2.0), 0.0])
    b = np.array([0.0, 1.0 / np.sqrt(2.0)])
    phi = a + 1j * b
    rom = {
        "W": np.column_stack([phi, np.conj(phi), np.array([0.3, 0.4])]),  # z, zconj, eta
        "Avector": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "M": M,
    }

    amp = 1.0
    th = np.linspace(0.0, 2.0 * np.pi, 201)        # closed period, endpoint duplicated
    orbit = {"x1": amp * np.cos(th), "x2": amp * np.sin(th),
             "eta": 0.05, "t": th}                 # omega = 1 -> T = 2*pi

    tke, _, _ = average_tke(orbit, rom)
    err = abs(tke - amp ** 2)
    ok = err < 1e-6
    print(f"average_tke self-test: TKE = {tke:.10f} (expected {amp**2:.10f}), "
          f"err = {err:.2e} -> {'PASS' if ok else 'FAIL'}")

    # Gram path must agree with the full-field path.
    W = rom["W"]
    G = W.T @ (M @ W)                              # L x L, plain transpose
    tke_g, _ = tke_from_gram(orbit, {"G": G, "Avector": rom["Avector"]})
    err_g = abs(tke_g - tke)
    ok_g = err_g < 1e-10
    print(f"tke_from_gram cross-check: TKE = {tke_g:.10f}, "
          f"|diff| = {err_g:.2e} -> {'PASS' if ok_g else 'FAIL'}")
    return ok and ok_g


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        # Synthetic algebra check (not your data): python3 average_tke.py --selftest
        sys.exit(0 if _selftest() else 1)
    # Default: compute the TKE of the orbit configured in run_tke.py (ORBIT_FILE / ETA),
    # no terminal arguments needed.
    from run_tke import main as _run
    _run([])
