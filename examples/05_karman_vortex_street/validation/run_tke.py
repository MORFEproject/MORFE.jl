#!/usr/bin/env python3
"""
run_tke.py — one-command runner for the period-averaged fluctuation TKE.

Given a reduced-state orbit (t, z1=Re z, z2=Im z) and a parameter eta', reports the
period-averaged fluctuation kinetic energy of the Kármán ROM

    TKE = (1/T) integral_0^T  0.5 * (u'-ubar)' M (u'-ubar) dt.

It uses the tiny energy Gram matrix G = W^T M W exported per run by main.jl
(fem/energy_gram.jl), so no huge parametrisation file is needed.

Examples
--------
    # orbit CSV (columns t,x,y) + Reynolds number (eta' = 1/Re - 1/Re0)
    python3 run_tke.py --data-dir results/Re49.03_ord9/data --orbit orbit.csv --re 52

    # give eta' directly
    python3 run_tke.py --data-dir results/Re49.03_ord9/data --orbit orbit.csv --eta -1.1e-3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from average_tke import tke_from_gram

_HERE = Path(__file__).resolve().parent
_EXAMPLE_DIR = _HERE.parent

# ---------------------------------------------------------------------------
# Configuration — edit these to run without command-line arguments.
# ---------------------------------------------------------------------------
DATA_DIR = None                                  # required: pass --data-dir results/ReXX.XX_ordN/data
ORBIT_FILE = "orbit_max_amplitude.csv"          # relative to DATA_DIR (or an absolute path)
ETA = -0.00003571047174310654                    # parameter eta' = 1/Re - 1/Re0 (Re ~ 49.1160)
RE0 = 49.03                                       # expansion Reynolds number
# ---------------------------------------------------------------------------


def _load_gram(data_dir: Path):
    """Load G and Avector (exported per run by main.jl via fem/energy_gram.jl)."""
    re_csv = data_dir / "tke_gram_re.csv"
    im_csv = data_dir / "tke_gram_im.csv"
    av_csv = data_dir / "tke_avector.csv"

    if not (re_csv.exists() and im_csv.exists() and av_csv.exists()):
        sys.exit(f"ERROR: Gram CSVs not found in {data_dir} — run main.jl first "
                 "(it exports tke_gram_*.csv per order).")

    G = (np.loadtxt(re_csv, delimiter=",")
         + 1j * np.loadtxt(im_csv, delimiter=","))
    A = np.loadtxt(av_csv, delimiter=",").astype(int)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    return {"G": np.atleast_2d(G), "Avector": A}


def _load_orbit(orbit_csv: Path):
    """Read an orbit CSV with columns t, x, y (x = Re z, y = Im z)."""
    data = np.loadtxt(orbit_csv, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t, x, y = data[:, 0], data[:, 1], data[:, 2]
    return t, x, y


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="ROM data directory (holds W.jls / gram CSVs), "
                        "e.g. results/Re49.03_ord3/data.")
    p.add_argument("--orbit", type=Path, default=None,
                   help="Orbit CSV with columns t,x,y. Default: in-file ORBIT_FILE.")
    p.add_argument("--re0", type=float, default=RE0,
                   help="Expansion Reynolds number Re0. Default: in-file RE0.")
    p.add_argument("--re", type=float, default=None,
                   help="Orbit Reynolds number; eta' = 1/Re - 1/Re0.")
    p.add_argument("--eta", type=float, default=None,
                   help="Parameter eta' directly (overrides --re). Default: in-file ETA.")
    args = p.parse_args(argv)

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        sys.exit(f"ERROR: data dir not found: {data_dir}")

    # Orbit: CLI --orbit wins; otherwise in-file ORBIT_FILE (relative to data_dir).
    if args.orbit is not None:
        orbit_csv = args.orbit
    else:
        orbit_csv = Path(ORBIT_FILE)
        if not orbit_csv.is_absolute():
            orbit_csv = data_dir / orbit_csv
    if not orbit_csv.exists():
        sys.exit(f"ERROR: orbit CSV not found: {orbit_csv}")

    # eta': CLI --eta wins, then --re, then in-file ETA.
    if args.eta is not None:
        eta = args.eta
    elif args.re is not None:
        eta = 1.0 / args.re - 1.0 / args.re0
    else:
        eta = ETA

    gram = _load_gram(data_dir)
    t, x, y = _load_orbit(orbit_csv)
    orbit = {"x1": x, "x2": y, "eta": eta, "t": t}

    tke, ke_t = tke_from_gram(orbit, gram)

    print("-" * 60)
    print(f"orbit      : {orbit_csv}")
    print(f"samples    : {t.size}   period T = {t[-1] - t[0]:.6g}")
    print(f"eta'       : {eta:.6e}"
          + (f"   (Re = {1.0 / (1.0/args.re0 + eta):.4f})" if eta != 0 else ""))
    print(f"KE range   : [{ke_t.min():.6e}, {ke_t.max():.6e}]")
    print(f"average TKE: {tke:.10e}")
    print("-" * 60)
    return tke


if __name__ == "__main__":
    main()
