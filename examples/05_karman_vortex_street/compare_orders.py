#!/usr/bin/env python3
"""compare_orders.py — cross-order comparison of lift and period-averaged TKE (STEP 3).

For every results/Re*_ord*/data/ containing rom_branch.csv, tke_gram_*.csv and
L_coefficients.csv, evaluates along each branch point's circular orbit
z(t) = rho*exp(i*Omega*t):
  - avg_TKE      via tke_from_gram (validation/average_tke.py)
  - max_abs_lift via the lift polynomial L(z) (includes the constant base-flow row)

Outputs results/comparison/{comparison.csv, lift_vs_Re.png, tke_vs_Re.png}.
Usage:  python3 compare_orders.py            (no arguments)
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "validation"))
from average_tke import tke_from_gram  # noqa: E402

NS = 256  # orbit samples per period


def load_gram(data_dir: Path) -> dict:
    G = (np.loadtxt(data_dir / "tke_gram_re.csv", delimiter=",")
         + 1j * np.loadtxt(data_dir / "tke_gram_im.csv", delimiter=","))
    A = np.loadtxt(data_dir / "tke_avector.csv", delimiter=",").astype(int)
    return {"G": np.atleast_2d(G), "Avector": np.atleast_2d(A)}


def load_lift(data_dir: Path):
    exps, coeffs = [], []
    with open(data_dir / "L_coefficients.csv") as f:
        for row in csv.DictReader(f):
            exps.append((int(row["exp_1"]), int(row["exp_2"]), int(row["exp_3"])))
            coeffs.append(float(row["L_re"]) + 1j * float(row["L_im"]))
    return np.array(exps, int), np.array(coeffs, complex)


def lift_series(z: np.ndarray, eta: float, exps: np.ndarray, coeffs: np.ndarray):
    zc = np.conj(z)
    L = np.zeros_like(z, dtype=complex)
    for (a, b, c), C in zip(exps, coeffs):
        L += C * z**a * zc**b * eta**c
    return L.real


def process_run(run_dir: Path):
    data = run_dir / "data"
    m = re.search(r"ord(\d+)$", run_dir.name)
    order = int(m.group(1))
    gram = load_gram(data)
    exps, coeffs = load_lift(data)
    rows = []
    with open(data / "rom_branch.csv") as f:
        for row in csv.DictReader(f):
            eta, Re = float(row["eta"]), float(row["Re"])
            rho, om, T = float(row["rho"]), float(row["omega"]), float(row["T"])
            th = 2 * np.pi * np.arange(NS) / NS          # uniform over one period
            z = rho * np.exp(1j * th)
            tke, _ = tke_from_gram({"x1": z.real, "x2": z.imag, "eta": eta}, gram)
            max_lift = float(np.max(np.abs(lift_series(z, eta, exps, coeffs))))
            rows.append((order, eta, Re, rho, om, T, tke, max_lift))
    return rows


def main():
    run_dirs = sorted(p for p in (HERE / "results").glob("Re*_ord*")
                      if (p / "data" / "rom_branch.csv").exists())
    if not run_dirs:
        sys.exit("no rom_branch.csv found — run main.jl then solve_rom.jl first")
    all_rows = [r for d in run_dirs for r in process_run(d)]

    out = HERE / "results" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    hdr = "order,eta,Re,rho,omega,T,avg_TKE,max_abs_lift"
    np.savetxt(out / "comparison.csv", np.array(all_rows), delimiter=",",
               header=hdr, comments="", fmt="%.10e")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = np.array(all_rows)
    for col, fname, ylabel in ((7, "lift_vs_Re.png", "max |lift|"),
                               (6, "tke_vs_Re.png", "period-averaged TKE")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for o in sorted(set(arr[:, 0].astype(int))):
            sel = arr[arr[:, 0] == o]
            ax.plot(sel[:, 2], sel[:, col], marker=".", ms=3, label=f"order {o}")
        if col == 6:
            ax.set_yscale("log")   # post-fold tails otherwise dwarf the physical branch
        ax.set_xlabel("Re")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=200)
        plt.close(fig)
        print(f"wrote {out / fname}")
    print(f"wrote {out / 'comparison.csv'}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
