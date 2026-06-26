"""
plot_cubic_force.py
===================
Log-log plot of the parametric cubic-force Taylor-approximation error
vs |θ|, for each truncation order N_θ.

Input:   ../../results/validation/cubic_force_errors.csv  (from validate_cubic_force.jl)
Output:  ../../results/validation/cubic_force_errors.png

Analytical expectation (correct assembly):
  N = 1, 2, 3, 4  → slopes 2, 3, 4, 5 in log-log
  N ≥ 5           → error collapses to machine precision  (H_k = 0 for k > 4)

Failure signature (wrong assembly):
  N ≥ 5 lines do NOT collapse to near zero.

Usage:
    python validation/plots/plot_cubic_force.py [--no-show]
"""

import sys
from pathlib import Path

SHOW = "--no-show" not in sys.argv

import numpy as np
import pandas as pd
import matplotlib
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
_CSV  = _ROOT / "results" / "validation" / "cubic_force_errors.csv"
_OUT  = _ROOT / "results" / "validation" / "cubic_force_errors.png"

if not _CSV.exists():
    sys.exit(f"Error: {_CSV} not found.  Run validation/validate_cubic_force.jl first.")

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(_CSV)
df["N_theta"] = df["N_theta"].astype(int)
orders = sorted(df["N_theta"].unique())

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12,
                     "axes.labelsize": 13, "axes.titlesize": 13,
                     "legend.fontsize": 10})

_colors = LinearSegmentedColormap.from_list(
    "order", ["tab:blue", "mediumseagreen", "yellowgreen", "gold", "darkorange", "indianred"]
)(np.linspace(0, 1, len(orders)))
ORDER_COLOR = {n: _colors[i] for i, n in enumerate(orders)}

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for N in orders:
    sub = df[df["N_theta"] == N].sort_values("theta")
    ax.loglog(sub["theta"], sub["mean_rel_error"],
              "o-", color=ORDER_COLOR[N], lw=1.8, ms=5, label=f"N_θ = {N}")

# Reference slope lines (gray dashed)
x_ref = np.array([0.03, 1.5])
for slope, label in [(2, "slope 2"), (3, "slope 3"), (4, "slope 4"), (5, "slope 5")]:
    anchor_y = 0.15 * x_ref[0] ** slope
    ax.loglog(x_ref, anchor_y * (x_ref / x_ref[0]) ** slope,
              color="gray", ls="--", lw=0.9, alpha=0.6, label=label)

ax.axhline(1e-14, color="gray", ls=":", lw=0.8, alpha=0.5)
ax.text(0.06, 1.5e-14, "machine ε", fontsize=8, color="gray", va="bottom")

ax.set_xlabel("|θ|")
ax.set_ylabel("Mean relative force error")
ax.set_title("Cubic force: Taylor approximation error vs θ")
ax.legend(fontsize=9, ncol=2)
ax.set_xlim(0.03, 1.5)
fig.tight_layout()

_OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(_OUT, dpi=150)
print(f"Saved → {_OUT}")

if SHOW:
    plt.show()
