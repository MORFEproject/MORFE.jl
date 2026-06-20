#!/usr/bin/env python3
"""
plot_backbone.py

Read backbone curve data exported by backbone_physical_amplitude.jl and produce
all figures.  Run after the Julia script has written backbone_curves.csv and
backbone_physical_metrics.csv to results/data/.

Usage:
    python validation/plot_backbone.py             # save + open interactive windows
    python validation/plot_backbone.py --no-show   # save to results/figures/ only (headless)

Outputs:
    results/figures/backbone_physical_t1=*_t2=*.png  — per-case overlays
    results/figures/backbone_shift_physical.png       — combined figure
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "results" / "data"
FIGS_DIR = SCRIPT_DIR.parent / "results" / "figures"

CURVES_CSV = DATA_DIR / "backbone_curves.csv"
METRICS_CSV = DATA_DIR / "backbone_physical_metrics.csv"

FIGS_DIR.mkdir(parents=True, exist_ok=True)

if not CURVES_CSV.exists():
    sys.exit(f"Error: {CURVES_CSV} not found.  Run backbone_physical_amplitude.jl first.")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(CURVES_CSV)
metrics = pd.read_csv(METRICS_CSV)

# Ordered unique cases — preserve the order they appear in the CSV (theta2-major)
seen = set()
cases = []
for _, row in df[["theta1", "theta2"]].iterrows():
    key = (round(row.theta1, 6), round(row.theta2, 6))
    if key not in seen:
        seen.add(key)
        cases.append(key)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
})

COLORS = plt.cm.tab10.colors

# ---------------------------------------------------------------------------
# Per-case overlay figures
# ---------------------------------------------------------------------------
for theta1, theta2 in cases:
    sel = (np.isclose(df.theta1, theta1) & np.isclose(df.theta2, theta2))
    exact = df[sel & (df.model == "exact")]
    param = df[sel & (df.model == "param")]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(exact["shift"], exact["amplitude"],
            lw=2, color="black",
            label=f"exact  (θ₁={theta1:.2f}, θ₂={theta2:.2f})")
    ax.plot(param["shift"], param["amplitude"],
            lw=2, color="red", linestyle="--",
            label="parametric ROM")
    ax.set_xlabel("Nonlinear frequency shift  Ω − ω₀  (rad/ms)")
    ax.set_ylabel("Max physical displacement  (via W)")
    ax.set_title(
        f"Backbone in physical amplitude  "
        f"(θ₁={theta1:.2f}, θ₂={theta2:.2f})"
    )
    ax.legend()
    fig.tight_layout()

    fname = f"backbone_physical_t1={theta1:.3f}_t2={theta2:.3f}.png"
    fig.savefig(FIGS_DIR / fname, dpi=150)
    if not SHOW:
        plt.close(fig)
    print(f"Saved -> {FIGS_DIR / fname}")

# ---------------------------------------------------------------------------
# Combined figure  (parametric solid, exact dashed, colour per case)
# ---------------------------------------------------------------------------
fig_all, ax_all = plt.subplots(figsize=(10, 6.4))

for k, (theta1, theta2) in enumerate(cases):
    sel = (np.isclose(df.theta1, theta1) & np.isclose(df.theta2, theta2))
    exact = df[sel & (df.model == "exact")]
    param = df[sel & (df.model == "param")]
    color = COLORS[k % len(COLORS)]

    ax_all.plot(param["shift"], param["amplitude"],
                lw=2, color=color,
                label=f"(θ₁,θ₂)=({theta1:.1f},{theta2:.1f})")
    ax_all.plot(exact["shift"], exact["amplitude"],
                lw=1.4, linestyle="--", color=color)

ax_all.set_xlabel(
    "Nonlinear frequency shift  Ω − ω₀(θ₁,θ₂)  (rad/ms)"
)
ax_all.set_ylabel("Max physical displacement  (via W)")
ax_all.set_title(
    "Backbone shift in physical amplitude — parametric (solid) vs exact (dashed)"
)
ax_all.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
fig_all.tight_layout()
fig_all.savefig(FIGS_DIR / "backbone_shift_physical.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {FIGS_DIR / 'backbone_shift_physical.png'}")
if SHOW:
    plt.show()
else:
    plt.close(fig_all)
