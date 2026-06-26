"""
plot_backbone_shift_reduced.py
===============================
Backbone curves in reduced domain — nonlinear frequency shift.

Plots (Ω − ω₀) / ω₀ vs. modal amplitude |z| for every arch height h₀,
overlaying reference ROMs (solid) against the parametric ROM evaluated at
θ(h₀) for each θ-truncation order (dashed / dash-dot / dotted).
A vertical line at shift = 0 marks the linear frequency.

Input:   ../../results/backbone/backbones_reduced.csv  (from validation/backbone.jl)
         ../../results/backbone/metrics.csv
Output:  ../../results/backbone/backbone_shift_reduced.png

Usage:
    python validation/plots/plot_backbone_shift_reduced.py [--no-show]
"""

import sys
import re
from pathlib import Path

SHOW = "--no-show" not in sys.argv

import numpy as np
import pandas as pd
import matplotlib
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
_config = (_ROOT / "config.jl").read_text()
H0_L_RATIO = float(re.search(r'\bh0_L_ratio\s*=\s*([\d.e+\-]+)', _config).group(1))
A_MAX = float(re.search(r'\ba_max_mm\s*=\s*([\d.e+\-]+)', _config).group(1))
OUT_DIR = _ROOT / "results" / "backbone"
OUT_DIR.mkdir(parents=True, exist_ok=True)
for _f in (OUT_DIR / "backbones.csv", OUT_DIR / "metrics.csv"):
    if not _f.exists():
        sys.exit(f"Error: {_f} not found.  Run validation/backbone.jl first.")

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT_DIR / "backbones.csv")
metrics = pd.read_csv(OUT_DIR / "metrics.csv")
df["z_order"] = df["z_order"].astype(int)
metrics["z_order"] = metrics["z_order"].astype(int)

_omega0 = {}
_mode_norm = {}
for _, r in metrics.iterrows():
    _omega0[(round(r.h_ratio, 6), r.model, int(r.z_order))] = r.omega0
    _mode_norm[(round(r.h_ratio, 6), r.model, int(r.z_order))] = r.mode_norm

def rescale_r(hr, tok):
    """Scale factor: multiply parametric r by this to match the reference eigenvector norm."""
    norm_ref = _mode_norm.get((round(hr, 6), "reference", -1), 1.0)
    norm_p = _mode_norm.get((round(hr, 6), "parametric", int(tok)), 1.0)
    return norm_p / norm_ref

h_ratios = sorted(df["h_ratio"].unique())
z_orders_param = sorted(df[df.model == "parametric"]["z_order"].unique().astype(int))
z_max = max(z_orders_param)
ω0_base = float(
    metrics[np.isclose(metrics.h_ratio, H0_L_RATIO) & (metrics.model == "reference")
            ]["omega0"].iloc[0]
)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12,
                     "axes.labelsize": 13, "axes.titlesize": 13,
                     "legend.fontsize": 10, "lines.linewidth": 2})
COLORS = LinearSegmentedColormap.from_list(
    "custom", ["indianred", "darkorange", "gold", "yellowgreen", "mediumseagreen"]
)(np.linspace(0, 1, len(h_ratios)))

LS = {-1: "-"}
for _i, _t in enumerate(sorted(z_orders_param, reverse=True)):
    LS[int(_t)] = ("--", "-.", ":")[_i] if _i < 3 else ":"

def _lbl(hr):
    return f"h₀ = {hr * 1000:.1f} mm  (θ = {hr / H0_L_RATIO - 1:+.1f})"

def _z_lbl(tok, k_max):
    return "reference" if tok == -1 else (
        f"param. (z-order {k_max})" if k_max == tok else f"param. (z-order {tok})"
    )

style_handles = [
    Line2D([0], [0], color="gray", ls=LS[t], lw=2, label=_z_lbl(t, z_max))
    for t in [-1] + sorted(z_orders_param, reverse=True)
]

def select(hr, model, z_order):
    tok = int(z_order)
    sub = df[np.isclose(df.h_ratio, hr) & (df.model == model) &
             (df.z_order == tok)].copy()
    if not sub.empty:
        omega0 = _omega0.get((round(hr, 6), model, tok), np.nan)
        sub["shift"] = sub["omega"] - omega0
    return sub

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

for k, hr in enumerate(h_ratios):
    color = COLORS[k]
    ref = select(hr, "reference", -1)
    ref = ref[ref["amplitude"] <= A_MAX]
    if not ref.empty:
        ax.plot(ref["shift"] / ω0_base, ref["r"],
                lw=2, color=color, ls=LS[-1], label=_lbl(hr))
    for tok in sorted(z_orders_param, reverse=True):
        param = select(hr, "parametric", tok)
        if not param.empty:
            ax.plot(param["shift"] / ω0_base, param["r"] * rescale_r(hr, tok),
                    lw=1.5, color=color, ls=LS[tok])

ax.add_artist(ax.legend(loc="upper right", title="Arch rise"))
ax.legend(handles=style_handles, loc="lower right")
ax.axvline(0, color="black", lw=0.6, ls=":")
ax.set_xlabel("Nonlinear frequency shift  (Ω − ω₀) / ω₀")
ax.set_ylabel("Modal amplitude  |z|")
ax.set_title("Backbone shift — parametric ROM vs reference — reduced domain")
ax.set_ylim(bottom=0)

fig.tight_layout()
out = OUT_DIR / "backbone_shift_reduced.png"
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
if SHOW:
    plt.show()
