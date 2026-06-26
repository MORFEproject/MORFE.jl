"""
plot_backbone_per_case.py
=========================
Per-arch-height backbone overlay: reference ROM vs parametric ROM.

For each arch height h₀ for which both reference and parametric backbone data
exist, saves one figure showing (Ω − ω₀) / ω₀ vs. peak transverse displacement,
with the reference ROM (solid black) overlaid by the parametric ROM at each
θ-truncation order (coloured dashed/dash-dot/dotted curves).

Input:   ../../results/backbone/backbones.csv   (from validation/backbone.jl)
         ../../results/backbone/metrics.csv
Output:  ../../results/backbone/backbone_<h₀>mm.png  (one file per arch height)

Usage:
    python validation/plots/plot_backbone_per_case.py [--no-show]
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
df = df[df["amplitude"] <= A_MAX]
metrics = pd.read_csv(OUT_DIR / "metrics.csv")
df["z_order"] = df["z_order"].astype(int)
metrics["z_order"] = metrics["z_order"].astype(int)

_omega0 = {}
for _, r in metrics.iterrows():
    _omega0[(round(r.h_ratio, 6), r.model, int(r.z_order))] = r.omega0

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

_order_colors = LinearSegmentedColormap.from_list(
    "order", ["tab:blue", "mediumseagreen", "yellowgreen", "gold", "darkorange", "indianred"]
)(np.linspace(0, 1, len(z_orders_param)))
ORDER_COLOR = {t: _order_colors[i] for i, t in enumerate(sorted(z_orders_param))}

def _title(hr):
    return f"Backbone  —  h₀ = {hr * 1000:.1f} mm  (θ = {hr / H0_L_RATIO - 1:+.2f})"

def select(hr, model, z_order):
    tok = int(z_order)
    sub = df[np.isclose(df.h_ratio, hr) & (df.model == model) &
             (df.z_order == tok)].copy()
    if not sub.empty:
        sub["shift"] = sub["omega"] - _omega0[(round(hr, 6), model, tok)]
    return sub

# ── figures (one per arch height) ─────────────────────────────────────────────
for hr in h_ratios:
    ref = select(hr, "reference", -1)
    has_param = any(not select(hr, "parametric", tok).empty for tok in z_orders_param)
    if ref.empty and not has_param:
        continue

    fig, ax = plt.subplots(figsize=(7, 5))
    if not ref.empty:
        ax.plot(ref["shift"] / ω0_base, ref["amplitude"],
                lw=2, color="black", ls="-", label="reference")
    for tok in sorted(z_orders_param, reverse=True):
        param = select(hr, "parametric", tok)
        if not param.empty:
            ax.plot(param["shift"] / ω0_base, param["amplitude"],
                    lw=1.5, color=ORDER_COLOR[tok], ls="-",
                    label=f"order {tok}")

    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("Nonlinear frequency shift  (Ω − ω₀) / ω₀")
    ax.set_ylabel("Peak midpoint displacement  (mm)")
    ax.set_title(_title(hr))
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    fname = f"backbone_{hr * 1000:.1f}mm.png"
    fig.savefig(OUT_DIR / fname, dpi=150)
    print(f"Saved → {OUT_DIR / fname}")
    if not SHOW:
        plt.close(fig)

if SHOW:
    plt.show()
