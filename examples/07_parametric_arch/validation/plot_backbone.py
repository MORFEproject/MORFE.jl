#!/usr/bin/env python3
"""
plot_backbone.py

Read backbone data produced by validation/backbone.jl and generate figures.

Usage:
    python validation/plot_backbone.py             # save + open interactive windows
    python validation/plot_backbone.py --no-show   # save to results/backbone/ only (headless)

Outputs  (all in results/backbone/):
    backbone_absolute.png    — Ω/ω₀ vs physical amplitude, all h₀/L cases
    backbone_shift.png       — (Ω − ω₀)/ω₀ vs physical amplitude, all h₀/L cases
    backbone_h<ratio>.png    — per-case overlay (reference vs parametric, all θ-orders)
    eigenfreq_vs_h.png       — linear ω₀/ω₀_base vs arch height ratio h₀/L
"""

import sys
import re
from pathlib import Path

SHOW = "--no-show" not in sys.argv

import numpy as np
import pandas as pd
import matplotlib

# ── read shared config (h0_L_ratio) from config.jl ───────────────────────────
_cfg = (Path(__file__).parent.parent / "config.jl").read_text()
H0_L_RATIO = float(re.search(r'\bh0_L_ratio\s*=\s*([\d.e+\-]+)', _cfg).group(1))

if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
OUT_DIR    = SCRIPT_DIR.parent / "results" / "backbone"

CURVES_CSV  = OUT_DIR / "backbones.csv"
METRICS_CSV = OUT_DIR / "metrics.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not CURVES_CSV.exists():
    sys.exit(f"Error: {CURVES_CSV} not found.  Run validation/backbone.jl first.")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df      = pd.read_csv(CURVES_CSV)
metrics = pd.read_csv(METRICS_CSV)

# Ensure theta_order is integer (not float) after CSV round-trip
df["theta_order"]      = df["theta_order"].astype(int)
metrics["theta_order"] = metrics["theta_order"].astype(int)

# ω₀ look-up: (h_ratio_rounded, model, theta_order) → ω₀
_omega0 = {}
for _, row in metrics.iterrows():
    hr  = round(row.h_ratio, 3)
    tok = int(row.theta_order)
    _omega0[(hr, row.model, tok)] = row.omega0

h_ratios = sorted(df["h_ratio"].unique())

# Parametric truncation orders present in the data (ascending)
theta_orders_param = sorted(
    df[df.model == "parametric"]["theta_order"].unique().astype(int)
)
theta_max = max(theta_orders_param)

# Base-configuration eigenfrequency (θ=0, reference) for normalisation
_base_row = metrics[
    np.isclose(metrics["h_ratio"], H0_L_RATIO) & (metrics["model"] == "reference")
]
ω0_base = float(_base_row["omega0"].iloc[0])

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

COLORS = plt.cm.viridis(np.linspace(0.15, 0.85, len(h_ratios)))

# Line styles ordered by distance from theta_max: max→dashed, max-1→dashdot, max-2→dotted, max-3→loosely dashed
_LS_PARAM_LIST = ["--", "-.", ":", (0, (3, 6))]
LS = {-1: "-"}   # reference → solid
for _i, _tok in enumerate(sorted(theta_orders_param, reverse=True)):
    LS[int(_tok)] = _LS_PARAM_LIST[_i] if _i < len(_LS_PARAM_LIST) else ":"

def _lbl(hr):
    θ = hr / H0_L_RATIO - 1
    return f"θ = {θ:+.1f}  (h₀/L = {hr:.4f})"

def _theta_lbl(tok, k_max):
    if tok == -1:
        return "reference"
    diff = k_max - tok
    return "param. (max θ-order)" if diff == 0 else f"param. (max − {diff})"

from matplotlib.lines import Line2D
style_handles = [
    Line2D([0], [0], color="gray", ls=LS[tok], lw=2,
           label=_theta_lbl(tok, theta_max))
    for tok in [-1] + list(sorted(theta_orders_param, reverse=True))
]

# ---------------------------------------------------------------------------
# Helper: select rows, compute shift column
# ---------------------------------------------------------------------------
def select(h_ratio, model, theta_order):
    tok = int(theta_order)
    sub = df[
        np.isclose(df.h_ratio, h_ratio) &
        (df.model == model) &
        (df.theta_order == tok)
    ].copy()
    if sub.empty:
        return sub
    hr = round(h_ratio, 3)
    sub["shift"] = sub["omega"] - _omega0[(hr, model, tok)]
    return sub

# ---------------------------------------------------------------------------
# Figure 1: Ω/ω₀ vs physical amplitude  (absolute, all cases)
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(9, 6))
for k, hr in enumerate(h_ratios):
    color = COLORS[k]
    ref = select(hr, "reference", -1)
    if not ref.empty:
        ax1.plot(ref["omega"] / ω0_base, ref["amplitude"],
                 lw=2, color=color, ls=LS[-1], label=_lbl(hr))
    for tok in sorted(theta_orders_param, reverse=True):
        param = select(hr, "parametric", tok)
        if not param.empty:
            ax1.plot(param["omega"] / ω0_base, param["amplitude"],
                     lw=1.5, color=color, ls=LS[tok])

ax1.add_artist(ax1.legend(loc="upper right", title="Arch height"))
ax1.legend(handles=style_handles, loc="lower right")
ax1.set_xlabel("Backbone frequency  Ω / ω₀")
ax1.set_ylabel("Max transverse displacement at midpoint  (mm)")
ax1.set_title("Backbone curves — absolute frequency (normalised)")
ax1.set_ylim(bottom=0)
fig1.tight_layout()
fig1.savefig(OUT_DIR / "backbone_absolute.png", dpi=150)
print(f"Saved → {OUT_DIR / 'backbone_absolute.png'}")
if not SHOW:
    plt.close(fig1)

# ---------------------------------------------------------------------------
# Figure 2: (Ω − ω₀)/ω₀ vs physical amplitude  (shift, all cases)
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 6))
for k, hr in enumerate(h_ratios):
    color = COLORS[k]
    ref = select(hr, "reference", -1)
    if not ref.empty:
        ax2.plot(ref["shift"] / ω0_base, ref["amplitude"],
                 lw=2, color=color, ls=LS[-1], label=_lbl(hr))
    for tok in sorted(theta_orders_param, reverse=True):
        param = select(hr, "parametric", tok)
        if not param.empty:
            ax2.plot(param["shift"] / ω0_base, param["amplitude"],
                     lw=1.5, color=color, ls=LS[tok])

ax2.add_artist(ax2.legend(loc="upper right", title="Arch height"))
ax2.legend(handles=style_handles, loc="lower right")
ax2.axvline(0, color="black", lw=0.6, ls=":")
ax2.set_xlabel("Nonlinear frequency shift  (Ω − ω₀) / ω₀")
ax2.set_ylabel("Max transverse displacement at midpoint  (mm)")
ax2.set_title("Backbone shift  —  parametric ROM vs reference")
ax2.set_ylim(bottom=0)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "backbone_shift.png", dpi=150)
print(f"Saved → {OUT_DIR / 'backbone_shift.png'}")
if not SHOW:
    plt.close(fig2)

# ---------------------------------------------------------------------------
# Figure 3: per-case overlay  (one file per h₀/L)
# ---------------------------------------------------------------------------
for hr in h_ratios:
    ref = select(hr, "reference", -1)
    if ref.empty and all(select(hr, "parametric", tok).empty for tok in theta_orders_param):
        continue

    fig, ax = plt.subplots(figsize=(7, 5))
    if not ref.empty:
        ax.plot(ref["shift"] / ω0_base, ref["amplitude"],
                lw=2, color="black", ls=LS[-1], label=_theta_lbl(-1, theta_max))
    for tok in sorted(theta_orders_param, reverse=True):
        param = select(hr, "parametric", tok)
        if not param.empty:
            ax.plot(param["shift"] / ω0_base, param["amplitude"],
                    lw=1.5, color="tab:red", ls=LS[tok],
                    label=_theta_lbl(tok, theta_max))
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("Nonlinear frequency shift  (Ω − ω₀) / ω₀")
    ax.set_ylabel("Max transverse displacement at midpoint  (mm)")
    ax.set_title(f"Backbone  —  {_lbl(hr)}")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fname = f"backbone_h{hr:.3f}.png"
    fig.savefig(OUT_DIR / fname, dpi=150)
    print(f"Saved → {OUT_DIR / fname}")
    if not SHOW:
        plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: linear eigenfrequency ω₀/ω₀_base vs arch height h₀/L
# ---------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(7, 5))

ref_m   = metrics[metrics.model == "reference"].sort_values("h_ratio")
hr_ref  = ref_m["h_ratio"].values
ax4.plot(hr_ref, ref_m["omega0"].values / ω0_base,
         "o", color="black", ls=LS[-1], lw=2, ms=6, label=_theta_lbl(-1, theta_max))

for tok in sorted(theta_orders_param, reverse=True):
    pm     = metrics[(metrics.model == "parametric") & (metrics.theta_order == tok)].sort_values("h_ratio")
    hr_p   = pm["h_ratio"].values
    ax4.plot(hr_p, pm["omega0"].values / ω0_base,
             "s", color="tab:red", ls=LS[tok], lw=1.8, ms=4,
             label=_theta_lbl(tok, theta_max))

ax4.set_xlabel("Arch height ratio  h₀/L")
ax4.set_ylabel("Linear eigenfrequency  ω₀ / ω₀_base")
ax4.set_title("Eigenfrequency vs arch height  (normalised)")
ax4.legend()
ax4.set_xlim(0, max(hr_ref) * 1.05)
ax4.set_ylim(bottom=0)
fig4.tight_layout()
fig4.savefig(OUT_DIR / "eigenfreq_vs_h.png", dpi=150)
print(f"Saved → {OUT_DIR / 'eigenfreq_vs_h.png'}")
if not SHOW:
    plt.close(fig4)

# ---------------------------------------------------------------------------
if SHOW:
    plt.show()
