"""
plot_eigenfreq_vs_h.py
======================
Linear eigenfrequency vs arch rise h₀.

Left axis:   ω₀(h₀) / ω₀_base (normalised, range 0.8–1.3).
             Reference ROMs shown as discrete markers connected by a line.
             Parametric ROM shown as a smooth dense curve per θ-truncation order.
Right axis:  |ω₀_param − ω₀_ref| / ω₀_ref  (absolute relative error, grey squares)
             evaluated at the discrete arch heights where reference data exist.

Input:   ../../results/backbone/metrics.csv   (from validation/backbone.jl)
Output:  ../../results/backbone/eigenfreq_vs_h.png

Usage:
    python validation/plots/plot_eigenfreq_vs_h.py [--no-show]
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

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
H0_L_RATIO = float(re.search(r'\bh0_L_ratio\s*=\s*([\d.e+\-]+)',
                              (_ROOT / "config.jl").read_text()).group(1))
OUT_DIR = _ROOT / "results" / "backbone"
OUT_DIR.mkdir(parents=True, exist_ok=True)
_mcsv = OUT_DIR / "metrics.csv"
if not _mcsv.exists():
    sys.exit(f"Error: {_mcsv} not found.  Run validation/backbone.jl first.")

# ── data ──────────────────────────────────────────────────────────────────────
metrics = pd.read_csv(_mcsv)
metrics["z_order"] = metrics["z_order"].astype(int)

z_orders_param = sorted(
    metrics[metrics.model == "parametric"]["z_order"].unique().astype(int)
)
z_max = max(z_orders_param)
ω0_base = float(
    metrics[np.isclose(metrics.h_ratio, H0_L_RATIO) & (metrics.model == "reference")
            ]["omega0"].iloc[0]
)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12,
                     "axes.labelsize": 13, "axes.titlesize": 13,
                     "legend.fontsize": 10, "lines.linewidth": 2})

LS = {-1: "-"}
for _i, _t in enumerate(sorted(z_orders_param, reverse=True)):
    LS[int(_t)] = ("-", "--", ":", "-.")[_i] if _i < 3 else ":"

def _z_lbl(tok, k_max):
    return "reference" if tok == -1 else (
        f"param. (z-order {k_max})" if k_max == tok else f"param. (z-order {tok})"
    )

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
axr = ax.twinx()

ref_m = metrics[metrics.model == "reference"].sort_values("h_ratio")
h_mm = ref_m["h_ratio"].values * 1000.0
#ax.plot(h_mm, ref_m["omega0"].values / ω0_base,
#        "x--", color="black", lw=2, ms=6, label=_z_lbl(-1, z_max))

print(z_orders_param)

for tok in [11]: #sorted(z_orders_param, reverse=True):
    # Dense smooth parametric curve
    pm_d = (metrics[(metrics.model == "parametric_dense") & (metrics.z_order == tok)]
            .sort_values("h_ratio"))
    if not pm_d.empty:
        ax.plot(pm_d["h_ratio"].values * 1000.0, pm_d["omega0"].values / ω0_base,
                ls=LS[tok], color="tab:red", lw=1.8, label=_z_lbl(tok, z_max))

for tok in [11, 7, 3]: #sorted(z_orders_param, reverse=True):

    # Absolute relative error at discrete reference points
    pm_disc = (metrics[(metrics.model == "parametric") & (metrics.z_order == tok)]
               .sort_values("h_ratio"))
    err_hmm, err_vals = [], []
    for _, prow in pm_disc.iterrows():
        rr = ref_m[np.isclose(ref_m.h_ratio, prow.h_ratio)]
        if not rr.empty:
            err_hmm.append(prow.h_ratio * 1000.0)
            err_vals.append((prow.omega0 - rr["omega0"].iloc[0]) / rr["omega0"].iloc[0])
    if err_hmm:
        # Subtract the value at θ=0 so the error is zero at the training point
        baseline = next((v for h, v in zip(err_hmm, err_vals)
                         if np.isclose(h, H0_L_RATIO * 1000.0)), 0.0)
        err_vals = [v - baseline for v in err_vals]
        axr.plot(err_hmm, err_vals, "s", ls=LS[tok],
                 color="blue", lw=1.2, ms=0, alpha=1.00)

axr.set_ylabel("(ω₀_param − ω₀_ref) / ω₀_ref  −  offset(θ=0)", color="gray")
axr.tick_params(axis="y", labelcolor="gray")

#ax.axvline(H0_L_RATIO * 1000.0, color="gray", lw=0.8, ls="--", alpha=0.5)
ax.plot([0, H0_L_RATIO * 1000.0], [1.0, 1.0], color="gray", lw=0.8, ls="--", alpha=0.5)
ax.plot([H0_L_RATIO * 1000.0, H0_L_RATIO * 1000.0], [0.0, 1.0], color="gray", lw=0.8, ls="--", alpha=0.5)
#ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)

ax.set_xticks([i for i in range(0, int(h_mm.max()) + 1)])

ax.set_xlabel("Arch rise  h₀  (mm)")
ax.set_ylabel("Linear eigenfrequency  ω₀ / ω₀_base")
ax.set_title("Eigenfrequency vs arch rise (normalised)")
ax.legend(loc="upper left")
ax.set_xlim(0, h_mm.max())
ax.set_ylim(0.85, 1.3)

fig.tight_layout()
out = OUT_DIR / "eigenfreq_vs_h.png"
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
if SHOW:
    plt.show()
