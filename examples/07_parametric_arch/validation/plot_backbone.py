"""
plot_backbone.py
================
Runner — executes all backbone plot scripts in sequence.

Individual scripts live in validation/plots/ and can be run independently:
    plot_backbone_absolute.py           — Ω/ω₀ vs amplitude, all arch heights
    plot_backbone_shift.py              — (Ω − ω₀)/ω₀ vs amplitude, all arch heights
    plot_backbone_per_case.py           — per-arch overlay (reference vs parametric)
    plot_eigenfreq_vs_h.py              — ω₀/ω₀_base vs arch rise with error on right axis
    plot_backbone_absolute_reduced.py   — Ω/ω₀ vs modal amplitude |z| (param full r, ref capped)
    plot_backbone_shift_reduced.py      — (Ω − ω₀)/ω₀ vs modal amplitude |z| (same)
    plot_backbone_per_case_reduced.py   — per-arch overlay in reduced domain (same)

Usage:
    python validation/plot_backbone.py             # save + open interactive windows
    python validation/plot_backbone.py --no-show   # headless, save only
"""

import subprocess
import sys
from pathlib import Path

_PLOTS = Path(__file__).parent / "plots"

scripts = [
    "plot_backbone_absolute.py",
    "plot_backbone_shift.py",
    "plot_backbone_per_case.py",
    "plot_eigenfreq_vs_h.py",
    "plot_backbone_absolute_reduced.py",
    "plot_backbone_shift_reduced.py",
    "plot_backbone_per_case_reduced.py",
]

for s in scripts:
    print(f"\n{'─' * 60}")
    print(f"  {s}")
    print(f"{'─' * 60}")
    result = subprocess.run([sys.executable, str(_PLOTS / s)] + sys.argv[1:])
    if result.returncode != 0:
        sys.exit(result.returncode)
