#!/usr/bin/env python3
"""Generate the Kármán tutorial-card thumbnail (karman_thumb.svg).

A freeze-frame of the left of the hero figure (FIG·01): the vorticity of

	w(x) = w_base(x) + eps * Re[ w_mode(x) ]      (phase theta = 0)

rendered from the shipped VTK output, cropped to the cylinder + near wake and sized to the
16/9 card thumbnail. Colours match the site diverging palette. Rerun after regenerating the
VTUs:

	python3 website/assets/karman/generate_thumb.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
import meshio
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PV = REPO / "examples/05_karman_vortex_street/results/paraview"

BG = "#07070b"
EPS_REL = 0.35   # mode amplitude relative to base-flow vorticity scale (matches the animation)

# 16/9 crop of the left of the domain (channel height 0.41 m); cylinder centre x=0.2, D=0.1.
XLO, XHI = 0.03, 0.03 + 0.41 * 16 / 9   # width = 0.729 m
YLO, YHI = 0.0, 0.41
CYL_X, CYL_Y, CYL_R = 0.2, 0.2, 0.05

# Site diverging colormap: julia-blue -> page background -> julia-red (as in FIG·01).
CMAP = LinearSegmentedColormap.from_list("karman", [
	(0.00, "#7ea2ff"), (0.22, "#4063d8"), (0.44, "#0b0b12"),
	(0.50, "#07070b"),
	(0.56, "#0b0b12"), (0.78, "#cb3c33"), (1.00, "#ff9d8a"),
])

b = meshio.read(PV / "base_flow.vtu")
m = meshio.read(PV / "eigenmodes/eigenmode_001.vtu")

x, y = b.points[:, 0], b.points[:, 1]
tris = b.cells[0].data
wb = b.point_data["vorticity"][:, 0]
wre = m.point_data["vorticity_Re"][:, 0]

eps = EPS_REL * np.percentile(np.abs(wb), 99) / np.percentile(np.abs(wre), 99)
w = wb + eps * wre
lim = np.percentile(np.abs(w), 99)
levels = np.linspace(-lim, lim, 21)

tri = mtri.Triangulation(x, y, tris)

fig, ax = plt.subplots(figsize=(7.2, 7.2 * 9 / 16))
ax.tricontourf(tri, np.clip(w, -lim, lim), levels=levels, cmap=CMAP,
			   extend="both", antialiased=True)
ax.add_patch(plt.Circle((CYL_X, CYL_Y), CYL_R, facecolor="#14141c",
						edgecolor="#6e6e7e", lw=1.0, zorder=5))
ax.set_xlim(XLO, XHI)
ax.set_ylim(YLO, YHI)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

out = HERE / "karman_thumb.svg"
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0)
plt.close(fig)
print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
