"""
Ferrite benchmark result visualisations for MORFE.jl.

Usage:
    python plot_ferrite_results.py

Auto-discovers all result folders matching beam_h27_*_degree*_* in the same
directory as this script and writes 12 PNG figures next to it.
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.polynomial import Polynomial

# ---------------------------------------------------------------------------
# Paths & regex
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent
PLOTS_DIR = RESULTS_DIR / "plots_ferrite"
FOLDER_RE = re.compile(r"^beam_h27_(.+)_degree(\d+)_(\w+)$")
SUMMARY_RE = re.compile(r"^(\w+)\s*=\s*(.+)$", re.MULTILINE)

_SUMMARY_NUMERIC = {
    "FOM", "ROM", "N_EXT", "max_degree", "monomials",
    "eig_time_s", "eig_bytes", "eig_gctime_s",
    "solve_time_s", "solve_bytes", "solve_gctime_s", "total_time_s",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _parse_summary(path: Path) -> dict:
    text = path.read_text()
    raw = {m.group(1): m.group(2).strip() for m in SUMMARY_RE.finditer(text)}
    return {k: (float(v) if k in _SUMMARY_NUMERIC else v) for k, v in raw.items()}


def _tag(df: pd.DataFrame, mesh_grid: str, degree: int, fom: int, ts: str) -> None:
    df["mesh_grid"] = mesh_grid
    df["degree"] = degree
    df["FOM"] = fom
    df["timestamp"] = ts


def load_all() -> list[dict]:
    """Return list of run dicts sorted by (degree, FOM)."""
    runs = []
    for folder in RESULTS_DIR.iterdir():
        if not folder.is_dir():
            continue
        m = FOLDER_RE.match(folder.name)
        if m is None:
            continue
        mesh_grid, degree_str, timestamp = m.group(1), m.group(2), m.group(3)
        degree = int(degree_str)
        summary = _parse_summary(folder / "summary.txt")
        fom = int(summary["FOM"])

        df_order = pd.read_csv(folder / "benchmark_per_order.csv")
        df_mono = pd.read_csv(folder / "benchmark_per_monomial.csv")
        _tag(df_order, mesh_grid, degree, fom, timestamp)
        _tag(df_mono, mesh_grid, degree, fom, timestamp)

        run = dict(
            mesh_grid=mesh_grid,
            degree=degree,
            FOM=fom,
            timestamp=timestamp,
            folder=folder,
            summary=summary,
            df_order=df_order,
            df_mono=df_mono,
        )
        _add_derived_metrics(run)
        runs.append(run)

    runs.sort(key=lambda r: (r["degree"], r["FOM"]))
    return runs


def _parse_exponents(series: pd.Series) -> pd.DataFrame:
    """Parse 'a_b_c_d' strings into integer columns exp_0, exp_1, ..."""
    split = series.str.split("_", expand=True).astype(int)
    split.columns = [f"exp_{i}" for i in range(split.shape[1])]
    return split


def _mesh_dims(mesh_grid: str) -> tuple:
    """Parse '40x8x8' → (nx, ny, nz)."""
    return tuple(int(x) for x in mesh_grid.split("x"))


def _add_derived_metrics(run: dict) -> None:
    """Attach solver-informed cost metrics to a run dict in-place.

    Cost model (KLU numeric refactorization dominates per-monomial solve):
      - RHS assembly:  cost ∝ N_elems  (element-local quadrature)
      - Linear solve:  cost ∝ FOM × bw²  (banded LU, bw = cross-section DOFs)
    """
    nx, ny, nz = _mesh_dims(run["mesh_grid"])
    run["n_elems"]        = nx * ny * nz
    run["bw"]             = 3 * (2 * ny + 1) * (2 * nz + 1)
    run["fact_work"]      = run["FOM"] * run["bw"] ** 2
    run["nnz_proxy"]      = run["FOM"] * run["bw"]
    run["n_monomials"]    = int(run["summary"]["monomials"])
    run["rhs_min"]        = run["df_order"]["rhs_time_s"].sum() / 60
    run["solve_min"]      = run["df_order"]["solve_time_s"].sum() / 60
    run["rhs_per_mono"]   = run["rhs_min"]   / run["n_monomials"]
    run["solve_per_mono"] = run["solve_min"] / run["n_monomials"]


# ---------------------------------------------------------------------------
# Colour & style helpers
# ---------------------------------------------------------------------------


def _fom_palette(foms) -> dict:
    unique = sorted(set(foms))
    colors = sns.color_palette("viridis", len(unique))
    return {fom: colors[i] for i, fom in enumerate(unique)}


def _run_label(r: dict) -> str:
    return f"FOM={r['FOM']:,}  deg={r['degree']}"


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def _order_vlines(ax, x_ends, orders, x_data_start):
    """Draw vertical order-boundary lines and top labels (shared helper)."""
    for x_end in x_ends[:-1]:
        ax.axvline(x=x_end + 0.5, color="grey", ls="--", lw=0.8, alpha=0.4)
    for i, (order, x_end) in enumerate(zip(orders, x_ends)):
        x_start = x_ends[i - 1] if i > 0 else x_data_start
        ax.text(
            (x_start + x_end) / 2, 1.01, f"ord {order}",
            fontsize=7, color="grey", ha="center", va="bottom",
            transform=ax.get_xaxis_transform(),
        )


# ---------------------------------------------------------------------------
# Plot 01 — Solve time scaling with mesh size (log-log, one line per order)
# ---------------------------------------------------------------------------


def plot_01_scaling(runs: list[dict]) -> None:
    runs_sorted = sorted(runs, key=lambda r: r["FOM"])
    all_orders = sorted({o for r in runs_sorted for o in r["df_order"]["order"].unique()})
    foms = np.array([r["FOM"] for r in runs_sorted], dtype=float)

    cmap = plt.get_cmap("plasma", len(all_orders) + 2)
    colors = [cmap(i + 1) for i in range(len(all_orders))]

    fig, ax = plt.subplots(figsize=(12, 7))

    for ci, order in enumerate(all_orders):
        order_rows = []
        for r in runs_sorted:
            idx = r["df_order"].set_index("order")
            if order in idx.index:
                order_rows.append((r["fact_work"], float(idx.loc[order, "order_total_time_s"]) / 60))
        if not order_rows:
            continue
        foms_o    = np.array([x[0] for x in order_rows], dtype=float)
        times_min = np.array([x[1] for x in order_rows])
        c = colors[ci]
        ax.scatter(foms_o, times_min, color=c, s=50, zorder=5)
        ax.plot(foms_o, times_min, color=c, lw=2, label=f"order {order}")

        if len(foms_o) >= 2:
            log_f, log_t = np.log(foms_o), np.log(times_min)
            alpha, log_C = np.polyfit(log_f, log_t, 1)
            C = np.exp(log_C)
            ss_res = np.sum((log_t - (alpha * log_f + log_C)) ** 2)
            ss_tot = np.sum((log_t - log_t.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            fom_s = np.logspace(np.log10(foms_o[0]), np.log10(foms_o[-1]), 300)
            ax.plot(fom_s, C * fom_s ** alpha, color=c, lw=1.2, ls="--", alpha=0.75)
            ax.annotate(
                f"{C:.2e}·x^{alpha:.2f}  R²={r2:.4f}",
                xy=(foms_o[-1], times_min[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=7.5, color=c, va="center",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FOM × bw²  (KLU factorisation work)")
    ax.set_ylabel("Order total time (min)")
    ax.set_title("Solve time scaling — one line per polynomial order\n"
                 "(dots = data, dashed = power-law fit  C·(FOM·bw²)^α)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    _save(fig, "plot_01_scaling.png")


# ---------------------------------------------------------------------------
# Plot 02 — High-level time breakdown (stacked horizontal bars)
# ---------------------------------------------------------------------------


def plot_02_time_breakdown_summary(runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, max(4, len(runs) * 1.1)))

    labels = [_run_label(r) for r in runs]
    eig_t = np.array([r["summary"]["eig_time_s"] / 60 for r in runs])
    solve_t = np.array([r["summary"]["solve_time_s"] / 60 for r in runs])
    gc_t = np.array([r["summary"]["solve_gctime_s"] / 60 for r in runs])
    rhs_t = np.array([r["df_order"]["rhs_time_s"].sum() / 60 for r in runs])
    lin_t = np.array([r["df_order"]["solve_time_s"].sum() / 60 for r in runs])
    order_total = np.array([r["df_order"]["order_total_time_s"].sum() / 60 for r in runs])
    overhead_t = np.maximum(solve_t - gc_t - order_total, 0.0)

    y = np.arange(len(runs))
    left = np.zeros(len(runs))
    segments = [
        (eig_t,      "#4c72b0", "Eigensolver"),
        (rhs_t,      "#dd8452", "RHS assembly"),
        (lin_t,      "#55a868", "Linear solve"),
        (overhead_t, "#8172b2", "Loop overhead"),
        (gc_t,       "#c44e52", "GC pause"),
    ]
    for values, color, label in segments:
        ax.barh(y, values, left=left, label=label, color=color)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (min)")
    ax.set_title("Detailed time breakdown per run")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.set_xlim(0)
    _save(fig, "plot_02_time_breakdown_summary.png")


# ---------------------------------------------------------------------------
# Plot 03 — Per-order computation time (fill + line, all meshes)
# ---------------------------------------------------------------------------


def plot_03_per_order_time(runs: list[dict]) -> None:
    pal = _fom_palette([r["FOM"] for r in runs])
    fig, ax = plt.subplots(figsize=(12, 5))

    for r in runs:
        df = r["df_order"].sort_values("order")
        orders = df["order"].values
        total = df["order_total_time_s"].values / 60
        rhs = df["rhs_time_s"].values / 60
        c = pal[r["FOM"]]
        ax.fill_between(orders, total, alpha=0.18, color=c)
        ax.plot(orders, total, color=c, lw=2, marker="o", ms=5,
                label=_run_label(r))
        ax.plot(orders, rhs, color=c, lw=1.2, ls="--", alpha=0.7)

    # Legend entries for solid vs dashed meaning
    solid_h = mlines.Line2D([], [], color="grey", lw=2, label="Total time (solid)")
    dash_h = mlines.Line2D([], [], color="grey", lw=1.2, ls="--", label="RHS assembly (dashed)")
    fom_handles = [mpatches.Patch(color=pal[r["FOM"]], label=_run_label(r)) for r in runs]
    ax.legend(handles=fom_handles + [solid_h, dash_h],
              fontsize=8, ncol=2, loc="upper left", framealpha=0.9)

    ax.set_xlabel("Polynomial order")
    ax.set_ylabel("Time (min)")
    ax.set_ylim(0)
    ax.set_title("Per-order computation time\n"
                 "(solid = total, dashed = RHS assembly, filled area per mesh)")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "plot_03_per_order_time.png")


# ---------------------------------------------------------------------------
# Plot 04 — Per-order memory allocation (log scale, lines + markers)
# ---------------------------------------------------------------------------


def plot_04_per_order_memory(runs: list[dict]) -> None:
    pal = _fom_palette([r["FOM"] for r in runs])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in runs:
        df = r["df_order"].sort_values("order")
        c = pal[r["FOM"]]
        lbl = _run_label(r)
        axes[0].plot(df["order"], df["rhs_alloc_bytes"] / 1e9,
                     color=c, lw=2, marker="o", ms=5, label=lbl)
        axes[1].plot(df["order"], df["solve_alloc_bytes"] / 1e9,
                     color=c, lw=2, marker="o", ms=5, label=lbl)

    for ax, title in zip(axes, ["RHS allocation (GB)", "Solve allocation (GB)"]):
        ax.set_yscale("log")
        ax.set_xlabel("Polynomial order")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", which="both", ls=":", alpha=0.4)

    fig.suptitle("Memory allocation per polynomial order")
    fig.tight_layout()
    _save(fig, "plot_04_per_order_memory.png")


# ---------------------------------------------------------------------------
# Plot 05 — Live heap memory over polynomial orders (fill + line)
# ---------------------------------------------------------------------------


def plot_05_live_memory(runs: list[dict]) -> None:
    pal = _fom_palette([r["FOM"] for r in runs])
    fig, ax = plt.subplots(figsize=(10, 5))

    for r in runs:
        df = r["df_order"].sort_values("order")
        orders = df["order"].values
        mem = df["mem_live_bytes"].values / 1e6
        c = pal[r["FOM"]]
        ax.fill_between(orders, mem, alpha=0.18, color=c)
        ax.plot(orders, mem, color=c, lw=2, marker="o", ms=5, label=_run_label(r))

    ax.set_xlabel("Polynomial order")
    ax.set_ylabel("Live heap memory (MB)")
    ax.set_ylim(0)
    ax.set_title("Live heap memory at end of each polynomial order")
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "plot_05_live_memory.png")


# ---------------------------------------------------------------------------
# Plot 06 — Cumulative solve time (fill + cubic fit at order endpoints)
# ---------------------------------------------------------------------------


def plot_06_cumulative_time(runs: list[dict]) -> None:
    pal = _fom_palette([r["FOM"] for r in runs])
    fig, ax = plt.subplots(figsize=(13, 6))

    order_end_xs: list[float] = []
    orders_all: list[int] = []

    for r in runs:
        df = r["df_mono"].sort_values(["order", "monomial_idx"])
        x = df["monomial_idx"].values.astype(float)
        y = df["cumul_time_s"].values / 60
        c = pal[r["FOM"]]

        order_ends = (
            df.groupby("order")
            .agg(x_end=("monomial_idx", "last"), y_end=("cumul_time_s", "last"))
            .reset_index()
        )
        x_ends = order_ends["x_end"].values.astype(float)
        y_ends = order_ends["y_end"].values / 60

        poly = Polynomial.fit(x_ends, y_ends, deg=3)
        ss_res = np.sum((y_ends - poly(x_ends)) ** 2)
        ss_tot = np.sum((y_ends - y_ends.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        ax.fill_between(x, y, alpha=0.18, color=c)
        ax.plot(x, y, color=c, lw=2, label=f"{_run_label(r)}  (R²={r2:.4f})")
        x_smooth = np.linspace(x[0], x[-1], 500)
        ax.plot(x_smooth, poly(x_smooth), color=c, lw=1.5, ls="--", alpha=0.85)
        ax.scatter(x_ends, y_ends, color=c, s=30, zorder=5)

        if not order_end_xs:
            order_end_xs = x_ends.tolist()
            orders_all = order_ends["order"].tolist()

    _order_vlines(ax, order_end_xs, orders_all, x[0])

    ax.set_xlabel("Global monomial index")
    ax.set_ylabel("Cumulative solve time (min)")
    ax.set_ylim(0)
    ax.set_xlim(x[0] - 5, x[-1] + 5)
    ax.set_title(
        "Cumulative cohomological solve time\n"
        "(filled area = data, dashed = cubic fit at order endpoints, dots = fit anchors)"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    _save(fig, "plot_06_cumulative_time.png")


# ---------------------------------------------------------------------------
# Plot 07 — Per-monomial time distribution by order (box plots)
# ---------------------------------------------------------------------------


def plot_07_order_time_distribution(runs: list[dict]) -> None:
    pal = _fom_palette([r["FOM"] for r in runs])
    n = len(runs)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.5 * nrows), squeeze=False)

    for idx, r in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        df = r["df_mono"]
        orders = sorted(df["order"].unique())
        data = [df[df["order"] == o]["monomial_total_time_s"].values for o in orders]
        c = pal[r["FOM"]]
        bp = ax.boxplot(
            data, tick_labels=orders, patch_artist=True,
            boxprops=dict(facecolor=(*c[:3], 0.35)),
            medianprops=dict(color=c, lw=2),
            whiskerprops=dict(color=c, lw=1.2),
            capprops=dict(color=c, lw=1.2),
            flierprops=dict(marker=".", ms=3, alpha=0.35, color=c),
        )
        _ = bp
        ax.set_yscale("log")
        ax.set_xlabel("Polynomial order")
        ax.set_ylabel("Monomial total time (s)")
        ax.set_title(_run_label(r))
        ax.grid(axis="y", which="both", ls=":", alpha=0.4)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Distribution of per-monomial computation time by order", y=1.01)
    fig.tight_layout()
    _save(fig, "plot_07_order_time_distribution.png")


# ---------------------------------------------------------------------------
# Plot 08 — RHS time vs solve time scatter (colour = order, shape = mesh)
# ---------------------------------------------------------------------------


def plot_08_rhs_vs_solve_scatter(runs: list[dict]) -> None:
    all_orders = sorted({o for r in runs for o in r["df_mono"]["order"].unique()})
    cmap = plt.get_cmap("plasma", len(all_orders))
    order_idx = {o: i for i, o in enumerate(all_orders)}
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(9, 7))

    for mi, r in enumerate(runs):
        df = r["df_mono"]
        sc = ax.scatter(
            df["rhs_time_s"], df["solve_time_s"],
            c=[order_idx[o] for o in df["order"]],
            cmap=cmap, vmin=-0.5, vmax=len(all_orders) - 0.5,
            marker=markers[mi % len(markers)], s=20, alpha=0.55,
            label=f"FOM={r['FOM']:,}",
        )

    cbar = fig.colorbar(sc, ax=ax, ticks=range(len(all_orders)))
    cbar.set_ticklabels(all_orders)
    cbar.set_label("Polynomial order")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("RHS assembly time (s)")
    ax.set_ylabel("Linear solve time (s)")
    ax.set_title("RHS vs solve time per monomial\n(colour = order, shape = mesh)")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    _save(fig, "plot_08_rhs_vs_solve_scatter.png")


# ---------------------------------------------------------------------------
# Plot 09 — Monomial total degree vs RHS time (scatter)
# ---------------------------------------------------------------------------


def plot_09_complexity_vs_time(runs: list[dict]) -> None:
    pal = _fom_palette([r["FOM"] for r in runs])
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in runs:
        df = r["df_mono"].copy()
        exp_df = _parse_exponents(df["exponents"])
        df["total_degree"] = exp_df.sum(axis=1)
        jitter = rng.uniform(-0.25, 0.25, len(df))
        ax.scatter(
            df["total_degree"] + jitter, df["rhs_time_s"],
            s=20, color=pal[r["FOM"]], alpha=0.45, label=_run_label(r),
        )

    ax.set_yscale("log")
    ax.set_xlabel("Total monomial degree (sum of exponents)")
    ax.set_ylabel("RHS assembly time (s)")
    ax.set_title("Monomial total degree vs RHS assembly time\n"
                 "(jitter added horizontally for legibility)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", which="both", ls=":", alpha=0.4)
    _save(fig, "plot_09_complexity_vs_time.png")


# ---------------------------------------------------------------------------
# Plot 10 — Top-20 slowest monomials per run (horizontal bars)
# ---------------------------------------------------------------------------


def plot_10_top20_slowest(runs: list[dict]) -> None:
    pal = _fom_palette([r["FOM"] for r in runs])
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 7), squeeze=False)

    for idx, r in enumerate(runs):
        ax = axes[0][idx]
        df = r["df_mono"].nlargest(20, "monomial_total_time_s").sort_values(
            "monomial_total_time_s"
        )
        labels = [f"ord={row.order}  [{row.exponents}]" for row in df.itertuples()]
        ax.barh(labels, df["monomial_total_time_s"], color=pal[r["FOM"]], alpha=0.8)
        ax.set_xlabel("Monomial total time (s)")
        ax.set_xlim(0)
        ax.set_title(_run_label(r))
        ax.grid(axis="x", ls=":", alpha=0.4)

    fig.suptitle("Top-20 slowest monomials per run", y=1.01)
    fig.tight_layout()
    _save(fig, "plot_10_top20_slowest.png")


# ---------------------------------------------------------------------------
# Plot 11 — RHS vs solve time fraction per order (100% stacked area)
# ---------------------------------------------------------------------------


def plot_11_rhs_fraction(runs: list[dict]) -> None:
    n = len(runs)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)

    for idx, r in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        df = r["df_order"].sort_values("order")
        total = df["order_total_time_s"].values
        rhs_frac = df["rhs_time_s"].values / total
        solve_frac = df["solve_time_s"].values / total
        x = df["order"].values
        ax.stackplot(
            x, rhs_frac, solve_frac,
            labels=["RHS assembly", "Linear solve"],
            colors=["#dd8452", "#4c72b0"],
        )
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Polynomial order")
        ax.set_ylabel("Fraction of order time")
        ax.set_title(_run_label(r))
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        ax.grid(axis="y", ls=":", alpha=0.4)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("RHS assembly vs linear solve time fraction per order", y=1.01)
    fig.tight_layout()
    _save(fig, "plot_11_rhs_fraction.png")


# ---------------------------------------------------------------------------
# Plot 12 — RHS allocation heatmap: order × monomial position
# ---------------------------------------------------------------------------


def plot_12_allocation_heatmap(runs: list[dict]) -> None:
    n = len(runs)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 4.5 * nrows), squeeze=False)

    for idx, r in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        df = r["df_mono"].copy()
        df["pos_in_order"] = df.groupby("order").cumcount()
        orders = sorted(df["order"].unique())
        max_pos = int(df["pos_in_order"].max()) + 1

        mat = np.full((len(orders), max_pos), np.nan)
        for oi, order in enumerate(orders):
            sub = df[df["order"] == order]
            mat[oi, sub["pos_in_order"].values.astype(int)] = (
                sub["rhs_alloc_bytes"].values.astype(float)
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            log_mat = np.where(np.isnan(mat), np.nan, np.log10(np.maximum(mat, 1.0)))

        im = ax.imshow(
            log_mat, aspect="auto", cmap="YlOrRd",
            vmin=np.nanmin(log_mat), vmax=np.nanmax(log_mat),
            interpolation="nearest",
        )
        ax.set_yticks(range(len(orders)))
        ax.set_yticklabels(orders)
        ax.set_xlabel("Monomial index within order")
        ax.set_ylabel("Polynomial order")
        ax.set_title(_run_label(r))
        plt.colorbar(im, ax=ax, label="log₁₀(RHS alloc bytes)")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("RHS allocation heatmap: order × monomial position", y=1.01)
    fig.tight_layout()
    _save(fig, "plot_12_allocation_heatmap.png")


# ---------------------------------------------------------------------------
# Helpers shared by plots 13 & 14
# ---------------------------------------------------------------------------


def _fixed_order_rows(runs: list[dict], target_order: int) -> list[tuple]:
    """
    For each unique FOM, return exactly one per-order CSV row for target_order.
    When multiple runs share a FOM, the one with the lowest degree is preferred
    (minimises any interaction with later orders).
    Returns list of (run, row_series) sorted by run["fact_work"].
    """
    best: dict[int, dict] = {}
    for r in runs:
        df = r["df_order"]
        if target_order not in df["order"].values:
            continue
        fom = r["FOM"]
        if fom not in best or r["degree"] < best[fom]["degree"]:
            row = df[df["order"] == target_order].iloc[0]
            best[fom] = {"degree": r["degree"], "run": r, "row": row}
    return [(v["run"], v["row"]) for v in sorted(best.values(), key=lambda x: x["run"]["fact_work"])]


def _power_fit_log(x, y):
    """Fit y = C * x^alpha in log space. Returns (C, alpha, R²)."""
    lx, ly = np.log(x), np.log(y)
    alpha, lc = np.polyfit(lx, ly, 1)
    C = np.exp(lc)
    r2 = 1.0 - np.sum((ly - (alpha * lx + lc)) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return C, alpha, r2


# ---------------------------------------------------------------------------
# Plot 13 — Fixed-order time costs vs mesh size (log-log)
# ---------------------------------------------------------------------------


def plot_13_fixed_order_time(runs: list[dict], target_order: int = 5) -> None:
    rows = _fixed_order_rows(runs, target_order)
    if not rows:
        print(f"  No runs contain order {target_order} — skipping plot 13.")
        return

    foms  = np.array([run["fact_work"] for run, _ in rows], dtype=float)
    rhs   = np.array([row["rhs_time_s"]         for _, row in rows]) / 60
    sol   = np.array([row["solve_time_s"]        for _, row in rows]) / 60
    total = rhs + sol

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(foms, rhs,   0,     color="#4c72b0", alpha=0.8, label="RHS assembly")
    ax.fill_between(foms, total, rhs,   color="#55a868", alpha=0.8, label="Linear solve")
    ax.plot(foms, total, color="black", lw=2, marker="o", ms=5, label="Total")

    if len(foms) >= 2:
        C, alpha, r2 = _power_fit_log(foms, total)
        fom_s = np.logspace(np.log10(foms[0]), np.log10(foms[-1]), 300)
        ax.plot(fom_s, C * fom_s ** alpha, color="black", ls="--", lw=1.5, alpha=0.8,
                label=f"Total fit  {C:.2e}·(FOM·bw²)^{alpha:.2f}  R²={r2:.3f}")

    ax.set_ylim(0)
    ax.set_xlabel("FOM × bw²  (KLU factorisation work)")
    ax.set_ylabel(f"Order {target_order} time (min)")
    ax.set_title(f"Time costs at order {target_order} vs FOM × bw²")
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, f"plot_13_order{target_order}_time_vs_fom.png")


# ---------------------------------------------------------------------------
# Plot 14 — Fixed-order memory allocation vs mesh size (log-log)
# ---------------------------------------------------------------------------


def plot_14_fixed_order_memory(runs: list[dict], target_order: int = 5) -> None:
    rows = _fixed_order_rows(runs, target_order)
    if not rows:
        print(f"  No runs contain order {target_order} — skipping plot 14.")
        return

    foms  = np.array([run["fact_work"] for run, _ in rows], dtype=float)
    rhs   = np.array([row["rhs_alloc_bytes"]   for _, row in rows]) / (1024**3)
    sol   = np.array([row["solve_alloc_bytes"]  for _, row in rows]) / (1024**3)
    total = rhs + sol

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(foms, rhs,   0,     color="#4c72b0", alpha=0.8, label="RHS assembly")
    ax.fill_between(foms, total, rhs,   color="#55a868", alpha=0.8, label="Linear solve")
    ax.plot(foms, total, color="black", lw=2, marker="o", ms=5, label="Total")

    if len(foms) >= 2:
        C, alpha, r2 = _power_fit_log(foms, total)
        fom_s = np.logspace(np.log10(foms[0]), np.log10(foms[-1]), 300)
        ax.plot(fom_s, C * fom_s ** alpha, color="black", ls="--", lw=1.5, alpha=0.8,
                label=f"Total fit  {C:.2e}·(FOM·bw²)^{alpha:.2f}  R²={r2:.3f}")

    ax.set_xscale("log")
    ax.set_ylim(0)
    ax.set_xlabel("FOM × bw²  (KLU factorisation work)")
    ax.set_ylabel(f"Order {target_order} allocation (GB)")
    ax.set_title(f"Memory allocation at order {target_order} vs FOM × bw²")
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, f"plot_14_order{target_order}_memory_vs_fom.png")


# ---------------------------------------------------------------------------
# Plot 15 — Solve/RHS cost ratio vs mesh size (log-x, one line per order)
# ---------------------------------------------------------------------------


def plot_15_ratio_solve_rhs(runs: list[dict]) -> None:
    all_orders = sorted({o for r in runs for o in r["df_order"]["order"].unique()})
    cmap = plt.get_cmap("plasma", len(all_orders) + 2)
    colors = [cmap(i + 1) for i in range(len(all_orders))]

    fig, ax = plt.subplots(figsize=(11, 6))

    for ci, order in enumerate(all_orders):
        rows = _fixed_order_rows(runs, order)
        if not rows:
            continue
        foms  = np.array([run["fact_work"] for run, _ in rows], dtype=float)
        t_rat = np.array([row["solve_time_s"] / row["rhs_time_s"] for _, row in rows])
        m_rat = np.array([row["solve_alloc_bytes"] / row["rhs_alloc_bytes"] for _, row in rows])
        c = colors[ci]
        ax.plot(foms, t_rat, color=c, lw=2,   marker="o", ms=5, label=f"order {order}")
        ax.plot(foms, m_rat, color=c, lw=1.5, marker="s", ms=4, ls="--")

    solid_h = mlines.Line2D([], [], color="grey", lw=2,   label="Time ratio (solid)")
    dash_h  = mlines.Line2D([], [], color="grey", lw=1.5, ls="--", label="Memory ratio (dashed)")
    order_handles = [mpatches.Patch(color=colors[i], label=f"order {o}") for i, o in enumerate(all_orders)]
    ax.legend(handles=order_handles + [solid_h, dash_h], fontsize=8, ncol=2)

    ax.set_xscale("log")
    ax.set_xlabel("FOM × bw²  (KLU factorisation work)")
    ax.set_ylabel("Ratio  solve / RHS")
    ax.set_title("Solve-to-RHS cost ratio vs FOM × bw²\n(solid = time, dashed = memory allocation)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "plot_15_ratio_solve_rhs.png")


# ---------------------------------------------------------------------------
# Plot 16 — Time-cost fraction at fixed order vs mesh size (stacked area)
# ---------------------------------------------------------------------------


def plot_16_fraction_vs_fom(runs: list[dict], target_order: int = 5) -> None:
    rows = _fixed_order_rows(runs, target_order)
    if not rows:
        print(f"  No runs contain order {target_order} — skipping plot 16.")
        return

    foms     = np.array([run["fact_work"] for run, _ in rows], dtype=float)
    rhs_frac = np.array([row["rhs_time_s"] / row["order_total_time_s"] for _, row in rows])
    sol_frac = np.array([row["solve_time_s"] / row["order_total_time_s"] for _, row in rows])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(foms, rhs_frac, sol_frac,
                 labels=["RHS assembly", "Linear solve"],
                 colors=["#dd8452", "#4c72b0"])
    ax.set_xscale("log")
    ax.set_xlim(foms[0], foms[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("FOM × bw²  (KLU factorisation work)")
    ax.set_ylabel("Fraction of order time")
    ax.set_title(f"Time-cost fraction at order {target_order} vs FOM × bw²")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, f"plot_16_fraction_order{target_order}_vs_fom.png")


# ---------------------------------------------------------------------------
# Plot 17 — Per-order time (left, log) and memory (right, log) vs mesh size
# ---------------------------------------------------------------------------


def plot_17_time_memory_vs_fom(runs: list[dict]) -> None:
    all_orders = sorted({o for r in runs for o in r["df_order"]["order"].unique()})
    cmap = plt.get_cmap("plasma", len(all_orders) + 2)
    colors = [cmap(i + 1) for i in range(len(all_orders))]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax2 = ax.twinx()

    for ci, order in enumerate(all_orders):
        rows = _fixed_order_rows(runs, order)
        if not rows:
            continue
        foms  = np.array([run["fact_work"] for run, _ in rows], dtype=float)
        times = np.array([row["order_total_time_s"] / 60 for _, row in rows])
        mems  = np.array([(row["rhs_alloc_bytes"] + row["solve_alloc_bytes"]) / 1e9
                          for _, row in rows])
        c = colors[ci]
        ax.plot(foms,  times, color=c, lw=2,   marker="o", ms=5, label=f"order {order}")
        ax2.plot(foms, mems,  color=c, lw=1.5, marker="s", ms=4, ls="--")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xlabel("FOM × bw²  (KLU factorisation work)")
    ax.set_ylabel("Order total time (min)")
    ax2.set_ylabel("Total allocation (GB)", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")

    solid_h = mlines.Line2D([], [], color="grey", lw=2,   label="Time (solid, left axis)")
    dash_h  = mlines.Line2D([], [], color="grey", lw=1.5, ls="--", label="Memory (dashed, right axis)")
    order_handles = [mpatches.Patch(color=colors[i], label=f"order {o}") for i, o in enumerate(all_orders)]
    ax.legend(handles=order_handles + [solid_h, dash_h], fontsize=8, ncol=2, loc="upper left")

    ax.set_title("Per-order total time and allocation vs FOM × bw²")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "plot_17_time_memory_vs_fom.png")


# ---------------------------------------------------------------------------
# Shared helper: degree palette
# ---------------------------------------------------------------------------


def _degree_palette(runs: list[dict]) -> dict:
    degrees = sorted({r["degree"] for r in runs})
    colors = sns.color_palette("tab10", len(degrees))
    return {d: colors[i] for i, d in enumerate(degrees)}


def _annotate_point(ax, x, y, label, color, fontsize=7.5):
    ax.annotate(
        label, xy=(x, y),
        xytext=(5, 3), textcoords="offset points",
        fontsize=fontsize, color=color, va="bottom",
    )


# ---------------------------------------------------------------------------
# Plot 18 — RHS assembly time vs FOM
# ---------------------------------------------------------------------------


def plot_18_rhs_vs_fom(runs: list[dict]) -> None:
    deg_pal = _degree_palette(runs)
    fig, ax = plt.subplots(figsize=(9, 6))

    for degree, grp in sorted(
        {r["degree"]: [] for r in runs}.items()
    ):
        grp = [r for r in runs if r["degree"] == degree]
        grp.sort(key=lambda r: r["FOM"])
        xs = np.array([r["FOM"] for r in grp], dtype=float)
        ys = np.array([r["rhs_min"] for r in grp])
        c = deg_pal[degree]
        ax.scatter(xs, ys, color=c, s=60, zorder=5)
        ax.plot(xs, ys, color=c, lw=1.5, alpha=0.5)

        if len(xs) >= 2:
            log_x, log_y = np.log(xs), np.log(ys)
            alpha, log_C = np.polyfit(log_x, log_y, 1)
            C = np.exp(log_C)
            xs_fit = np.logspace(np.log10(xs[0]), np.log10(xs[-1]), 200)
            ax.plot(xs_fit, C * xs_fit ** alpha, color=c, lw=1.5, ls="--", alpha=0.8,
                    label=f"degree {degree}  fit slope={alpha:.2f}")
        else:
            ax.plot([], [], color=c, label=f"degree {degree}")

        for r in grp:
            _annotate_point(ax, r["FOM"], r["rhs_min"], r["mesh_grid"], c)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FOM (free DOFs)")
    ax.set_ylabel("Total RHS assembly time (min)")
    ax.set_title(
        "RHS assembly time vs FOM\n"
        "(expected slope ≈ 1)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "plot_18_rhs_vs_fom.png")


# ---------------------------------------------------------------------------
# Plot 19 — Solve time vs FOM×bw²
# ---------------------------------------------------------------------------


def plot_19_solve_vs_factwork(runs: list[dict]) -> None:
    deg_pal = _degree_palette(runs)
    fig, ax = plt.subplots(figsize=(9, 6))

    for degree, _ in sorted({r["degree"]: None for r in runs}.items()):
        grp = sorted([r for r in runs if r["degree"] == degree], key=lambda r: r["fact_work"])
        xs = np.array([r["fact_work"] for r in grp], dtype=float)
        ys = np.array([r["solve_min"] for r in grp])
        c = deg_pal[degree]
        ax.scatter(xs, ys, color=c, s=60, zorder=5)
        ax.plot(xs, ys, color=c, lw=1.5, alpha=0.5)

        if len(xs) >= 2:
            log_x, log_y = np.log(xs), np.log(ys)
            alpha, log_C = np.polyfit(log_x, log_y, 1)
            C = np.exp(log_C)
            xs_fit = np.logspace(np.log10(xs[0]), np.log10(xs[-1]), 200)
            ax.plot(xs_fit, C * xs_fit ** alpha, color=c, lw=1.5, ls="--", alpha=0.8,
                    label=f"degree {degree}  slope={alpha:.2f}")
        else:
            ax.plot([], [], color=c, label=f"degree {degree}")

        for r in grp:
            _annotate_point(ax, r["fact_work"], r["solve_min"], r["mesh_grid"], c)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FOM × bw²  (KLU factorisation work)")
    ax.set_ylabel("Total cohomological solve time (min)")
    ax.set_title("Cohomological solve time vs FOM × bw²\n(expected slope ≈ 1)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "plot_19_solve_vs_factwork.png")


# ---------------------------------------------------------------------------
# Plot 20 — Per-monomial cost: solve vs FOM×bw², RHS vs N_elems
# ---------------------------------------------------------------------------


def plot_20_per_monomial_cost(runs: list[dict]) -> None:
    deg_pal = _degree_palette(runs)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    specs = [
        ("fact_work",  "solve_per_mono",
         "FOM × bw²  (factorisation work)",
         "Avg solve time per monomial (min)",
         "Per-monomial solve cost vs FOM×bw²\n(expected slope ≈ 1)"),
        ("FOM",        "rhs_per_mono",
         "FOM (free DOFs)",
         "Avg RHS assembly time per monomial (min)",
         "Per-monomial RHS cost vs FOM\n(expected slope ≈ 1)"),
    ]

    for ax, (xkey, ykey, xlabel, ylabel, title) in zip(axes, specs):
        for degree, _ in sorted({r["degree"]: None for r in runs}.items()):
            grp = sorted([r for r in runs if r["degree"] == degree], key=lambda r: r[xkey])
            xs = np.array([r[xkey] for r in grp], dtype=float)
            ys = np.array([r[ykey] for r in grp])
            c = deg_pal[degree]
            ax.scatter(xs, ys, color=c, s=70, zorder=5, label=f"degree {degree}")
            for r in grp:
                _annotate_point(ax, r[xkey], r[ykey], r["mesh_grid"], c)

        if len(runs) >= 2:
            all_xs = np.array([r[xkey] for r in runs], dtype=float)
            all_ys = np.array([r[ykey] for r in runs])
            valid = (all_xs > 0) & (all_ys > 0)
            if valid.sum() >= 2:
                log_x, log_y = np.log(all_xs[valid]), np.log(all_ys[valid])
                alpha, log_C = np.polyfit(log_x, log_y, 1)
                C = np.exp(log_C)
                xs_fit = np.logspace(np.log10(all_xs[valid].min()), np.log10(all_xs[valid].max()), 200)
                ax.plot(xs_fit, C * xs_fit ** alpha, color="grey", lw=2, ls="--", alpha=0.7,
                        label=f"all-runs fit  slope={alpha:.2f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("Per-monomial cost (normalised by monomial count) isolates the matrix-structure effect")
    fig.tight_layout()
    _save(fig, "plot_20_per_monomial_cost.png")


# ---------------------------------------------------------------------------
# Plot 21 — Mesh space: FOM vs bandwidth with iso-cost contours
# ---------------------------------------------------------------------------


def plot_21_mesh_space_fom_bw(runs: list[dict]) -> None:
    deg_pal = _degree_palette(runs)

    foms      = np.array([r["FOM"]      for r in runs], dtype=float)
    bws       = np.array([r["bw"]       for r in runs], dtype=float)
    solves    = np.array([r["solve_min"] for r in runs])
    max_solve = solves.max()

    fig, ax = plt.subplots(figsize=(10, 7))

    # Iso-cost contours: FOM × bw² = const  →  log(bw) = (log(const) - log(FOM)) / 2
    fom_grid = np.logspace(np.log10(foms.min() * 0.5), np.log10(foms.max() * 2), 300)
    work_vals = np.logspace(
        np.log10((foms * bws ** 2).min() * 0.5),
        np.log10((foms * bws ** 2).max() * 2),
        6,
    )
    for wv in work_vals:
        bw_iso = np.sqrt(wv / fom_grid)
        valid = (bw_iso >= bws.min() * 0.3) & (bw_iso <= bws.max() * 3)
        if valid.sum() > 1:
            ax.plot(fom_grid[valid], bw_iso[valid], color="lightgrey", lw=1.2, ls="--", zorder=1)
            # label at right end of visible segment
            ix = np.where(valid)[0][-1]
            ax.text(
                fom_grid[ix], bw_iso[ix],
                f"  {wv:.1e}",
                fontsize=6.5, color="grey", va="center",
            )

    # Scatter, size ∝ solve time
    for degree, _ in sorted({r["degree"]: None for r in runs}.items()):
        grp = [r for r in runs if r["degree"] == degree]
        c = deg_pal[degree]
        for r in grp:
            sz = 60 + 500 * (r["solve_min"] / max_solve) ** 0.5
            ax.scatter(r["FOM"], r["bw"], color=c, s=sz, zorder=5, alpha=0.85)
            ax.annotate(
                r["mesh_grid"],
                xy=(r["FOM"], r["bw"]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8, color=c,
            )
        ax.scatter([], [], color=c, s=80, label=f"degree {degree}")

    # Size legend
    for ref_min in [1, 10, 100]:
        if ref_min <= solves.max():
            sz = 60 + 500 * (ref_min / max_solve) ** 0.5
            ax.scatter([], [], color="grey", s=sz, label=f"{ref_min} min", alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FOM (free DOFs)")
    ax.set_ylabel("bw = 3(2ny+1)(2nz+1)  [cross-section DOFs]")
    ax.set_title(
        "Mesh space: FOM vs bandwidth\n"
        "Grey dashed = iso-cost lines (FOM·bw²=const);  marker size ∝ √solve_time"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    fig.tight_layout()
    _save(fig, "plot_21_mesh_space_fom_bw.png")


# ---------------------------------------------------------------------------
# Plot 22 — Cost breakdown sorted by FOM×bw²
# ---------------------------------------------------------------------------


def plot_22_cost_breakdown_factwork(runs: list[dict]) -> None:
    runs_sorted = sorted(runs, key=lambda r: r["fact_work"])

    labels = [
        f"{r['mesh_grid']}  FOM={r['FOM']:,}  bw={r['bw']}"
        for r in runs_sorted
    ]
    eig_t     = np.array([r["summary"]["eig_time_s"] / 60 for r in runs_sorted])
    solve_t   = np.array([r["summary"]["solve_time_s"] / 60 for r in runs_sorted])
    gc_t      = np.array([r["summary"]["solve_gctime_s"] / 60 for r in runs_sorted])
    rhs_t     = np.array([r["df_order"]["rhs_time_s"].sum() / 60 for r in runs_sorted])
    lin_t     = np.array([r["df_order"]["solve_time_s"].sum() / 60 for r in runs_sorted])
    order_total = np.array([r["df_order"]["order_total_time_s"].sum() / 60 for r in runs_sorted])
    overhead_t  = np.maximum(solve_t - gc_t - order_total, 0.0)

    y    = np.arange(len(runs_sorted))
    left = np.zeros(len(runs_sorted))
    segments = [
        (eig_t,      "#4c72b0", "Eigensolver"),
        (rhs_t,      "#dd8452", "RHS assembly  (∝ FOM)"),
        (lin_t,      "#55a868", "Linear solve   (∝ FOM·bw²)"),
        (overhead_t, "#8172b2", "Loop overhead"),
        (gc_t,       "#c44e52", "GC pause"),
    ]

    fig, ax = plt.subplots(figsize=(12, max(4, len(runs_sorted) * 1.1)))
    for values, color, label in segments:
        ax.barh(y, values, left=left, label=label, color=color)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (min)")
    ax.set_title(
        "Detailed time breakdown — sorted by FOM×bw² (KLU factorisation work)\n"
        "RHS scales with element count; linear solve scales with FOM·bw²"
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.set_xlim(0)
    fig.tight_layout()
    _save(fig, "plot_22_cost_breakdown_factwork.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    runs = load_all()
    if not runs:
        print("No result folders matching beam_h27_*_degree*_* found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} result folder(s):")
    for r in runs:
        print(f"  {r['folder'].name}  FOM={r['FOM']}  degree={r['degree']}")
    print()

    plt.rcParams.update({
        "figure.dpi": 100,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    plot_01_scaling(runs)
    plot_02_time_breakdown_summary(runs)
    plot_03_per_order_time(runs)
    plot_04_per_order_memory(runs)
    plot_05_live_memory(runs)
    plot_06_cumulative_time(runs)
    plot_07_order_time_distribution(runs)
    plot_08_rhs_vs_solve_scatter(runs)
    plot_09_complexity_vs_time(runs)
    plot_10_top20_slowest(runs)
    plot_11_rhs_fraction(runs)
    plot_12_allocation_heatmap(runs)
    plot_13_fixed_order_time(runs, target_order=5)
    plot_14_fixed_order_memory(runs, target_order=5)
    plot_15_ratio_solve_rhs(runs)
    plot_16_fraction_vs_fom(runs, target_order=5)
    plot_17_time_memory_vs_fom(runs)
    plot_18_rhs_vs_fom(runs)
    plot_19_solve_vs_factwork(runs)
    plot_20_per_monomial_cost(runs)
    plot_21_mesh_space_fom_bw(runs)
    plot_22_cost_breakdown_factwork(runs)

    print(f"\nAll 22 plots written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
