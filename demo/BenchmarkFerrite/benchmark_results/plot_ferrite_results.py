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

        runs.append(dict(
            mesh_grid=mesh_grid,
            degree=degree,
            FOM=fom,
            timestamp=timestamp,
            folder=folder,
            summary=summary,
            df_order=df_order,
            df_mono=df_mono,
        ))

    runs.sort(key=lambda r: (r["degree"], r["FOM"]))
    return runs


def _parse_exponents(series: pd.Series) -> pd.DataFrame:
    """Parse 'a_b_c_d' strings into integer columns exp_0, exp_1, ..."""
    split = series.str.split("_", expand=True).astype(int)
    split.columns = [f"exp_{i}" for i in range(split.shape[1])]
    return split


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
                order_rows.append((r["FOM"], float(idx.loc[order, "order_total_time_s"]) / 60))
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
    ax.set_xlabel("FOM size (DOFs)")
    ax.set_ylabel("Order total time (min)")
    ax.set_title("Solve time scaling with mesh size — one line per polynomial order\n"
                 "(dots = data, dashed = power-law fit  C·FOM^α)")
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
    Returns list of (fom, row_series) sorted by FOM.
    """
    best: dict[int, dict] = {}
    for r in runs:
        df = r["df_order"]
        if target_order not in df["order"].values:
            continue
        fom = r["FOM"]
        if fom not in best or r["degree"] < best[fom]["degree"]:
            row = df[df["order"] == target_order].iloc[0]
            best[fom] = {"degree": r["degree"], "fom": fom, "row": row}
    return [(v["fom"], v["row"]) for v in sorted(best.values(), key=lambda x: x["fom"])]


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

    foms  = np.array([fom for fom, _ in rows], dtype=float)
    rhs   = np.array([row["rhs_time_s"]         for _, row in rows]) / 60
    sol   = np.array([row["solve_time_s"]        for _, row in rows]) / 60
    total = rhs + sol

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(foms, rhs,   0,     color="#4c72b0", alpha=0.8, label="RHS assembly")
    ax.fill_between(foms, total, rhs,   color="#55a868", alpha=0.8, label="Linear solve")
    ax.plot(foms, total, color="black", lw=2, marker="o", ms=5, label="Total")

    if len(foms) >= 2:
        C, alpha, r2 = _power_fit_log(foms, total)
        fom_s = np.linspace(foms[0], foms[-1], 300)
        ax.plot(fom_s, C * fom_s ** alpha, color="black", ls="--", lw=1.5, alpha=0.8,
                label=f"Total fit  {C:.2e}·FOM^{alpha:.2f}  R²={r2:.3f}")

    ax.set_ylim(0)
    ax.set_xlabel("FOM size (DOFs)")
    ax.set_ylabel(f"Order {target_order} time (min)")
    ax.set_title(f"Time costs at order {target_order} vs mesh size")
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

    foms  = np.array([fom for fom, _ in rows], dtype=float)
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
                label=f"Total fit  {C:.2e}·FOM^{alpha:.2f}  R²={r2:.3f}")

    ax.set_xscale("log")
    ax.set_ylim(0)
    ax.set_xlabel("FOM size (DOFs)")
    ax.set_ylabel(f"Order {target_order} allocation (GB)")
    ax.set_title(f"Memory allocation at order {target_order} vs mesh size")
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, f"plot_14_order{target_order}_memory_vs_fom.png")


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

    print(f"\nAll 14 plots written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
