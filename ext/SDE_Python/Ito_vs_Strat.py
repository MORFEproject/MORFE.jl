"""Itô vs Stratonovich: the same path, two evaluation points, two integrals.

Run with no arguments for the matplotlib preview. Run with --latex to write the
plain-text tables consumed by the pgfplots figures in latex/, which are the
version meant for the paper: there LaTeX typesets every label, tick and legend
entry in whatever font the surrounding document uses.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
np.random.seed(42)
T = 4.0                          # total time
dt_fine = 0.0001                 # fine step: the reference path drawn underneath
scale_fine = 100                 # the integration grid is this factor coarser
N_fine = int(T / dt_fine)
dt_coarse = dt_fine * scale_fine
N_coarse = N_fine // scale_fine

n_zoom = 10                      # coarse steps shown in the evaluation-point panel
t_zoom = n_zoom * dt_coarse

# Plotting resolution for the dense curves in the LaTeX export. The integrals are
# always computed on the full dt_fine path; this only thins what LaTeX has to draw.
decimate = 10

# --- Fine Brownian path (visual underlay + reference integrals) ---
dW_fine = np.sqrt(dt_fine) * np.random.randn(N_fine)
W_fine = np.zeros(N_fine + 1)
W_fine[1:] = np.cumsum(dW_fine)
t_fine = np.arange(N_fine + 1) * dt_fine

# --- Coarse integration grid: every scale_fine-th point of the same path ---
W_coarse = W_fine[::scale_fine]
dW_coarse = np.diff(W_coarse)    # dW_coarse[k] = W(t_{k+1}) - W(t_k)
t_coarse = np.arange(N_coarse + 1) * dt_coarse

# Pre-allocate coarse arrays
Ito_int = np.zeros(N_coarse + 1)
Strat_int = np.zeros(N_coarse + 1)
Ito_iterates = np.zeros(N_coarse)
Strat_iterates = np.zeros(N_coarse)

# Compute the integrals on the coarse grid. Written as an explicit loop because the
# only difference between the two conventions is the iterate picked on each step.
for k in range(N_coarse):
    Wk = W_coarse[k]             # W at the left end of the step
    Wk1 = W_coarse[k + 1]        # W at the right end
    dWk = dW_coarse[k]           # increment over the step

    # Iterates
    Ito_iterates[k] = Wk                 # left endpoint
    Strat_iterates[k] = 0.5 * (Wk + Wk1)  # midpoint rule (chord average)

    # Cumulative integrals
    Ito_int[k + 1] = Ito_int[k] + Ito_iterates[k] * dWk
    Strat_int[k + 1] = Strat_int[k] + Strat_iterates[k] * dWk

# Same two integrals on the fine grid, as a convergence reference
Ito_fine = np.zeros(N_fine + 1)
Strat_fine = np.zeros(N_fine + 1)
Ito_fine[1:] = np.cumsum(W_fine[:-1] * dW_fine)
Strat_fine[1:] = np.cumsum(0.5 * (W_fine[:-1] + W_fine[1:]) * dW_fine)

# Difference and theory
diff = Strat_int - Ito_int
diff_fine = Strat_fine - Ito_fine
theory_diff = 0.5 * t_coarse

# Closed forms on the coarse grid
Ito_theory = 0.5 * (W_coarse**2 - t_coarse)
Strat_theory = 0.5 * W_coarse**2


def preview():
    """Screen version: one 2x2 matplotlib figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Brownian path: fine underlay, coarse polyline on top
    ax1 = axes[0, 0]
    ax1.plot(t_fine, W_fine, '-', color='0.2', lw=1.0, alpha=0.3,
             label=f'fine path, dt = {dt_fine:g}')
    ax1.plot(t_coarse, W_coarse, 'k-', lw=1.2, ms=2.5,
             label=f'coarse grid, dt = {dt_coarse:g}')
    ax1.set_title(f'Brownian path W(t): integration grid is {scale_fine}x coarser')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('W(t)')
    ax1.legend()
    ax1.grid(True)

    # 2. Evaluation points, zoomed to the first few coarse steps
    ax2 = axes[0, 1]
    mask = t_fine <= t_zoom
    ax2.plot(t_fine[mask], W_fine[mask], '-', color='0.2', lw=1.0, alpha=0.3,
             label='fine path')
    ax2.plot(t_coarse[:n_zoom + 1], W_coarse[:n_zoom + 1], 'k-', lw=1.0, alpha=0.6,
             label='coarse step chords')
    ax2.scatter(t_coarse[:n_zoom], Ito_iterates[:n_zoom], s=55, color='blue',
                zorder=3, label='Itô iterate: $W(t_k)$')
    ax2.scatter(t_coarse[:n_zoom] + dt_coarse / 2, Strat_iterates[:n_zoom], s=55,
                color='red', marker='s', zorder=3,
                label=r'Strat iterate: $\frac{1}{2}[W(t_k)+W(t_{k+1})]$')
    # The Stratonovich iterate is the midpoint of the chord, not a value of the path:
    # mark where it sits relative to the two endpoints it averages.
    for k in range(n_zoom):
        ax2.plot([t_coarse[k], t_coarse[k] + dt_coarse / 2],
                 [Ito_iterates[k], Strat_iterates[k]], 'k:', lw=0.8, alpha=0.5)
    ax2.set_title(f'Evaluation point per coarse step (first {n_zoom} steps)')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('W at evaluation point')
    ax2.legend()
    ax2.grid(True)

    # 3. Cumulative integrals. The Stratonovich sum reproduces the chain rule exactly
    # on any grid; the Itô sum is offset from its closed form by 0.5*(sum dW^2 - t).
    ax3 = axes[1, 0]
    ax3.plot(t_fine, Ito_fine, 'b-', lw=3, alpha=0.25, label='Itô, fine grid')
    ax3.plot(t_fine, Strat_fine, 'r-', lw=3, alpha=0.25, label='Stratonovich, fine grid')
    ax3.plot(t_coarse, Ito_int, 'b-', lw=1.8, label='Itô, coarse grid')
    ax3.plot(t_coarse, Strat_int, 'r-', lw=1.8, label='Stratonovich, coarse grid')
    ax3.plot(t_coarse, Ito_theory, 'k--', lw=0.9, alpha=0.8,
             dashes=(5, 4), label=r'Itô theory: $(W^2-t)/2$')
    ax3.plot(t_coarse, Strat_theory, 'k--', lw=0.9, alpha=0.8,
             dashes=(5, 4), label=r'Strat theory: $W^2/2$')
    ax3.set_title('Cumulative integrals of W dW')
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Integral value')
    ax3.legend(ncol=2, fontsize=8, loc='upper left')
    ax3.grid(True)

    # 4. Cumulative difference (Strat - Ito), i.e. half the quadratic variation
    ax4 = axes[1, 1]
    ax4.fill_between(t_coarse, 0, diff, alpha=0.2, color='green')
    ax4.plot(t_fine, diff_fine, 'g-', lw=3, alpha=0.25, label='S(t) - I(t), fine grid')
    ax4.plot(t_coarse, diff, 'g-', lw=1.4, label='S(t) - I(t), coarse grid')
    ax4.plot(t_coarse, theory_diff, 'k--', lw=1.5, label='Theory: t/2')
    ax4.set_title('Accumulated effect of the iterate choice')
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('S(t) - I(t)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


def _save(path, columns, header):
    """Write a pgfplots table: one header line of column names, then the rows."""
    np.savetxt(path, np.column_stack(columns), fmt='%.6g', header=header, comments='')


def export_latex_data(outdir):
    """Write the .dat tables read by latex/ito_path.tex and latex/ito_integrals.tex."""
    os.makedirs(outdir, exist_ok=True)
    d = decimate
    zoom = t_fine <= t_zoom

    # Figure (a): full path, fine underlay thinned to dt_fine*decimate
    _save(os.path.join(outdir, 'path_fine.dat'), (t_fine[::d], W_fine[::d]), 't W')
    _save(os.path.join(outdir, 'path_coarse.dat'), (t_coarse, W_coarse), 't W')

    # Figure (b): the zoom window keeps full dt_fine resolution, since the sub-step
    # wiggle of the path is exactly what the panel is about.
    _save(os.path.join(outdir, 'zoom_fine.dat'), (t_fine[zoom], W_fine[zoom]), 't W')
    _save(os.path.join(outdir, 'zoom_coarse.dat'),
          (t_coarse[:n_zoom + 1], W_coarse[:n_zoom + 1]), 't W')
    _save(os.path.join(outdir, 'iterates.dat'),
          (t_coarse[:n_zoom], Ito_iterates[:n_zoom],
           t_coarse[:n_zoom] + dt_coarse / 2, Strat_iterates[:n_zoom]),
          'tI WI tS WS')

    # One file of disjoint two-point segments. The `nan nan` rows are the breaks,
    # read back on the LaTeX side with `unbounded coords=jump`.
    with open(os.path.join(outdir, 'connectors.dat'), 'w') as f:
        f.write('t W\n')
        for k in range(n_zoom):
            if k:
                f.write('nan nan\n')
            f.write(f'{t_coarse[k]:.6g} {Ito_iterates[k]:.6g}\n')
            f.write(f'{t_coarse[k] + dt_coarse / 2:.6g} {Strat_iterates[k]:.6g}\n')

    # Figures (c) and (d)
    _save(os.path.join(outdir, 'integrals_fine.dat'),
          (t_fine[::d], Ito_fine[::d], Strat_fine[::d]), 't I S')
    _save(os.path.join(outdir, 'integrals_coarse.dat'),
          (t_coarse, Ito_int, Strat_int, Ito_theory, Strat_theory), 't I S Ith Sth')
    _save(os.path.join(outdir, 'difference_fine.dat'),
          (t_fine[::d], diff_fine[::d]), 't d')
    _save(os.path.join(outdir, 'difference_coarse.dat'),
          (t_coarse, diff, theory_diff), 't d th')

    for name in sorted(os.listdir(outdir)):
        if name.endswith('.dat'):
            path = os.path.join(outdir, name)
            with open(path) as f:
                rows = sum(1 for line in f if line.strip()) - 1
            print(f"  {name:24s} {rows:5d} rows")


def print_summary():
    print(f"Coarse grid (dt = {dt_coarse:g}, {N_coarse} steps)")
    print(f"  Itô integral            = {Ito_int[-1]:.4f}")
    print(f"  Stratonovich integral   = {Strat_int[-1]:.4f}")
    print(f"  W(T)^2 / 2              = {0.5 * W_coarse[-1]**2:.4f}"
          f"   (Strat mismatch {abs(Strat_int[-1] - 0.5 * W_coarse[-1]**2):.2e})")
    print(f"  S - I                   = {diff[-1]:.4f}")
    print(f"  quadratic variation     = {np.sum(dW_coarse**2):.4f}")
    print(f"Fine grid (dt = {dt_fine:g}, {N_fine} steps)")
    print(f"  Itô integral            = {Ito_fine[-1]:.4f}")
    print(f"  Stratonovich integral   = {Strat_fine[-1]:.4f}")
    print(f"  S - I                   = {diff_fine[-1]:.4f}")
    print(f"  quadratic variation     = {np.sum(dW_fine**2):.4f}")
    print(f"Theory: S - I = T/2       = {T / 2:.4f},  quadratic variation = T = {T:.4f}")


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--latex', action='store_true',
                        help='write the pgfplots data tables instead of showing the figure')
    parser.add_argument('--outdir', default=os.path.join(here, 'latex', 'data'),
                        help='where --latex writes its .dat tables')
    args = parser.parse_args()

    if args.latex:
        print(f"Writing pgfplots tables to {args.outdir}")
        export_latex_data(args.outdir)
    else:
        preview()
    print_summary()
