#!/usr/bin/env python3
"""
generate_matlab.py

Generate COCO-format MATLAB vector-field functions for the Kármán vortex street ROM
from a MORFE R_coefficients.csv.

Produces:
  vec_fields_karman.m      – f(x, par) = [x1dot, x2dot]
  vec_fields_karman_DFDX.m – df/dx(x,par)  (2×2×N Jacobian)

Convention (same as generate_dynamical_system.py):
  z = x(1,:) + 1j*x(2,:)   (z1 = Re(z) + i*Im(z))
  z_conj = x(1,:) - 1j*x(2,:)   (z2 = conj(z1))
  eta = par(1,:)              (eta_prime = 1/Re - 1/Re0)

Usage:
    python generate_matlab.py <csv_path> [--output-dir <dir>] [--re0 <float>]
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict


THRESHOLD = 1e-25

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Edit these to run without command-line arguments
# ---------------------------------------------------------------------------
CSV_PATH = _HERE / "../results/Re49.00_ord7/data/R_coefficients.csv"
OUTPUT_DIR = None   # None → same folder as CSV_PATH; or e.g. "/path/to/output"
RE0 = 49.03  # expansion Reynolds number (used in comments only)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_matlab_coeff(coeff, threshold=THRESHOLD):
    """Return MATLAB coefficient string: (re+im*1j), im*1j, re, etc."""
    c_re, c_im = coeff.real, coeff.imag
    re_sig = abs(c_re) >= threshold
    im_sig = abs(c_im) >= threshold
    if not re_sig and not im_sig:
        return None
    if re_sig and not im_sig:
        return f"{c_re:.15e}"
    if not re_sig and im_sig:
        return f"{c_im:.15e}*1j"
    im_sign = "+" if c_im >= 0 else "-"
    return f"({c_re:.15e}{im_sign}{abs(c_im):.15e}*1j)"


def make_mono_matlab(a, b, c):
    """Return MATLAB monomial string in z, z_conj, eta_p{c}."""
    factors = []
    if a == 1:
        factors.append("z")
    elif a > 1:
        factors.append(f"z.^{a}")
    if b == 1:
        factors.append("z_conj")
    elif b > 1:
        factors.append(f"z_conj.^{b}")
    if c == 1:
        factors.append("eta_p{1}")
    elif c > 1:
        factors.append(f"eta_p{{{c}}}")
    return ".*".join(factors) if factors else ""


def make_term(coeff, a, b, c, threshold=THRESHOLD):
    """Return a MATLAB term string: '+/-coeff.*mono' or None."""
    cs = make_matlab_coeff(coeff, threshold)
    if cs is None:
        return None
    mono = make_mono_matlab(a, b, c)
    if mono:
        return f"{cs}.*{mono}"
    return cs


def build_block(terms, lhs, indent="    "):
    """Emit  'lhs = lhs ...\n    + term1 ...\n    + term2;'"""
    if not terms:
        return ""
    lines = [f"{indent}{lhs} = {lhs} ..."]
    for i, t in enumerate(terms):
        sep = ";" if i == len(terms) - 1 else " ..."
        lines.append(f"{indent}    + {t}{sep}")
    return "\n".join(lines)


def build_poly(by_c, varname, indent="    ", threshold=THRESHOLD):
    """Build 'varname = c0 + c1.*eta_p{1} + ...;' assignment."""
    terms = []
    for c_pow in sorted(by_c.keys()):
        val = by_c[c_pow]
        if abs(val) < threshold:
            continue
        if c_pow == 0:
            terms.append(f"{val:.15e}")
        else:
            terms.append(f"( {val:+.15e}).*eta_p{{{c_pow}}}")
    if not terms:
        return f"{indent}{varname} = 0.0;"
    lines = [f"{indent}{varname} = {terms[0]}           ..."]
    for i, t in enumerate(terms[1:], 1):
        sep = ";" if i == len(terms) - 1 else "  ..."
        lines.append(f"{indent}    + {t}{sep}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(csv_path, output_dir, re0=None, max_ord=None, threshold=THRESHOLD):
    csv_path = Path(csv_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Parse CSV
    # ------------------------------------------------------------------
    rows = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            a = int(row["exp_1"])
            b = int(row["exp_2"])
            c = int(row["exp_3"])
            coeff = complex(float(row["R1_re"]), float(row["R1_im"]))
            if abs(coeff) > threshold:
                rows.append((a, b, c, coeff))

    # ------------------------------------------------------------------
    # 2. Bounds and linear extraction
    # ------------------------------------------------------------------
    max_c = max((r[2] for r in rows), default=0)
    nl_rows = sorted(
        [(a, b, c, coeff) for (a, b, c, coeff) in rows if a + b >= 2],
        key=lambda r: (r[0] + r[1], r[2], -r[0]),
    )
    z_orders = sorted(set(r[0] + r[1] for r in nl_rows))

    delta_by_c = defaultdict(float)
    omega_by_c = defaultdict(float)
    for (a, b, c, coeff) in rows:
        if a == 1 and b == 0:
            delta_by_c[c] += coeff.real
            omega_by_c[c] += coeff.imag

    groups = {}
    for (a, b, c, coeff) in nl_rows:
        groups.setdefault(a + b, []).append((a, b, c, coeff))

    delta0_val = delta_by_c.get(0, 0.0)
    omega0_val = omega_by_c.get(0, 0.0)

    re0_str = f"{re0}" if re0 is not None else "unknown"
    source_comment = f"% Source: {csv_path}"

    # ------------------------------------------------------------------
    # 3. eta_p preamble (shared by both files)
    # ------------------------------------------------------------------
    eta_preamble = [
        "eta = par(1,:);",
        "",
        f"eta_p = cell(1,{max_c});",
        "eta_p{1} = eta;",
        f"for k = 2:{max_c}",
        "    eta_p{k} = eta_p{k-1} .* eta;",
        "end",
    ]

    delta_assignment = build_poly(delta_by_c, "delta0")
    omega_assignment = build_poly(omega_by_c, "omega0")

    # ------------------------------------------------------------------
    # 4. vec_fields_karman.m
    # ------------------------------------------------------------------
    max_ord_line = f"MAX_ORD = par(2,:);" if max_ord is not None else ""

    nl_body_lines = []
    for z_ord in z_orders:
        g_rows = groups[z_ord]
        terms = []
        for (a, b, c, coeff) in g_rows:
            t = make_term(coeff, a, b, c, threshold)
            if t:
                terms.append(t)
        blk = build_block(terms, "ydot", indent="    ")
        if blk:
            nl_body_lines.append("")
            if max_ord is not None:
                nl_body_lines.append(f"if MAX_ORD >= {z_ord}")
                nl_body_lines.append(blk)
                nl_body_lines.append("end")
            else:
                nl_body_lines.append(blk)

    vf_lines = [
        f"function y = vec_fields_karman(x, par)",
        f"% 2-D Kármán vortex street ROM  –  COCO format",
        f"%",
        f"% AUTO-GENERATED by generate_matlab.py",
        source_comment,
        f"%",
        f"% States:  x(1,:) = Re(z),  x(2,:) = Im(z)",
        f"% Par:     par(1,:) = eta_prime  (= 1/Re - 1/Re0, Re0 = {re0_str})",
        "",
    ] + ([max_ord_line, ""] if max_ord_line else []) + eta_preamble + [
        "",
        delta_assignment,
        "",
        omega_assignment,
        "",
        "z = x(1,:) + 1j.*x(2,:);",
        "z_conj = x(1,:) - 1j.*x(2,:);",
        "",
        "ydot = (delta0 + 1j.*omega0).*z;",
    ] + nl_body_lines + [
        "",
        "y(1,:) = real(ydot);",
        "y(2,:) = imag(ydot);",
        "",
        "end",
        "",
    ]

    vf_path = output_dir / "vec_fields_karman.m"
    with open(vf_path, "w") as fh:
        fh.write("\n".join(vf_lines))
    print(f"Written  -> {vf_path}")

    # ------------------------------------------------------------------
    # 5. vec_fields_karman_DFDX.m
    # ------------------------------------------------------------------
    jac_nl_lines = []
    for z_ord in z_orders:
        g_rows = groups[z_ord]
        dz_terms, dzc_terms = [], []
        for (a, b, c, coeff) in g_rows:
            if a >= 1:
                t = make_term(a * coeff, a - 1, b, c, threshold)
                if t:
                    dz_terms.append(t)
            if b >= 1:
                t = make_term(b * coeff, a, b - 1, c, threshold)
                if t:
                    dzc_terms.append(t)
        blk_dz  = build_block(dz_terms,  "dy_dz",      indent="    ")
        blk_dzc = build_block(dzc_terms, "dy_dz_conj", indent="    ")
        if blk_dz or blk_dzc:
            jac_nl_lines.append("")
            if max_ord is not None:
                jac_nl_lines.append(f"if MAX_ORD >= {z_ord}")
                if blk_dz:
                    jac_nl_lines.append(blk_dz)
                if blk_dzc:
                    jac_nl_lines.append(blk_dzc)
                jac_nl_lines.append("end")
            else:
                jac_nl_lines.append(f"% Order {z_ord}")
                if blk_dz:
                    jac_nl_lines.append(blk_dz)
                if blk_dzc:
                    jac_nl_lines.append(blk_dzc)

    dfdx_lines = [
        "function J = vec_fields_karman_DFDX(x, par)",
        "% State Jacobian df/dx for the Kármán ROM  –  COCO format",
        "%",
        "% AUTO-GENERATED by generate_matlab.py",
        source_comment,
        "%",
        "% Mirrors jacobian_nonlinear_term() + _wirtinger_to_jacobian() in",
        "% dynamical_system.py, augmented with the linear part.",
        "%",
        "% States:  x(1,:) = Re(z),  x(2,:) = Im(z)",
        f"% Par:     par(1,:) = eta_prime  (= 1/Re - 1/Re0, Re0 = {re0_str})",
        "% Output:  J  (2 x 2 x N)",
        "",
    ] + ([max_ord_line, ""] if max_ord_line else []) + eta_preamble + [
        "",
        delta_assignment,
        "",
        omega_assignment,
        "",
        "z = x(1,:) + 1j.*x(2,:);",
        "z_conj = x(1,:) - 1j.*x(2,:);",
        "",
        "dy_dz = zeros(size(z));",
        "dy_dz_conj = zeros(size(z));",
    ] + jac_nl_lines + [
        "",
        "% Add linear part: d[(delta0 + i*omega0)*z]/dz = delta0 + i*omega0,  d/dz_conj = 0",
        "dy_dz = dy_dz + (delta0 + 1j.*omega0);",
        "",
        "% Wirtinger → real Jacobian  (_wirtinger_to_jacobian in dynamical_system.py)",
        "%   dy_dx0 = dy_dz + dy_dz_conj",
        "%   dy_dx1 = 1j * (dy_dz - dy_dz_conj)",
        "%   J = [[Re(dy_dx0), Re(dy_dx1)],",
        "%        [Im(dy_dx0), Im(dy_dx1)]]",
        "dy_dx0 = dy_dz + dy_dz_conj;",
        "dy_dx1 = 1j .* (dy_dz - dy_dz_conj);",
        "",
        "N = numel(dy_dx0);",
        "J = zeros(2, 2, N);",
        "J(1, 1, :) = real(dy_dx0);",
        "J(1, 2, :) = real(dy_dx1);",
        "J(2, 1, :) = imag(dy_dx0);",
        "J(2, 2, :) = imag(dy_dx1);",
        "",
        "end",
        "",
    ]

    dfdx_path = output_dir / "vec_fields_karman_DFDX.m"
    with open(dfdx_path, "w") as fh:
        fh.write("\n".join(dfdx_lines))
    print(f"Written  -> {dfdx_path}")

    # ------------------------------------------------------------------
    # 6. lift_karman.m  (auto-detected from sibling L_coefficients.csv)
    # ------------------------------------------------------------------
    lift_csv = csv_path.parent / "L_coefficients.csv"
    if lift_csv.exists():
        generate_lift(lift_csv, output_dir, re0=re0, max_ord=max_ord, threshold=threshold)

    print(f"  delta0(eta=0) = {delta0_val:.10f}")
    print(f"  omega0(eta=0) = {omega0_val:.10f}")
    print(f"  z-orders: {z_orders}  (max_z_ord={max(z_orders) if z_orders else 0})")
    print(f"  max_c (eta polynomial degree): {max_c}")
    print(f"  nonlinear CSV rows: {len(nl_rows)}")


def generate_lift(lift_csv_path, output_dir, re0=None, max_ord=None, threshold=THRESHOLD):
    """Generate lift_karman.m from L_coefficients.csv."""
    lift_csv_path = Path(lift_csv_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse CSV: skip all (a==0, b==0) rows — those are L0 and pure-η′ terms
    # which must be zero by symmetry (cylinder geometry) and are numerical noise.
    all_rows = []
    with open(lift_csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            a = int(row["exp_1"])
            b = int(row["exp_2"])
            c = int(row["exp_3"])
            coeff = complex(float(row["L_re"]), float(row["L_im"]))
            if abs(coeff) > threshold:
                all_rows.append((a, b, c, coeff))

    max_c = max((r[2] for r in all_rows), default=1)
    rows = [(a, b, c, coeff) for (a, b, c, coeff) in all_rows if not (a == 0 and b == 0)]
    nl_rows = sorted(rows, key=lambda r: (r[0] + r[1], r[2], -r[0]))
    z_orders = sorted(set(r[0] + r[1] for r in nl_rows))
    groups = {}
    for (a, b, c, coeff) in nl_rows:
        groups.setdefault(a + b, []).append((a, b, c, coeff))

    max_ord_line = f"MAX_ORD = par(2,:);" if max_ord is not None else ""

    eta_preamble = [
        "eta = par(1,:);",
        "",
        f"eta_p = cell(1,{max_c});",
        "eta_p{1} = eta;",
        f"for k = 2:{max_c}",
        "    eta_p{k} = eta_p{k-1} .* eta;",
        "end",
    ]

    nl_body_lines = []
    for z_ord in z_orders:
        terms = []
        for (a, b, c, coeff) in groups[z_ord]:
            t = make_term(coeff, a, b, c, threshold)
            if t:
                terms.append(t)
        blk = build_block(terms, "lift_force", indent="    ")
        if blk:
            nl_body_lines.append("")
            if max_ord is not None:
                nl_body_lines.append(f"if MAX_ORD >= {z_ord}")
                nl_body_lines.append(blk)
                nl_body_lines.append("end")
            else:
                nl_body_lines.append(blk)

    re0_str = f"{re0}" if re0 is not None else "unknown"
    source_comment = f"% Source: {lift_csv_path}"

    lift_lines = [
        "function lift_force = lift_karman(x, par)",
        "% Pressure-lift force polynomial for the Kármán vortex street ROM  –  COCO format",
        "%",
        "% AUTO-GENERATED by generate_matlab.py",
        source_comment,
        "%",
        "% Evaluates  lift_force(z, eta) = l^T * W(z),  where l is the pressure-lift",
        "% weight vector assembled on the cylinder surface.  Returns the pressure lift",
        "% force (NOT the coefficient); multiply by -2/(U_mean^2 * D) for Cl.",
        "%",
        "% States:  x(1,:) = Re(z),  x(2,:) = Im(z)",
        f"% Par:     par(1,:) = eta_prime  (= 1/Re - 1/Re0, Re0 = {re0_str})",
        "% Output:  lift_force  (pressure lift force, real scalar or array)",
        "",
    ] + ([max_ord_line, ""] if max_ord_line else []) + eta_preamble + [
        "",
        "z = x(1,:) + 1j.*x(2,:);",
        "z_conj = x(1,:) - 1j.*x(2,:);",
        "",
        "lift_force = 0.0;",
    ] + nl_body_lines + [
        "",
        "lift_force = real(lift_force);",
        "",
        "end",
        "",
    ]

    lift_path = output_dir / "lift_karman.m"
    with open(lift_path, "w") as fh:
        fh.write("\n".join(lift_lines))
    print(f"Written  -> {lift_path}")
    print(f"  nonlinear rows: {len(rows)},  z-orders: {z_orders}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate COCO MATLAB functions from MORFE R_coefficients.csv (Kármán vortex street)",
    )
    parser.add_argument("csv_path", nargs="?", default=CSV_PATH,
                        help="Path to R_coefficients.csv")
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory (default: same folder as csv_path)",
    )
    parser.add_argument(
        "--re0",
        type=float,
        default=RE0,
        help="Expansion Reynolds number Re0 (for comments only)",
    )
    parser.add_argument(
        "--max-ord",
        type=int,
        default=None,
        help="Polynomial truncation order (emitted as MAX_ORD constant; enables if/end blocks)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-25,
        help="Drop coefficients with |val| below this (default: 1e-25)",
    )
    args = parser.parse_args()
    out = args.output_dir or str(Path(args.csv_path).parent)
    generate(args.csv_path, out, re0=args.re0, max_ord=args.max_ord, threshold=args.threshold)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()  # called from main.jl or CLI with explicit args
    else:
        out = OUTPUT_DIR or str(Path(CSV_PATH).parent)
        generate(CSV_PATH, out, re0=RE0)
