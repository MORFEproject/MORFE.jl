#!/usr/bin/env python3
"""
generate_dynamical_system.py

Generate dynamical_system.py for the parametric arch backbone HBM
from a MORFE R_coefficients.csv.

Convention (Realification.jl: z = x + i*y):
  z1 = x0 + i*x1   (CSV slot 1, exp_1)
  z2 = x0 - i*x1   (CSV slot 2, exp_2, conjugate of z1)
  theta             (CSV slot 3, exp_3, frozen parameter)

Uses the R1 component (R1_re + i*R1_im), evaluated directly in complex:
  z      = x0 + 1j*x1
  z_conj = x0 - 1j*x1   # == conj(z)
  y      = R1(z, z_conj, theta)
  x0dot  = y.real
  x1dot  = y.imag

Jacobian via the Wirtinger chain rule:
  dy/dx0 = dy/dz + dy/dz_conj
  dy/dx1 = 1j * (dy/dz - dy/dz_conj)

Truncation parameters (set at construction time in the generated class):
  z_truncation_order    : include nonlinear terms up to this z-degree
  theta_truncation_order: include theta-polynomial terms up to this power;
                          self.theta_p entries beyond this are kept as 0.0
                          so higher-order terms vanish without extra branching.

Usage:
    python generate_dynamical_system.py <csv_path> [--output <path>]
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict


THRESHOLD = 1e-25


# ---------------------------------------------------------------------------
# Code-string helpers
# ---------------------------------------------------------------------------

def make_z_term(coeff, a, b, c, threshold=THRESHOLD):
    """
    Return '+/-|coeff|*[1j*][z**a]*[z_conj**b]*[self.theta_p[c]]'.
    Leading sign is always explicit.  Returns None when |coeff| < threshold.
    Uses self.theta_p[c] for theta factors (supports truncation via zero-padding).
    """
    c_re, c_im = coeff.real, coeff.imag
    re_sig = abs(c_re) >= threshold
    im_sig = abs(c_im) >= threshold
    if not re_sig and not im_sig:
        return None

    # Spatial / parameter monomial string
    factors = []
    if a == 1:
        factors.append("z")
    elif a > 1:
        factors.append(f"z**{a}")
    if b == 1:
        factors.append("z_conj")
    elif b > 1:
        factors.append(f"z_conj**{b}")
    if c == 1:
        factors.append("self.theta_p[1]")
    elif c > 1:
        factors.append(f"self.theta_p[{c}]")
    mono = "*".join(factors)

    def signed(val, unit=""):
        s = "+" if val >= 0 else "-"
        u = f"*{unit}" if unit else ""
        return f"{s}{abs(val):.15e}{u}"

    if not re_sig:
        prefix = signed(c_im, "1j")
    elif not im_sig:
        prefix = signed(c_re)
    else:
        im_sign = "+" if c_im >= 0 else "-"
        prefix = f"+({c_re:.15e}{im_sign}{abs(c_im):.15e}*1j)"

    if mono:
        return f"{prefix}*{mono}"
    return prefix


def add_block(terms, lhs, indent="        "):
    """
    Emit '{lhs} += term1+\\term2...'.
    Returns empty string (skip) when terms list is empty.
    """
    if not terms:
        return ""
    joined = ("+\\\n").join(f"{indent}{t}" for t in terms)
    return f"{indent}{lhs} += \\\n{joined}"


def early_return_nl(indent):
    """Return self._pack_real_imag(y) at the given indent level."""
    return f"{indent}return self._pack_real_imag(y)"


def early_return_jac(indent):
    """Return self._wirtinger_to_jacobian(dy_dz, dy_dz_conj) at the given indent level."""
    return f"{indent}return self._wirtinger_to_jacobian(dy_dz, dy_dz_conj)"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(csv_path, output_path, threshold=THRESHOLD):
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
    # 2. Separate linear (a+b == 1) and nonlinear (a+b >= 2) rows
    # ------------------------------------------------------------------
    nl_rows = sorted(
        [(a, b, c, coeff) for (a, b, c, coeff) in rows if a + b >= 2],
        key=lambda r: (r[0] + r[1], r[2], -r[0]),
    )

    # ------------------------------------------------------------------
    # 3. Compute CSV-level bounds
    # ------------------------------------------------------------------
    max_c = max((r[2] for r in rows), default=0)
    z_orders = sorted(set(r[0] + r[1] for r in nl_rows))
    max_z_ord = z_orders[-1] if z_orders else 1

    # ------------------------------------------------------------------
    # 4. Extract omega0(theta) from linear (1,0,c) rows
    #    R1 linear coefficient is i*omega, so omega = Im(R1)
    # ------------------------------------------------------------------
    omega_by_c = defaultdict(float)
    for (a, b, c, coeff) in rows:
        if a == 1 and b == 0:
            omega_by_c[c] += coeff.imag

    omega_terms = []
    for c_pow in sorted(omega_by_c.keys()):
        val = omega_by_c[c_pow]
        if abs(val) < threshold:
            continue
        if c_pow == 0:
            omega_terms.append(f"{val:.15e}")
        else:
            omega_terms.append(f"{val:.15e}*self.theta_p[{c_pow}]")
    omega_str = " + \\\n            ".join(omega_terms) if omega_terms else "0.0"
    omega0_val = omega_by_c.get(0, 0.0)

    # ------------------------------------------------------------------
    # 5. Group nl_rows by z-order (a+b)
    # ------------------------------------------------------------------
    groups = {}
    for (a, b, c, coeff) in nl_rows:
        groups.setdefault(a + b, []).append((a, b, c, coeff))

    def group_terms(g_rows):
        """Build (y_terms, dy_dz_terms, dy_dz_conj_terms) for a z-order group."""
        y_t, dz_t, dzc_t = [], [], []
        for (a, b, c, coeff) in g_rows:
            t = make_z_term(coeff, a, b, c, threshold)
            if t:
                y_t.append(t)
            if a >= 1:
                t = make_z_term(a * coeff, a - 1, b, c, threshold)
                if t:
                    dz_t.append(t)
            if b >= 1:
                t = make_z_term(b * coeff, a, b - 1, c, threshold)
                if t:
                    dzc_t.append(t)
        return y_t, dz_t, dzc_t

    # ------------------------------------------------------------------
    # 6. Build nonlinear_term method body
    # ------------------------------------------------------------------
    first_z = z_orders[0] if z_orders else None
    nl_lines = [
        "        x0 = state[..., 0:1, :]",
        "        x1 = state[..., 1:2, :]",
        "        z = x0 + 1j*x1",
        "        z_conj = x0 - 1j*x1",
        "        y = np.zeros_like(z)",
        "",
    ]
    if first_z is not None:
        nl_lines += [
            f"        if self.z_truncation_order < {first_z}:",
            early_return_nl("            "),
            "",
        ]
    for i, z_ord in enumerate(z_orders):
        y_t, _, _ = group_terms(groups[z_ord])
        blk = add_block(y_t, "y")
        if blk:
            nl_lines.append(blk)
        next_z = z_orders[i + 1] if i + 1 < len(z_orders) else None
        if next_z is not None:
            nl_lines += [
                "",
                f"        if self.z_truncation_order < {next_z}:",
                early_return_nl("            "),
                "",
            ]
    nl_lines += ["", early_return_nl("        ")]
    nl_body = "\n".join(nl_lines)

    # ------------------------------------------------------------------
    # 7. Build jacobian_nonlinear_term method body
    # ------------------------------------------------------------------
    jac_lines = [
        "        x0 = state[..., 0:1, :]",
        "        x1 = state[..., 1:2, :]",
        "        z = x0 + 1j*x1",
        "        z_conj = x0 - 1j*x1",
        "        dy_dz = np.zeros_like(z)",
        "        dy_dz_conj = np.zeros_like(z)",
        "",
    ]
    if first_z is not None:
        jac_lines += [
            f"        if self.z_truncation_order < {first_z}:",
            early_return_jac("            "),
            "",
        ]
    for i, z_ord in enumerate(z_orders):
        _, dz_t, dzc_t = group_terms(groups[z_ord])
        blk_dz = add_block(dz_t, "dy_dz")
        blk_dzc = add_block(dzc_t, "dy_dz_conj")
        if blk_dz:
            jac_lines.append(blk_dz)
        if blk_dzc:
            jac_lines.append(blk_dzc)
        next_z = z_orders[i + 1] if i + 1 < len(z_orders) else None
        if next_z is not None:
            jac_lines += [
                "",
                f"        if self.z_truncation_order < {next_z}:",
                early_return_jac("            "),
                "",
            ]
    jac_lines += ["", early_return_jac("        ")]
    jac_body = "\n".join(jac_lines)

    # ------------------------------------------------------------------
    # 8. Write output file
    # ------------------------------------------------------------------
    code = f'''\
#%%
# AUTO-GENERATED by generate_dynamical_system.py
# Source: {Path(csv_path).resolve()}
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "pyhbm" / "pyhbm" / "src"))

import numpy as np
from numpy import array, concatenate
from pyhbm.dynamical_system import FirstOrderODE


class ParametricArchBackbone(FirstOrderODE):
    """
    Conservative autonomous ROM for the parametric arch backbone (pyhbm HBM).

    State (x0, x1) = (Re(z1), Im(z1)) where z1 = x0 + i*x1.
    Realification convention: z1 = x0 + i*x1, z2 = x0 - i*x1.

    Dynamics from R1 component of the MORFE ROM:
        z      = x0 + 1j*x1
        z_conj = x0 - 1j*x1
        y      = R1(z, z_conj, theta)
        x0dot  = y.real
        x1dot  = y.imag

    theta  : arch-height parameter, frozen at construction.
             theta = 0  -> base arch (h0_L_ratio from main.jl)
             theta = -1 -> straight beam
             theta = +1 -> doubled arch height

    z_truncation_order    : include nonlinear z-degree up to this order (default {max_z_ord})
    theta_truncation_order: include theta powers up to this order (default {max_c});
                            self.theta_p entries beyond this stay 0.0.
    """

    is_real_valued = True

    def __init__(self, theta: float = 0.0,
                 z_truncation_order: int = {max_z_ord},
                 theta_truncation_order: int = {max_c}):
        self.theta = theta
        theta = float(theta)
        self.z_truncation_order = z_truncation_order
        theta_truncation_order = min(theta_truncation_order, {max_c})
        self.theta_p = [0.0] * {max_c + 1}
        self.theta_p[0] = 1.0
        for i in range(1, theta_truncation_order + 1):
            self.theta_p[i] = self.theta_p[i - 1] * theta
        self.omega0 = {omega_str}
        self.linear_coefficient = array([
            [0.0, -self.omega0],
            [self.omega0,  0.0],
        ])
        self.dimension = 2
        self.polynomial_degree = {max_z_ord}

    def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
        zeros = np.zeros_like(adimensional_time)
        return array([[zeros, zeros]]).transpose()

    def linear_term(self, state: np.ndarray) -> np.ndarray:
        return self.linear_coefficient @ state

    def _pack_real_imag(self, y: np.ndarray) -> np.ndarray:
        return concatenate((y.real, y.imag), axis=-2)

    def _wirtinger_to_jacobian(self, dy_dz: np.ndarray, dy_dz_conj: np.ndarray) -> np.ndarray:
        dy_dx0 = dy_dz + dy_dz_conj
        dy_dx1 = 1j * (dy_dz - dy_dz_conj)
        jacobian0 = concatenate((dy_dx0.real, dy_dx1.real), axis=-1)
        jacobian1 = concatenate((dy_dx0.imag, dy_dx1.imag), axis=-1)
        return concatenate((jacobian0, jacobian1), axis=-2)

    def nonlinear_term(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
{nl_body}

    def jacobian_nonlinear_term(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
{jac_body}

    def all_terms(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        return self.linear_term(state) + self.nonlinear_term(state, adimensional_time)

# %%
'''

    with open(output_path, "w") as fh:
        fh.write(code)

    print(f"Written  -> {output_path}")
    print(f"  omega0(theta=0) = {omega0_val:.10f}")
    print(f"  z-orders: {z_orders}  (max_z_ord={max_z_ord})")
    print(f"  max_c (theta polynomial degree): {max_c}")
    print(f"  nonlinear CSV rows: {len(nl_rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamical_system.py from MORFE R_coefficients.csv",
    )
    parser.add_argument("csv_path", help="Path to R_coefficients.csv")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "dynamical_system.py"),
        help="Output path (default: same directory as this script)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-25,
        help="Drop coefficients with |val| below this (default: 1e-25)",
    )
    args = parser.parse_args()
    generate(args.csv_path, args.output, threshold=args.threshold)


if __name__ == "__main__":
    main()
