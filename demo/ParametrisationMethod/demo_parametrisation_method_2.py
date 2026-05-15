from pathlib import Path
from numpy import block, conj, array, hstack, diag, vstack, dot, kron
from numpy.linalg import norm
import numpy as np
from math import factorial
from itertools import product

# ---------- system matrices (unchanged) ----------
K = array([[2.0, -1.0],[-1.0, 2.0]])  # stiffness
M = array([[1.0, 0.0],[0.0, 1.0]])    # mass (highest-order coefficient)
C = 0.001 * M                         # light damping

Id = np.eye(2)        # identity matrix
n = 3                 # reduced dimension
I_red = np.eye(n)     # identity matrix for reduced space
Zz = np.zeros((2, 2))
Zz2 = np.zeros((2, 1))

F = array([[1.0, 1.0]]).T   # Force input matrix
external_eig = -0.5

# Linear operator B and A₁
B = block([
    [Id,    Zz,     Zz2],
    [Zz,    M,      Zz2],
    [Zz2.T, Zz2.T,  1],
])

A1 = block([
    [Zz,    Id,     Zz2],
    [-K,    -C,     F],
    [Zz2.T, Zz2.T,  external_eig]
])

# Quadratic tensor A₂  (5×5×5)
A2 = np.zeros((5, 5, 5))
A2[2, 2, 2] = -0.1   # v1, v1²
A2[3, 3, 3] = -0.1   # v2, v2²
A2[2, 4, 4] =  1.0   # v1, r²
A2[3, 4, 4] =  1.0   # v2, r²
A2[4, 4, 4] =  0.1   # external oscillator nonlinearity 0.1 r²

# Cubic tensor A₃  (5×5×5×5)
A3 = np.zeros((5, 5, 5, 5))
A3[2, 0, 0, 0] = -1.0   # v1, u1³
A3[3, 1, 1, 1] = -1.0   # v2, u2³
A3[4, 4, 4, 4] = 2.0    # external oscillator nonlinearity 2.0 r³

# Quartic tensor (5×5×5×5x5)
A4 = np.zeros((5, 5, 5, 5, 5))
A4[2, 0, 0, 2, 4] = 3.0   # v1, u1² v1 r
A4[3, 1, 1, 3, 4] = 3.0   # v2, u2² v2 r

import itertools
# Symmetrise A4 over the last four axes
sym_axes = [1, 2, 3, 4]
perms = list(itertools.permutations(sym_axes))
A4_sym = np.zeros_like(A4)
for p in perms:
    A4_sym += np.transpose(A4, [0] + list(p))
A4_sym /= len(perms)

# List of tensors Aₖ (k = 1,2,3)
A = [A1, A2.reshape(5, 5**2), A3.reshape(5, 5**3), A4_sym.reshape(5, 5**4)]

max_expansion_order = 3   # will be extended later
Rkron = [None for _ in range(max_expansion_order)]
Wkron = [None for _ in range(max_expansion_order)]
Gamma = [[None for _ in range(max_expansion_order)] for _ in range(max_expansion_order)]
Xi    = [[None for _ in range(max_expansion_order)] for _ in range(max_expansion_order)]


def fill_in_Gamma_Xi(order: int):
    """Fill the auxiliary Gamma and Xi matrices for a given order (1‑indexed)."""
    Gamma[order-1][0] = Rkron[order-1]

    for k in range(1, order):
        Gamma[order-1][k] = kron(I_red, Gamma[order-2][k-1]) + \
                            kron(Rkron[order-1-k], np.eye(n**k))

    Xi[order-1][0] = Wkron[order-1]

    for k in range(1, min(order, len(A))):
        Xi[order-1][k] = np.sum(
            [kron(Wkron[order-sh-2], Xi[sh][k-1]) for sh in range(k-1, order-1)],
            axis=0
        )


def compute_residue(order: int):
    """Evaluate the invariance equation residue at the given order."""
    if order == 1:
        return A[0] @ Wkron[0] - B @ Wkron[0] @ Rkron[0]

    return np.sum([B @ Wkron[k] @ Gamma[order-1][k] for k in range(order)], axis=0) \
         - np.sum([A[k] @ Xi[order-1][k] for k in range(min(order, len(A)))], axis=0)


# ---------- read reduced manifold data ----------
import h5py
with h5py.File(Path(__file__).parent / "output.h5", "r") as file:
    mset = file["multiindex_exponents"][:]          # (num_monomials, n_red)
    W = file["W_coefficients"][:].reshape(len(mset), 4).T  # (4, num_monomials)
    R = file["R_coefficients"][:].T                        # (n_red, num_monomials)
    eig_master = diag(file["master_eigenvalues"])

#print("Multindex set (exponents of the monomials):\n", mset)

# Add the external variable component to W
W = vstack((W, np.zeros((1, len(mset)))))
W[4, 2] = 1.0    # linear coupling for the external oscillator

# ---------- helper: build Kronecker forms for arbitrary order ----------
def build_kron_matrices(mset, W, R, order):
    """
    Return (Wkron, Rkron) for the given order k = order.
    mset : (N, n_red) array of multi‑index exponents (graded lexicographic order)
    W    : (full_dim, N)  parametrisation coefficients
    R    : (n_red, N)     reduced dynamics coefficients
    """
    n_red = R.shape[0]
    # total order of each monomial
    orders = np.sum(mset, axis=1)

    # dictionary mapping tuple(multi‑index) -> column index in W/R
    # restrict to monomials of this exact order
    idx_map = {}
    for i, alpha in enumerate(mset):
        if orders[i] == order:
            idx_map[tuple(alpha)] = i

    # generate all tuples (i₁,…,iₖ) in lexicographic order
    tuples = np.array(list(product(range(n_red), repeat=order)))   # shape (n_red^order, order)
    n_cols = tuples.shape[0]

    Wk_list = []
    Rk_list = []

    for tup in tuples:
        # count occurrences to obtain the multi‑index α
        alpha = tuple(np.bincount(tup, minlength=n_red))
        col = idx_map[alpha]

        # symmetry factor s = k! / α!   (multinomial coefficient)
        s = factorial(order)
        for a in alpha:
            s //= factorial(a)

        Wk_list.append(W[:, col] / s)
        Rk_list.append(R[:, col] / s)

    Wkron = np.column_stack(Wk_list) if Wk_list else np.empty((W.shape[0], 0))
    Rkron = np.column_stack(Rk_list) if Rk_list else np.empty((R.shape[0], 0))
    return Wkron, Rkron


# ---------- determine max order from data ----------
orders = np.sum(mset, axis=1)
max_order_data = orders.max() if len(orders) > 0 else 0
print(f"Maximum monomial order in the reduced data: {max_order_data}")

# reallocate lists to accommodate up to max_order_data
Rkron = [None] * max_order_data
Wkron = [None] * max_order_data
Gamma = [[None for _ in range(max_order_data)] for _ in range(max_order_data)]
Xi    = [[None for _ in range(max_order_data)] for _ in range(max_order_data)]

# ---------- build and check all orders ----------
for k in range(1, max_order_data + 1):
    Wkron[k-1], Rkron[k-1] = build_kron_matrices(mset, W, R, k)
    fill_in_Gamma_Xi(k)
    res = compute_residue(k)
    print(f"Order {k} Invariance Equation Residue: {norm(res):.4e}")