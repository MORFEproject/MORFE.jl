from pathlib import Path
import numpy as np
from numpy import (block, conj, array, hstack, diag, vstack, dot, kron, eye,
                   zeros, ones, reshape)
from numpy.linalg import norm, solve
import numpy as np

# ------------------------------------------------------------
#  Problem dimensions
# ------------------------------------------------------------
FOM = 5          # full order model dimension
ROM = 3          # reduced order model dimension
max_expansion_order = 3   # highest polynomial degree kept

# ------------------------------------------------------------
#  Define problem
# ------------------------------------------------------------
K = array([[2.0, -1.0],[-1.0, 2.0]])  # stiffness
C = array([[0.01, 0.0],[0.0, 0.01]])  # light damping
M = array([[1.0, 0.0],[0.0, 1.0]])    # mass (highest-order coefficient)

Id = np.eye(2)    # identity matrix
n = 3
I_red = np.eye(n)       # identity matrix for the reduced space
Zz = np.zeros((2, 2))   # zero matrix
Zz2 = np.zeros((2, 1))  # zero column vector

F = array([[1.0, 1.0]]).T # Force input matrix
external_eig = 1.0j

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

A2 = np.zeros((5, 5, 5))

"""
# Quadratic damping:  γ * ẋ²  (γ = 0.1)
term_drag = MultilinearMap(
	(res, v1, v2) -> (@. res += -0.1 * v1 * v2), # minus because it is on the right-hand side
	(0, 2),
)
"""
A2[2, 2, 2] = -0.1 # force on v1 with monomial v1²
A2[3, 3, 3] = -0.1 # force on v2 with monomial v2²

"""
# External harmonic forcing with twice the frequency
term_forcing_quadratic = MultilinearMap(
	(res, r1, r2) -> (@. res += F_ext * r1 * r2),
	(0, 0), 2,   # one external variable
)
"""
A2[2, 4, 4] = 1.0 # force on v1 with monomial r²
A2[3, 4, 4] = 1.0 # force on v2 with monomial r²

"""
# ExternalSystem: harmonic forcing ṙ = iΩ·r + 0.1 r² with Ω = 2.5
external_system = ExternalSystem(
	DensePolynomial(
		ComplexF64[1.0im 0.1+0.0im], # 1×2 matrix: coefficients for r and r² terms
		MultiindexSet([[1], [2]]),
	),
)
"""
A2[4, 4, 4] = 0.1 # ṙ = ...r + 0.1 * r²

A3 = np.zeros((5, 5, 5, 5))

"""
# Cubic stiffness:  β * x³  (Duffing-type, β = 1.0)
term_cubic = MultilinearMap(
	(res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), # minus because it is on the right-hand side
	(3, 0),
)
"""
A3[2, 0, 0, 0] = -1.0 # force on v1 with monomial u1³
A3[3, 1, 1, 1] = -1.0 # force on v2 with monomial u2³

#  Define full order dynamics A as a list of tensors
#  A[k-1]   -> matrix of size (FOM, FOM**k)  for order k = 1,2,3
A_raw = [A1, A2, A3]

# ------------------------------------------------------------
#  Define reduced linear quantities
#  (needs to satisfy A[0] @ W1 = B @ W1 @ R1)
# ------------------------------------------------------------
# choose simple diagonal linear part for testing

import h5py
with h5py.File(Path(__file__).parent / "output.h5", "r") as file:
    mset = file["multiindex_exponents"][:] 
    Wmorfe = file["W_coefficients"][:].reshape(len(mset), 4).T
    Rmorfe = file["R_coefficients"][:].T
    eigenvalues = np.reshape(file["super_eigenvalues"], (ROM,1))
    eig_master = diag(file["master_eigenvalues"])
    
#print("Multindex set (exponents of the monomials):\n", mset)

Wmorfe = vstack((Wmorfe, np.zeros((1, len(mset)))))
Wmorfe[4,2] = 1.0 # add the external forcing component

R1 = Rmorfe[:, 0:3]
W1 = Wmorfe[:, 0:3]

# left eigenvectors, biorthogonal: X.H @ B @ W1 = I
                               # ROM x FOM  (complex transpose form)
X = W1.conj().T                    # ROM x ROM

# Reshape each tensor into standard Kronecker form (FOM, FOM^k)

A = [matrix.reshape(FOM, FOM**(k+1)) for k, matrix in enumerate(A_raw)]

# ------------------------------------------------------------
#  Storage for reduced dynamics and parametrisation
# ------------------------------------------------------------
Rkron = [None for _ in range(max_expansion_order)]   # each entry (ROM, ROM**order)
Wkron = [None for _ in range(max_expansion_order)]   # each entry (FOM, ROM**order)
Gamma = [[None for _ in range(max_expansion_order)] for _ in range(max_expansion_order)]
Xi    = [[None for _ in range(max_expansion_order)] for _ in range(max_expansion_order)]

# ------------------------------------------------------------
#  Helper: fill Gamma and Xi matrices for a given order
# ------------------------------------------------------------
def fill_in_Gamma_Xi(order: int):
    """
    Populate Gamma[order-1][*] and Xi[order-1][*]
    using Rkron and Wkron matrices up to the given order.
    """
    # first first column (k = 0) of previous order
    Gamma[order-2][0] = Rkron[order-2]
    Xi[order-2][0] = Wkron[order-2]

    for k in range(1, order):
        if Gamma[order-1][k] is None:
            Gamma[order-1][k] = (kron(np.eye(ROM), Gamma[order-2][k-1])
                             + kron(Rkron[order-1-k], np.eye(ROM**k)))

    for k in range(1, min(order, len(A))):
        if Xi[order-1][k] is None:
            Xi[order-1][k] = np.sum(
                [kron(Wkron[order-sh-2], Xi[sh][k-1])
                for sh in range(k-1, order-1)],
                axis=0
            )

# ------------------------------------------------------------
#  Residual of the invariance equation (for checking)
# ------------------------------------------------------------
def compute_residue(order: int):
    return np.sum([B @ Wkron[k] @ Gamma[order-1][k] for k in range(order)], axis=0) \
        - np.sum([A[k] @ Xi[order-1][k] for k in range(min(order, len(A)))], axis=0)

# ------------------------------------------------------------
#  Right‑hand side for the cohomological equation
# ------------------------------------------------------------
def compute_rhs(order: int):
    if order == 1:
        return None   # linear part already solved
            
    tensor = np.sum([A[k] @ Xi[order-1][k] for k in range(1, min(order, len(A)))], axis=0)
    tensor -= np.sum([B @ Wkron[k] @ Gamma[order-1][k] for k in range(1, order-1)], axis=0)

    return tensor.ravel()           # FOM * ROM**order vector

# ------------------------------------------------------------
#  Linear operators used in the bordered system
# ------------------------------------------------------------
def compute_L_W(order: int):
    return kron(Gamma[order-1][order-1].T, B) - kron(np.eye(ROM**order), A[0])

def compute_L_R(order: int, param_style: callable, **param_kwargs):
    aux = kron(np.eye(ROM**order), B @ W1)           # (FOM*ROM**order, ROM**(order+1))
    is_resonant = param_style(order, **param_kwargs) # boolean mask, length ROM**(order+1)
    L_R = aux[:, is_resonant]                        # keep only resonant columns
    return L_R, is_resonant

def compute_C_W(order: int, is_resonant):
    aux = kron(np.eye(ROM**order), X @ B)           # (ROM**(order+1), FOM*ROM**order)
    return aux[is_resonant, :]                      # keep only resonant rows

# ------------------------------------------------------------
#  Style definitions for the reduced dynamics
# ------------------------------------------------------------
def normal_form_style(order, superharmonics, eigenvalues, tolerance_resonance):
    # matrix of shape (ROM**order, ROM) – compare each row with eigenvalues
    is_resonant = np.abs(superharmonics[:, None] - eigenvalues) < tolerance_resonance
    return is_resonant.ravel()   # boolean, length ROM**(order+1)

def graph_style(order, **param_kwargs):
    # graph style: keep all higher‑order reduced dynamics *zero*
    return np.ones(ROM**(order+1), dtype=bool)

# ------------------------------------------------------------
#  Initialise order 1 (linear part)
# ------------------------------------------------------------
tolerance_order1 = 1e-12
Rkron[0] = R1
Wkron[0] = W1
Gamma[0][0] = Rkron[0]
Xi[0][0] = Wkron[0]
assert norm(compute_residue(1)) < tolerance_order1, "Linear solution inaccurate"

# ------------------------------------------------------------
#  Loop over higher orders
# ------------------------------------------------------------
superharmonics = eigenvalues  # will be updated to eigenvalues ⊗ eigenvalues ⊗ ...

# choose style and parameters
param_style = normal_form_style # graph_style     # or normal_form_style
param_kwargs_base = {
    "eigenvalues": eigenvalues,
    "tolerance_resonance": 1e-6,   # large value for testing
}

for order in range(2, max_expansion_order + 1):
    # Kronecker powers of the eigenvalue vector
    superharmonics = np.kron(superharmonics, eigenvalues)

    # update style parameters with current superharmonics
    param_kwargs = {**param_kwargs_base, "superharmonics": superharmonics}
    
    fill_in_Gamma_Xi(order)                  # Gamma and Xi of previous order
    rhs = compute_rhs(order)                 # size FOM * ROM**order
    L_W = compute_L_W(order)                 # (FOM*ROM**order, FOM*ROM**order)
    L_R, is_resonant = compute_L_R(order, param_style, **param_kwargs)
    C_W = compute_C_W(order, is_resonant)    # (num_resonant, FOM*ROM**order)

    num_resonant = np.sum(is_resonant)
    if num_resonant == 0:
        # No resonant terms (e.g. graph style): solve directly
        w = solve(L_W, rhs)
        r_res = np.array([])
    else:
        # Bordered linear system
        M = np.block([[L_W, L_R],
                      [C_W, np.zeros((num_resonant, num_resonant))]])
        rhs_ext = np.concatenate([rhs, np.zeros(num_resonant)])
        sol = solve(M, rhs_ext)
        w = sol[:FOM * ROM**order]
        r_res = sol[FOM * ROM**order:]

    # Store parametrising map W
    Wkron[order-1] = w.reshape(FOM, ROM**order)

    # Store reduced dynamics R (resonant terms placed, non‑resonant zero)
    R_full = np.zeros(ROM**(order+1), dtype=complex)
    R_full[is_resonant] = r_res
    Rkron[order-1] = R_full.reshape(ROM, ROM**order)

Gamma[order-1][0] = Rkron[order-1]
Xi[order-1][0] = Wkron[order-1]
    
for order in range(1, max_expansion_order+1):
    print(f"order {order} error =", norm(compute_residue(order)))

print("SSM Kronecker computation finished successfully.")