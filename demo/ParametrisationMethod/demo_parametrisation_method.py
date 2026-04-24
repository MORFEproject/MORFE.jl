from pathlib import Path
from numpy import block, conj, array, hstack, diag, block, vstack, dot, kron
from numpy.linalg import norm
import numpy as np

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

A = [A1, A2.reshape(5, 5**2), A3.reshape(5, 5**3)]

max_expansion_order = 3
Rkron = [None for _ in range(max_expansion_order)] 
Wkron = [None for _ in range(max_expansion_order)]
Gamma = [[None for _ in range(max_expansion_order)] for _ in range(max_expansion_order)]
Xi = [[None for _ in range(max_expansion_order)] for _ in range(max_expansion_order)]

def fill_in_Gamma_Xi(order:int):
    Gamma[order-1][0] = Rkron[order-1]
        
    for k in range(1, order):
        Gamma[order-1][k] = kron(I_red, Gamma[order-2][k-1]) + kron(Rkron[order-1-k], np.eye(n**k))
        
    Xi[order-1][0] = Wkron[order-1]
        
    for k in range(1, min(order, len(A))):
        Xi[order-1][k] = np.sum([kron(Wkron[order-sh-2], Xi[sh][k-1]) for sh in range(k-1, order-1)], axis=0)

def compute_residue(order:int):
    if order == 1:
        return A[0] @ Wkron[0] - B @ Wkron[0] @ Rkron[0]
    
    return np.sum([ B @ Wkron[k] @ Gamma[order-1][k] for k in range(order)], axis=0) \
         - np.sum([A[k] @ Xi[order-1][k] for k in range(min(order, len(A)))], axis=0)
    
import h5py
with h5py.File(Path(__file__).parent / "output.h5", "r") as file:
    mset = file["multiindex_exponents"][:] 
    W = file["W_coefficients"][:].reshape(len(mset), 4).T
    R = file["R_coefficients"][:].T
    eig_master = diag(file["master_eigenvalues"])
    
print("Multindex set (exponents of the monomials):\n", mset)

W = vstack((W, np.zeros((1, len(mset)))))
W[4,2] = 1.0 # add the external forcing component

Rkron[0] = R[:, 0:3]
Wkron[0] = W[:, 0:3]

fill_in_Gamma_Xi(1)

print("Order 1 Invariance Equation Residue:",  norm(compute_residue(1)))

#%% Order 2

W200 = W[:, 3:4]
W110 = W[:, 4:5]
W101 = W[:, 5:6]
W020 = W[:, 6:7]
W011 = W[:, 7:8]
W002 = W[:, 8:9]

R002 = R[:, 8:9]
R011 = R[:, 7:8]
R020 = R[:, 6:7]
R101 = R[:, 5:6]
R110 = R[:, 4:5]
R200 = R[:, 3:4]

# xx, xy, xz
# xy, yy, yz
# xz, yz, zz

# order 2

Rkron[1] = hstack(( 
    R200,   R110/2, R101/2,
    R110/2, R020,   R011/2,
    R101/2, R011/2, R002
))

Wkron[1] = hstack((
    W200,   W110/2, W101/2,
    W110/2, W020,   W011/2,
    W101/2, W011/2, W002
))

fill_in_Gamma_Xi(2)
inv_res_2 = compute_residue(2)
print("Order 2 Invariance Equation Residue:", norm(inv_res_2))

#%% Order 3

W300 = W[:, 9:10]
W210 = W[:, 10:11]
W201 = W[:, 11:12]
W120 = W[:, 12:13]
W111 = W[:, 13:14]
W102 = W[:, 14:15]
W030 = W[:, 15:16]
W021 = W[:, 16:17]
W012 = W[:, 16:17]
W003 = W[:, 17:18]

R300 = R[:, 9:10]
R210 = R[:, 10:11]
R201 = R[:, 11:12]
R120 = R[:, 12:13]
R111 = R[:, 13:14]
R102 = R[:, 14:15]
R030 = R[:, 15:16]
R021 = R[:, 16:17]
R012 = R[:, 16:17]
R003 = R[:, 17:18]

# xxx, xxy, xxz
# xxy, xyy, xyz
# xxz, xyz, xzz

# xxy, xyy, xyz
# xyy, yyy, yyz
# xyz, yyz, yzz

# xxz, xyz, xzz
# xyz, yyz, yzz
# xzz, yzz, zzz

# order 3

Rkron[2] = hstack(( 
    R300,   R210/3, R201/3,
    R210/3, R120/3, R111/6,
    R201/3, R111/6, R102/3,
    #
    R210/3, R120/3, R111/6,
    R120/3, R030,   R021/3,
    R111/6, R021/3, R012/3,
    #
    R201/3, R111/6, R102/3,
    R111/6, R021/3, R012/3,
    R102/3, R012/3, R003
))

Wkron[2] = hstack(( 
    W300,   W210/3, W201/3,
    W210/3, W120/3, W111/6,
    W201/3, W111/6, W102/3,
    #
    W210/3, W120/3, W111/6,
    W120/3, W030,   W021/3,
    W111/6, W021/3, W012/3,
    #
    W201/3, W111/6, W102/3,
    W111/6, W021/3, W012/3,
    W102/3, W012/3, W003
))

fill_in_Gamma_Xi(3)

inv_res_3 = compute_residue(3)
print("Order 3 Invariance Equation Residue:", norm(inv_res_3))