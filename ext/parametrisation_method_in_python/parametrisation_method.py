from pathlib import Path
from numpy import block, conj, array, hstack, diag, block, vstack, dot, kron, eye
from numpy.linalg import norm
import numpy as np

FOM = 5
ROM = 3

A = [matrix.reshape(FOM, FOM**k) for k, matrix in enumerate(A)]
B = given

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
        
    for k in range(1, order):
        Xi[order-1][k] = np.sum([kron(Wkron[order-sh-2], Xi[sh][k-1]) for sh in range(k-1, order-1)], axis=0)

def compute_residue(order:int):
    if order == 1:
        return A[0] @ Wkron[0] - B @ Wkron[0] @ Rkron[0]
    
    return np.sum([
        B @ Wkron[k] @ Gamma[order-1][k] - A[k] @ Xi[order-1][k] 
        for k in range(order)], axis=0)
    
def compute_rhs(order:int):
    if order == 1:
        return None
    
    tensor = A[order-1] @ Xi[order-1][order-1] + np.sum([
         A[k] @ Xi[order-1][k] - B @ Wkron[k] @ Gamma[order-1][k]
        for k in range(1, order-1)], axis=0)
    
    return tensor.ravel()

def compute_L_W(order:int):
    return np.kron(Gamma[order-1][order-1].T, B)

def normal_form_style(order, superharmonics, eigenvalues, tolerance_resonance):
    is_resonant = abs(superharmonics - eigenvalues) < tolerance_resonance
    # matrix of shape (ROM**order, ROM)
    return is_resonant.ravel()

def graph_style(order, **param_kwargs):
    return np.array(True, shape=(ROM**(order+1)))

def compute_L_R(order:int, param_style: callable, **param_kwargs):
    aux = kron(eye(ROM**order), B @ W1)
    is_resonant = param_style(order, param_kwargs) # mask
    L_R = # output aux with mask applied to columns
    return L_R, is_resonant

def compute_C_W(order:int, is_resonant):
    aux = kron(eye(ROM**order), X @ B)
    return # output aux with mask is_resonant applied to rows

Rkron[0] = R1
Wkron[0] = W1
eigenvalues = eigenvalues # vector
X = # left eigenmodes in complex transpose form
assert np.linalg.norm(compute_residue(1)) < tolerance_order1
fill_in_Gamma_Xi(1)

param_style = graph_style
param_kwargs = {
    "superharmonics": superharmonics,
    "eigenvalues": eigenvalues,
    "tolerance_resonance": 1e6,
}

for order in range(1, max_order+1):
    superharmonics = kron(superharmonics, eigenvalues)
    rhs = compute_rhs(order)
    L_W = compute_L_W(order)
    L_R, is_resonant = compute_L_R(order, param_style, **param_kwargs)
    # build block matrix
    # M = [[L_W, L_R], [C_W, zero]]
    solution = np.linalg.solve(M, rhs)
    
    # split solution into the first FOM * ROM**order entries and the rest
    # the first block must be reshaped into shape (FOM, ROM**order) "unvectorise" and placed into W[order-1]
    # the second block needs the zeros to be placed back in the positions where is_resonant == False
    # the second block  is reshaped into shape (ROM, ROM**order) "unvectorise" and placed into R[order-1]
    fill_in_Gamma_Xi(2)
    
    






    