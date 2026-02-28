# Problem Statement: Testing Eigensolvers for Sparse Generalized Eigenvalue Problems with Singular B

## 1. Objective
We need to identify and evaluate numerical eigensolvers capable of solving large-scale **generalized eigenvalue problems** of the form

$$
A \mathbf{x} = \lambda B \mathbf{x}
$$

where:
- $A$ and $B$ are **large, sparse matrices** (real or complex, non‑Hermitian / non‑symmetric).
- $B$ is **singular** (so the problem may have infinite eigenvalues).
- We are interested in a subset of eigenvalues, specifically those with the **largest real part**.
- The solution should be **accurate, efficient, and scalable**.
- Ideally, the solver should be callable from **Julia**, but alternatives in C/C++/Fortran are acceptable if they can be wrapped or interfaced.

## 2. Requirements
- **Sparsity:** The matrices are large and sparse; the solver must exploit sparsity.
- **Generalized problem:** Both $A$ and $B$ are provided explicitly; $B$ is singular.
- **Non‑symmetry:** No symmetry or definiteness can be assumed.
- **Eigenvalue selection:** Ability to compute a few (say 10–100) eigenvalues with largest real part, and optionally the corresponding eigenvectors.
- **Ordering:** The computed eigenvalues must be sorted by descending real part.
- **Language preference:** Julia is preferred for seamless integration into existing workflows; otherwise, a solver with a clear C interface or Julia wrapper is acceptable.

## 3. Test Matrices
To evaluate solvers, we will use a set of representative test problems:

1. **Synthetic problems**  
   - Construct $A$ and $B$ with known eigenvalues (including infinite ones) by factorisation, e.g.,  
     $A = X \Lambda Y^T$, $B = X Y^T$ with $X, Y$ random sparse matrices and $\Lambda$ diagonal containing desired eigenvalues.  
   - Introduce singularity in $B$ by making its last rows zero or by using a rank‑deficient factorisation.

2. **Discretized PDEs**  
   - Example: Convection‑diffusion operator for $A$ and mass matrix for $B$ on a finite element or finite difference grid.  
   - Modify $B$ by zeroing rows/columns corresponding to Dirichlet boundary conditions to make it singular.

3. **Standard benchmark collections**  
   - Matrices from the **SuiteSparse Matrix Collection** (formerly University of Florida Sparse Matrix Collection) that are known to be challenging, e.g., `bcsstk*.mat` (though many are symmetric) or other non‑symmetric examples.

4. **Structural dynamics / fluid mechanics** problems where singular mass matrices arise naturally.

## 4. Evaluation Criteria
For each solver, we will measure:

- **Correctness:** Eigenvalues should match reference values (or known exact ones) to within a tolerance. For infinite eigenvalues, the solver should either return them as very large (or Inf) or allow the user to filter them.
- **Performance:**
  - Time to solution (setup + solve)
  - Memory consumption
  - Scalability with matrix size and number of requested eigenvalues
- **Robustness:** Ability to handle singular $B$ without factorisation breakdown, and to converge for difficult spectra (clustered eigenvalues, ill‑conditioned matrices).
- **Ease of use:** Quality of documentation, simplicity of calling from Julia, flexibility in setting options (shift, target real part, etc.).