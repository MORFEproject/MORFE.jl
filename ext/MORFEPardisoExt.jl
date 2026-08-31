"""
Pardiso backend for the sparse bordered cohomological solve.

Deliberately does **not** use `Pardiso.solve`/`solve!`. Those run
`ANALYSIS_NUM_FACT_SOLVE_REFINE` and then `RELEASE_ALL` on every call, so the
symbolic analysis is recomputed and thrown away for each monomial — the opposite of
what the constant-size bordered formulation is built for. They also re-run
`ishermitian`/`issymmetric`/`isstructurallysymmetric` (all `O(nnz)`) per call, and
call `set_matrixtype!` unconditionally, overwriting anything configured here.

Instead the phases are driven directly:

	prepare  :  detect type → pardisoinit → fix_iparm! → ANALYSIS          (once)
	solve    :  NUM_FACT → SOLVE_ITERATIVE_REFINE                          (per monomial)
	release  :  RELEASE_ALL                                                (finaliser)

## Matrix type: `COMPLEX_NONSYM`, unconditionally

No symmetry of any kind is claimed — not numerical, and **not structural either**.

*Numerical* symmetry is out because it is not even invariant: the border values
change with every monomial's superharmonic, so a matrix that happens to be symmetric
at one monomial need not be at the next.

*Structural* symmetry is out for two independent reasons. It is not a property of
the formulation — the border contributes a dense row together with its matching
dense column, so the bordered pattern is symmetric exactly when the `L` union
pattern is, and that is model-dependent (a non-symmetric convection operator, or
asymmetric constraint elimination, breaks it). Declaring it would therefore make the
pivoting strategy vary silently from model to model. And even where it does hold,
`COMPLEX_STRUCT_SYM` (mtype 3) constrains the permutation to preserve that pattern,
which forfeits free row interchange across the border — the one property the whole
bordered formulation rests on, and precisely at the resonant monomials where the
`(1,1)` block is singular and the border is load-bearing.

Declaring `COMPLEX_NONSYM` is not an assumption that the matrix *is* unsymmetric; it
is the type that leaves Pardiso free to pivot. Its analysis still applies its own
fill-reducing ordering and supernode detection, so structure that is there is still
exploited — the optimisation is left to the solver rather than asserted by us.

## Robustness on a near-singular matrix

The bordered matrix is near-singular in its `(1,1)` block at every resonant monomial
— that is the whole point of bordering. Scaling (`iparm[10]` in Intel's 0-based
numbering) and weighted matching (`iparm[12]`) are what keep such a system solvable.
Pardiso enables both by default for `mtype` 13, but they are set explicitly here so
the behaviour does not depend on a default.

Pardiso is a weak dependency, so this backend is unreachable from the test suite.
`_pardiso_factorise_solve!` therefore checks its own first solve against the
residual `‖A·x − b‖/‖b‖` and raises on a bad answer, rather than letting a
misconfigured solver return plausible-looking numbers.
"""
module MORFEPardisoExt

using MORFE
using MORFE.BorderedLinearSolvers: _try_build_pardiso_solver, _pardiso_prepare!,
                                   _pardiso_factorise_solve!, _pardiso_solve!,
                                   _pardiso_release!
using Pardiso
using SparseArrays
using LinearAlgebra: norm

function MORFE.BorderedLinearSolvers._try_build_pardiso_solver()
    ps = nothing
    try
        ps = MKLPardisoSolver()
    catch
    end
    if ps === nothing
        try
            ps = Pardiso.PardisoSolver()
        catch
        end
    end
    if ps === nothing
        @warn "Neither MKL Pardiso nor open-source Pardiso is available. " *
              "Falling back to KLU (SuiteSparse) for the sparse cohomological solve."
    end
    return ps
end

# Intel's iparm indices are 0-based; Pardiso.jl's `set_iparm!` is 1-based, so each is
# one higher here. (Pardiso.jl's own `fix_iparm!` uses 12 for the transpose flag,
# which is Intel's iparm[11] — same convention.)
const IPARM_SCALING = 11   # Intel iparm[10]
const IPARM_MATCHING = 13  # Intel iparm[12]

"""
	_pardiso_prepare!(ps, A) -> A_pardiso

Configure the solver for `A` and run the analysis phase once. Returns the matrix in
whatever form the chosen type requires; the caller holds it and passes it back to
every subsequent `_pardiso_factorise_solve!`.
"""
function MORFE.BorderedLinearSolvers._pardiso_prepare!(ps, A::SparseMatrixCSC)
    # Unconditionally unsymmetric — see the module docstring. This is not a claim
    # about A; it is the only complex type that leaves pivoting unrestricted, which
    # the bordered solve requires at every resonant monomial. Pardiso still applies
    # its own ordering and supernode detection, so real structure is still exploited.
    Pardiso.set_matrixtype!(ps, Pardiso.COMPLEX_NONSYM)
    Pardiso.pardisoinit(ps)          # seeds iparm from the matrix type; must follow it
    Pardiso.fix_iparm!(ps, :N)       # iparm[12]: Julia CSC handed over as Pardiso CSR

    # Restore the accuracy options Pardiso only defaults on for mtype 11/13.
    Pardiso.set_iparm!(ps, IPARM_SCALING, 1)
    Pardiso.set_iparm!(ps, IPARM_MATCHING, 1)

    A_pardiso = Pardiso.get_matrix(ps, A, :N)
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
    Pardiso.pardiso(ps, A_pardiso, eltype(A)[])
    return A_pardiso
end

"""
	_pardiso_factorise_solve!(ps, A_pardiso, x, b) -> x

Numeric factorisation followed by a solve, reusing the cached analysis. Writes into
`x`; `b` must not alias it.

The first call additionally checks its own answer. This path is unreachable from CI,
so a silently misconfigured solver — the wrong matrix type, a transpose-flag
mismatch — would otherwise surface as a quietly wrong ROM rather than an error.
"""
function MORFE.BorderedLinearSolvers._pardiso_factorise_solve!(
        ps, A_pardiso, x::AbstractVector, b::AbstractVector)
    Pardiso.set_phase!(ps, Pardiso.NUM_FACT)
    Pardiso.pardiso(ps, A_pardiso, eltype(b)[])
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, x, A_pardiso, b)

    if !_PARDISO_VERIFIED[]
        _PARDISO_VERIFIED[] = true
        residual = norm(A_pardiso * x - b) / max(norm(b), eps())
        residual ≤ 1e-6 || error("""
          Pardiso returned a bad solution on the first cohomological system:
          relative residual ‖A·x − b‖/‖b‖ = $residual.

          The bordered solve is being driven through Pardiso's low-level phase API with
          matrix type $(Pardiso.get_matrixtype(ps)). A large residual here points at the
          configuration rather than the model — most likely the CSC/CSR transpose flag
          (iparm[12]) or the matrix type. Load MORFE without Pardiso to fall back to
          KLU, which is exercised by the test suite.""")
    end
    return x
end

function MORFE.BorderedLinearSolvers._pardiso_solve!(
        ps, A_pardiso, x::AbstractVector, b::AbstractVector)
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, x, A_pardiso, b)
    return x
end

"""
	_pardiso_release!(ps, A_pardiso) -> nothing

Release Pardiso's internal memory. Called from a finaliser on the solver state:
Pardiso allocates C-side, outside the GC's view, so without this every solve leaks a
factorisation.
"""
function MORFE.BorderedLinearSolvers._pardiso_release!(ps, A_pardiso)
    A_pardiso === nothing && return nothing
    Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(ps, A_pardiso, eltype(A_pardiso)[])
    return nothing
end

# One-shot flag for the first-solve residual check above.
const _PARDISO_VERIFIED = Ref(false)

end # module MORFEPardisoExt
