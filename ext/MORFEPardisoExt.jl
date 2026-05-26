module MORFEPardisoExt

using MORFE
using MORFE.CohomologicalEquations: _try_build_pardiso_solver, _pardiso_solve
using Pardiso

function MORFE.CohomologicalEquations._try_build_pardiso_solver()
    ps = nothing
    try ps = MKLPardisoSolver() catch end
    if ps === nothing
        try ps = Pardiso.PardisoSolver() catch end
    end
    if ps === nothing
        @warn "Neither MKL Pardiso nor open-source Pardiso is available. " *
              "Falling back to KLU (SuiteSparse) for the sparse cohomological solve."
    end
    return ps
end

function MORFE.CohomologicalEquations._pardiso_solve(ps, A, B)
    return Pardiso.solve(ps, A, B)
end

end # module MORFEPardisoExt
