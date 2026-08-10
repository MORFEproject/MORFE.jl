
@inline function _sort_largest_real(vals, vecs)
    p = sortperm(real.(vals); rev = true)
    return vals[p], vecs[:, p]
end

"""
	generalised_eigenpairs(A, B; nev, shift=nothing, which=:LM, tol=0.0,
						   maxiter=3000, ncv=nothing, v0=nothing,
						   ritzvec=true, sort_largest_real=false)

Solve the generalised eigenproblem A x = lambda B x using Arpack.

Requires Arpack.jl and LinearMaps.jl. Load them to activate the MORFE extension.
"""
function generalised_eigenpairs(args...; kwargs...)
    error(
        "generalised_eigenpairs requires Arpack.jl and LinearMaps.jl.\n" *
        "Load them with `using Arpack, LinearMaps` to activate the MORFE extension.",
    )
end
