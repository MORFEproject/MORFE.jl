"""
Module `Eigensolvers` — high-level interface for computing generalised eigenpairs
of large sparse systems arising from `NDOrderModel`.

The primary entry point is `generalised_eigenpairs`, which wraps `Arpack.eigs` to
compute a selected subset of eigenpairs of the companion-form system `A v = λ B v`.
A shift-invert spectral transformation is used internally to target eigenpairs near
a specified shift `σ`, which is essential for large sparse problems where only a
small number of physically relevant modes are needed.
"""
module Eigensolvers

using LinearAlgebra
using SparseArrays

export generalised_eigenpairs

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

end
