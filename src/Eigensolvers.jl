module Eigensolvers

using Arpack
using LinearAlgebra
using LinearMaps
using SparseArrays

export generalized_eigenpairs

@inline function _sort_largest_real(vals, vecs)
	p = sortperm(real.(vals); rev=true)
	return vals[p], vecs[:, p]
end

"""
	generalized_eigenpairs(A, B; nev, sigma=nothing, which=:LM, tol=0.0,
						   maxiter=3000, ncv=nothing, v0=nothing,
						   ritzvec=true, sort_largest_real=false)

Solve the generalized eigenproblem A x = lambda B x

Returns a named tuple with:
- `values`: eigenvalues
- `vectors`: eigenvectors (columns)
- `nconv`, `niter`, `nmult`, `resid`: ARPACK diagnostics
"""
function generalized_eigenpairs(
	A::AbstractMatrix,
	B::AbstractMatrix;
	nev::Integer,
	sigma=nothing,
	which::Symbol=:LR,
	tol::Real=0.0,
	maxiter::Integer=3000,
	ncv::Union{Nothing,Integer}=nothing,
	v0=nothing,
	ritzvec::Bool=true,
	sort_largest_real::Bool=false,
)
	n = size(A, 1)
	@assert size(A, 2) == n "A must be square"
	@assert size(B, 1) == n && size(B, 2) == n "B must be square and match A size"
	@assert 0 < nev < n "nev must satisfy 0 < nev < size(A,1)"

	Tval = isnothing(sigma) ? promote_type(eltype(A), eltype(B)) : promote_type(eltype(A), eltype(B), typeof(sigma))
	Ac = sparse(Tval.(A))
	Bc = sparse(Tval.(B))

	ncv_eff = min(isnothing(ncv) ? max(Int(nev) + 30, 120) : Int(ncv), n - 1)
	v0_eff = isnothing(v0) ? nothing : Tval.(v0)

	base_eigs_kwargs = (
		nev=Int(nev),
		which=which,
		tol=Float64(tol),
		maxiter=Int(maxiter),
		ncv=ncv_eff,
		ritzvec=ritzvec,
	)
	eigs_kwargs = isnothing(v0_eff) ? base_eigs_kwargs : merge(base_eigs_kwargs, (v0=v0_eff,))

	vals = Vector{Tval}()
	vecs = Matrix{Tval}(undef, n, 0)
	nconv = 0
	niter = 0
	nmult = 0
	resid = Tval[]

	if isnothing(sigma)
		# No shift requested: use ARPACK generalized eigs directly, so `which`
		# follows the package's native semantics.
		vals, vecs, nconv, niter, nmult, resid = eigs(Ac, Bc; eigs_kwargs...)
	else
		sigc = convert(Tval, sigma)
		F = lu(Ac - sigc * Bc)
		T = LinearMap{Tval}(n, n; ismutating=false) do x
			F \ (Bc * x)
		end

		mu, vecs, nconv, niter, nmult, resid = eigs(T; merge(eigs_kwargs, (which=:LM,))...)

		tiny = eps(real(float(one(Tval))))
		mu_safe = similar(mu)
		for i in eachindex(mu)
			mu_safe[i] = abs(mu[i]) < tiny ? convert(Tval, tiny) : mu[i]
		end

		vals = sigc .+ inv.(mu_safe)
	end

	if sort_largest_real
		vals, vecs = _sort_largest_real(vals, vecs)
	end

	return (
		values=vals,
		vectors=vecs,
		nconv=nconv,
		niter=niter,
		nmult=nmult,
		resid=resid,
	)
end

end