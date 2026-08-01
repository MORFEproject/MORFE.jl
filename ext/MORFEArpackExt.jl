module MORFEArpackExt

using MORFE
using MORFE.Eigenproblems
using MORFE.Eigensolvers
using Arpack
using LinearAlgebra
using LinearMaps
using SparseArrays

# ── generalised_eigenpairs ─────────────────────────────────────────────────────

function MORFE.Eigensolvers.generalised_eigenpairs(
        A::AbstractMatrix,
        B::AbstractMatrix;
        nev::Integer,
        shift = nothing,
        which::Symbol = :LR,
        tol::Real = 0.0,
        maxiter::Integer = 3000,
        ncv::Union{Nothing, Integer} = nothing,
        v0 = nothing,
        ritzvec::Bool = true,
        sort_largest_real::Bool = false
)
    n = size(A, 1)
    @assert size(A, 2) == n "A must be square"
    @assert size(B, 1) == n && size(B, 2) == n "B must be square and match A size"
    @assert 0 <= nev <= n "nev must satisfy 0 < nev < size(A,1)"

    Tval = isnothing(shift) ? promote_type(eltype(A), eltype(B)) :
           promote_type(eltype(A), eltype(B), typeof(shift))
    Ac = sparse(Tval.(A))
    Bc = sparse(Tval.(B))

    ncv_eff = min(isnothing(ncv) ? max(Int(nev) + 30, 120) : Int(ncv), n - 1)
    v0_eff = isnothing(v0) ? nothing : Tval.(v0)

    base_eigs_kwargs = (
        nev = Int(nev),
        which = which,
        tol = Float64(tol),
        maxiter = Int(maxiter),
        ncv = ncv_eff,
        ritzvec = ritzvec
    )
    eigs_kwargs = isnothing(v0_eff) ? base_eigs_kwargs :
                  merge(base_eigs_kwargs, (v0 = v0_eff,))

    vals = Vector{Tval}()
    vecs = Matrix{Tval}(undef, n, 0)
    nconv = 0
    niter = 0
    nmult = 0
    resid = Tval[]

    if isnothing(shift)
        vals, vecs, nconv, niter, nmult, resid = eigs(Ac, Bc; eigs_kwargs...)
    else
        sigc = convert(Tval, shift)
        F = lu(Ac - sigc * Bc)
        T_lm = LinearMap{Tval}(n, n; ismutating = false) do x
            F \ (Bc * x)
        end
        mu, vecs, nconv, niter, nmult,
        resid = eigs(T_lm; merge(eigs_kwargs, (which = :LM,))...)

        tiny = eps(real(float(one(Tval))))
        mu_safe = similar(mu)
        for i in eachindex(mu)
            mu_safe[i] = abs(mu[i]) < tiny ? convert(Tval, tiny) : mu[i]
        end
        vals = sigc .+ inv.(mu_safe)
    end

    if sort_largest_real
        vals, vecs = MORFE.Eigensolvers._sort_largest_real(vals, vecs)
    end

    return (
        values = vals,
        vectors = vecs,
        nconv = nconv,
        niter = niter,
        nmult = nmult,
        resid = resid
    )
end

# ── ArpackEigensolver.solve ────────────────────────────────────────────────────

function MORFE.Eigenproblems.solve(
        model::MORFE.FullOrderModel.NDOrderModel,
        solver::MORFE.Eigenproblems.ArpackEigensolver
)
    A, B = MORFE.FullOrderModel.linear_first_order_matrices(model)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    if solver.nev === nothing
        solver.nev = FOM * ORD
        (values, vectors) = eigs(A, B, which = :SM)
    else
        (values, vectors) = eigs(A, B, nev = solver.nev, which = :SM)
    end
    num_eigenvals = length(values)
    reshaped_eigenvectors = reshape(vectors, FOM, ORD, num_eigenvals)
    solver.eigenvalues = values
    return values, reshaped_eigenvectors
end

# ── ArpackEigensolver.solve_left ───────────────────────────────────────────────
# Uses per-eigenvalue sigma-shifts — NOT a simple eigs(A', B'; which=:SM).

function MORFE.Eigenproblems.solve_left(
        model::MORFE.FullOrderModel.NDOrderModel,
        solver::MORFE.Eigenproblems.ArpackEigensolver
)
    A, B = MORFE.FullOrderModel.linear_first_order_matrices(model)
    A_c = complex.(A)
    B_c = complex.(B)
    A_adjoint = A_c'
    B_adjoint = B_c'
    @assert solver.nev == length(solver.eigenvalues)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    left_eigenvectors = Array{ComplexF64}(undef, FOM, ORD, solver.nev)
    eigenvalues = Vector{ComplexF64}(undef, solver.nev)
    for i in 1:(solver.nev)
        values, vectors = eigs(
            A_adjoint, B_adjoint,
            sigma = conj(solver.eigenvalues[i]),
            which = :LM, nev = 1, ncv = 30
        )
        left_eigenvectors[:, :, i] = reshape(vectors[:, 1], FOM, ORD)
        eigenvalues[i] = conj(values[1])
    end
    return eigenvalues, left_eigenvectors
end

# ── StructureModalDampingEigensolver.solve(mass, stiffness, solver) ────────────
# The thin wrapper solve(model, solver) stays in core and delegates here.

function MORFE.Eigenproblems.solve(
        mass::AbstractMatrix{T},
        stiffness::AbstractMatrix{T},
        solver::MORFE.Eigenproblems.StructureModalDampingEigensolver
) where {T}
    ω2, ϕ = eigs(stiffness, mass; nev = solver.nev, which = :LM, sigma = 0.0, check = 1)
    any(x -> abs(imag(x)) > 1e-12 * abs(real(x)), ω2) && error("Eigenvalues not real.")
    ω2 = real.(ω2)
    ϕ = real.(ϕ)

    FOM = size(ϕ, 1)
    CT = Complex{T}
    λ = zeros(CT, solver.nev * 2)
    eigenvectors = Array{CT}(undef, FOM, 2, solver.nev * 2)

    for i in 1:(solver.nev)
        ω2i = ω2[i]
        real_part = -0.5 * (solver.α + solver.β * ω2i)
        discriminant = real_part^2 - ω2i
        if discriminant < 0
            imag_part = sqrt(-discriminant)
            λ[2 * i - 1] = complex(real_part, imag_part)
            λ[2 * i] = complex(real_part, -imag_part)
        else
            delta = sqrt(discriminant)
            λ[2 * i - 1] = complex(real_part + delta, 0.0)
            λ[2 * i] = complex(real_part - delta, 0.0)
        end

        ϕ_i = view(ϕ, :, i)
        norm_sq = dot(ϕ_i, mass, ϕ_i)
        ϕ_i ./= sqrt(norm_sq)

        eigenvectors[:, 1, 2i - 1] .= ϕ_i
        eigenvectors[:, 1, 2i] .= ϕ_i
        eigenvectors[:, 2, 2i - 1] .= λ[2i - 1] * ϕ_i
        eigenvectors[:, 2, 2i] .= λ[2i] * ϕ_i
    end
    return λ, eigenvectors
end

end # module MORFEArpackExt
