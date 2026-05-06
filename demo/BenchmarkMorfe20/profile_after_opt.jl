"""
profile_after_opt.jl
====================
Collect a new Profile of solve_cohomological_problem and save the flame graph
to HTML (same format as the original profile_morfe.html) so that old vs new
can be compared side-by-side.

Run from the repo root:
    julia --project=. demo/BenchmarkMorfe20/profile_after_opt.jl
"""

using MORFE
include("Morfe_2_0/Morfe_2_0.jl")
using .Morfe_2_0

using LinearAlgebra
using SparseArrays
using StaticArrays
using KrylovKit
using Profile
using ProfileCanvas

# ===========================================================================
# Model setup (identical to benchmark_speedup.jl)
# ===========================================================================

info = Infostruct()
include(joinpath(@__DIR__, "beam_damp.jl"))
mesh = read_mesh(joinpath(@__DIR__, "beam.mphtxt"),
    domains_list, materials_list, materials_dict,
    boundaries_list, constrained_dof, bc_vals)
U = Field(mesh, Morfe_2_0.dim)

colptr, rowval = Morfe_2_0.assembler_dummy_MK(mesh, U)
val = zeros(Float64, length(rowval))
K = SparseMatrixCSC(U.neq, U.neq, colptr, rowval, val)
M = deepcopy(K)
Morfe_2_0.assembler_MK!(mesh, U, K, M)
C = info.α * M + info.β * K

quadratic_term = MultilinearMap((res, Ψ₁, Ψ₂)    -> Morfe_2_0.assembly_G!(res, Ψ₁, Ψ₂, mesh, U),    (2, 0))
cubic_term     = MultilinearMap((res, Ψ₁, Ψ₂, Ψ₃) -> Morfe_2_0.assembly_H!(res, Ψ₁, Ψ₂, Ψ₃, mesh, U), (3, 0))
model = NDOrderModel((K, C, M), (quadratic_term, cubic_term))
FOM = size(K, 1)
println("FOM = $FOM")

# Eigensolver (KrylovKit Lanczos — reliable at this FOM)
mutable struct MechSolver <: AbstractEigensolver
    right_eig_result::Union{Nothing, Matrix}
    eigenvalues::Union{Nothing, Vector}
    nev::Int; α::Float64; β::Float64
end
function mass_normalize!(ϕ, M, n)
    for i in 1:n
        c = transpose(ϕ[:, i]) * M * ϕ[:, i]
        ϕ[:, i] ./= sqrt(c)
    end
end
function MORFE.Eigenproblems.solve(model::NDOrderModel, s::MechSolver)
    K, M = model.linear_terms[1], model.linear_terms[3]
    FK = factorize(K)
    vals_inv, vecs, _ = eigsolve(x -> FK \ (M * x), ones(size(K,1)), s.nev, :LR;
                                  krylovdim=max(3s.nev+1,30), maxiter=500,
                                  tol=1e-10, issymmetric=true)
    ω2  = real.(1 ./ vals_inv[1:s.nev])
    idx = sortperm(ω2); ω2 = ω2[idx]
    ϕ   = real.(hcat(vecs[idx]...))
    mass_normalize!(ϕ, M, s.nev)
    ω = sqrt.(ω2)
    λ = zeros(ComplexF64, s.nev * 2)
    for i in 1:s.nev
        ξ = 0.5 * (s.α / ω[i] + s.β * ω[i])
        λ[2i-1] = ω[i] * (-ξ + sqrt(Complex(1 - ξ^2)) * im)
        λ[2i]   = ω[i] * (-ξ - sqrt(Complex(1 - ξ^2)) * im)
    end
    ev = Matrix{ComplexF64}(undef, FOM * 2, s.nev * 2)
    for i in 1:s.nev
        ev[1:FOM,       2i-1] .= ϕ[:, i]; ev[1:FOM,       2i] .= ϕ[:, i]
        ev[(FOM+1):end, 2i-1] .= λ[2i-1] .* ϕ[:, i]
        ev[(FOM+1):end, 2i  ] .= λ[2i  ] .* ϕ[:, i]
    end
    s.right_eig_result = ev; s.eigenvalues = λ
    return λ, ev
end
function MORFE.Eigenproblems.solve_left(model::NDOrderModel, s::MechSolver)
    ev = s.right_eig_result
    lev = similar(ev)
    n2 = size(ev, 2) ÷ 2
    for i in 1:n2
        lev[(FOM+1):end, 2i-1] = ev[1:FOM, 2i]
        lev[(FOM+1):end, 2i  ] = ev[1:FOM, 2i-1]
    end
    for (i, λ) in enumerate(s.eigenvalues)
        lev[1:FOM, i] = -(1/conj(λ)) * model.linear_terms[1]' * lev[(FOM+1):end, i]
    end
    return s.eigenvalues, lev
end

ROM = 2; N_EXT = 0; NVAR = ROM + N_EXT
eig = compute_eigenproblem(model,
    solver = MechSolver(nothing, nothing, 1, info.α, info.β),
    sorter! = (args...) -> nothing, normalizer! = (args...) -> nothing)
eigenvalues, Y, _ = get_eigenpairs(eig)
select_master_modes_by_sorting(eig, ROM)
master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes       = Y[1:FOM, 1:ROM]
left_eigenmodes    = Y[(FOM+1):end, 1:ROM]
ORD_model = length(model.linear_terms) - 1
mmd = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
for r in 1:ROM, k in 1:(ORD_model-1)
    mmd[:, k, r] .= Y[(k*FOM+1):((k+1)*FOM), r]
end

outer_eigenvalues  = eigenvalues[(ROM+1):end]
super_eigenvalues  = Vector{ComplexF64}(master_eigenvalues)
target_eigenvalues = Vector{ComplexF64}(master_eigenvalues)
max_degree = 3
mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)
resonance_set = resonance_set_from_complex_normal_form_style(
    ROM, mset, super_eigenvalues, target_eigenvalues, 0.05)

# ===========================================================================
# Warm-up (forces JIT compilation, not counted in profile)
# ===========================================================================
println("JIT warm-up…")
W, R = solve_cohomological_problem(model, mset, master_eigenvalues,
    master_modes, left_eigenmodes, resonance_set;
    master_modes_derivatives = mmd)
println("Warm-up done.\n")

# ===========================================================================
# Profile
# ===========================================================================
println("Profiling…")
Profile.clear()
@profile for _ in 1:3
    solve_cohomological_problem(model, mset, master_eigenvalues,
        master_modes, left_eigenmodes, resonance_set;
        master_modes_derivatives = mmd)
end
println("Profile collected.")

# ===========================================================================
# Save flame graph to HTML
# ===========================================================================
outfile = joinpath(@__DIR__, "profile_after_opt.html")
pd = ProfileCanvas.view(Profile.fetch(); C = true)
html_snippet = sprint(show, MIME"text/html"(), pd)
open(outfile, "w") do io
    write(io, """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>MORFE.jl — profile after OPT-1…4</title></head><body>
<h2>MORFE.jl solve_cohomological_problem — after OPT-1…4</h2>
$html_snippet
</body></html>""")
end
println("Flame graph saved → $outfile")

# ===========================================================================
# Text profile: top leaf functions sorted by sample count
# ===========================================================================
println("\n", "="^70)
println("TEXT PROFILE  (leaf-node counts, C frames included)")
println("="^70)
Profile.print(; sortedby = :count, mincount = 50, C = true)

println("\n", "="^70)
println("TEXT PROFILE  (Julia-only frames, mincount = 20)")
println("="^70)
Profile.print(; sortedby = :count, mincount = 20, C = false)
