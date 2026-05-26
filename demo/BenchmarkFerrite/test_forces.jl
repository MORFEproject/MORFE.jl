"""
test_forces.jl  —  Verify nonlinear force assembly: MORFE.jl Ferrite vs naive reference

Two comparisons are made, both on beam_h27.msh:

  1. Ferrite (MORFE.jl)  vs  Naive reference loop
     Same mesh, same DOF numbering.  Tests the O5 Re/Im optimisation in accumulate_qp!.

  2. MORFE.jl Ferrite  vs  Gridap (Legacy)
     Same mesh loaded in both backends.  DOF orderings differ, so we compare the
     scalar modal projections  φᵣᵀ f₂(φᵢ, φⱼ)  and  φᵣᵀ f₃(φᵢ, φⱼ, φₖ)  which are
     ordering-independent.  Eigenvectors are mass-normalised in both backends.

Run:
  julia --project demo/BenchmarkFerrite/test_forces.jl
"""

import Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "MORFE")
    Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
    Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays",
        "BenchmarkTools", "Gmsh"])
end
Pkg.instantiate()

using MORFE
using MORFE.MultilinearMaps: evaluate_term!
using SparseArrays, LinearAlgebra, Arpack, LinearMaps, StaticArrays, Ferrite, FerriteGmsh, Random

include(joinpath(@__DIR__, "../Ferrite/ferrite_assembly.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Ferrite setup  (shared between both comparisons)
# ─────────────────────────────────────────────────────────────────────────────

const _msh = joinpath(@__DIR__, "beam_h27.msh")
isfile(_msh) || error("Mesh not found: run generate_beam_mesh.jl first.")

println("Loading mesh (Ferrite) …")
grid   = togrid(_msh)
ip     = Lagrange{RefHexahedron, 2}()^3
geo_ip = Lagrange{RefHexahedron, 2}()
qr     = QuadratureRule{RefHexahedron}(3)
cv     = CellValues(qr, ip, geo_ip)

dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch); update!(ch, 0.0)

const E_mat = 160e3; const ν_mat = 0.22; const ρ_mat = 2.32e-3
const λl = (E_mat * ν_mat) / ((1 + ν_mat) * (1 - 2ν_mat))
const μl = E_mat / (2(1 + ν_mat))

K_full = allocate_matrix(dh); M_full = allocate_matrix(dh)
assemble_KM!(K_full, M_full, dh, cv, λl, μl, ρ_mat)

free          = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free        = length(free)
K = K_full[free, free]; M = M_full[free, free]

println("Free DOFs: ", n_free)

# Eigenvectors (real, mass-normalised)
println("Solving eigenproblem …")
(λ_eigs_c, V_eigs) = eigs(K, M; nev = 4, which = :SM, tol = 1e-12)
λ_eigs = real.(λ_eigs_c)
V_eigs = real.(V_eigs)
for j in axes(V_eigs, 2)            # mass-normalise
    V_eigs[:, j] ./= sqrt(V_eigs[:, j]' * M * V_eigs[:, j])
end
println("  First eigenfrequencies (rad/s): ", sqrt.(λ_eigs))

φ = [V_eigs[:, j] for j in 1:4]    # eigenvectors as a list

# ─────────────────────────────────────────────────────────────────────────────
# 2.  MORFE.jl Ferrite terms
# ─────────────────────────────────────────────────────────────────────────────

const N_UNIQ = 10    # more than enough for the tests below
term_quad_F  = FerriteGeometricNonlinearity{2}(dh, cv, free_to_local, n_free, λl, μl;
                                                max_unique_cols = N_UNIQ)
term_cubic_F = FerriteGeometricNonlinearity{3}(dh, cv, free_to_local, n_free, λl, μl;
                                                max_unique_cols = N_UNIQ)

# evaluate_term! wrappers.  For FEMMultilinearMap{ORD=2}, xs=(v, v) puts the same
# vector in both slots; xs[2] is never accessed for multiindex=(2,0) or (3,0).
# All internal argument slots draw from xs[1], so a single input vector suffices.
_f2_morfe!(res, v) = evaluate_term!(res, term_quad_F,  (v, v), nothing)
_f3_morfe!(res, v) = evaluate_term!(res, term_cubic_F, (v, v), nothing)

# Polarisation: f2(v1,v2) from same-vector calls (bilinear, symmetric).
function f2_morfe(v1, v2)
    fpp = zeros(ComplexF64, n_free)
    f11 = zeros(ComplexF64, n_free)
    f22 = zeros(ComplexF64, n_free)
    _f2_morfe!(fpp, v1 .+ v2)
    _f2_morfe!(f11, v1)
    _f2_morfe!(f22, v2)
    return 0.5 .* (fpp .- f11 .- f22)
end

# Polarisation: f3(v1,v2,v3) (trilinear, symmetric) — 8-term formula.
# 6 T(a,b,c) = T(a+b+c) - T(a+b-c) - T(a-b+c) - T(-a+b+c)
#            + T(a-b-c) + T(-a+b-c) + T(-a-b+c) - T(-a-b-c)
function f3_morfe(v1, v2, v3)
    terms = [
        (v1 .+ v2 .+ v3,  1),
        (v1 .+ v2 .- v3, -1),
        (v1 .- v2 .+ v3, -1),
        (.-v1 .+ v2 .+ v3, -1),
        (v1 .- v2 .- v3,  1),
        (.-v1 .+ v2 .- v3,  1),
        (.-v1 .- v2 .+ v3,  1),
        (.-v1 .- v2 .- v3, -1),
    ]
    result = zeros(ComplexF64, n_free)
    tmp    = zeros(ComplexF64, n_free)
    for (w, sign) in terms
        fill!(tmp, 0)
        _f3_morfe!(tmp, w)
        result .+= sign .* tmp
    end
    return result ./ 48   # 8 sign combinations × 6 from trilinear symmetry
end

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Naive reference implementations  (plain Ferrite loop, no O5 trick)
# ─────────────────────────────────────────────────────────────────────────────

function _f2_naive_real!(res, v1r::AbstractVector{Float64}, v2r::AbstractVector{Float64})
    n_dpc = ndofs_per_cell(dh)
    Fe    = zeros(Float64, n_dpc)
    fill!(res, 0.0)
    for element in CellIterator(dh)
        reinit!(cv, element)
        dofs = celldofs(element)
        lv1 = [get(free_to_local, d, 0) == 0 ? 0.0 : v1r[free_to_local[d]] for d in dofs]
        lv2 = [get(free_to_local, d, 0) == 0 ? 0.0 : v2r[free_to_local[d]] for d in dofs]
        fill!(Fe, 0.0)
        for q in 1:getnquadpoints(cv)
            dΩ  = getdetJdV(cv, q)
            ∇u1 = function_gradient(cv, q, lv1)
            ∇u2 = function_gradient(cv, q, lv2)
            E_nl = symmetric(0.25 * (transpose(∇u1) ⋅ ∇u2 + transpose(∇u2) ⋅ ∇u1))
            σEnl = λl * tr(E_nl) * one(E_nl) + 2μl * E_nl
            ε1 = symmetric(∇u1); σε1 = λl * tr(ε1) * one(ε1) + 2μl * ε1
            ε2 = symmetric(∇u2); σε2 = λl * tr(ε2) * one(ε2) + 2μl * ε2
            for r in 1:n_dpc
                ∂Nr = shape_gradient(cv, q, r)
                δε  = symmetric(∂Nr)
                cr1 = symmetric(transpose(∇u1) ⋅ ∂Nr)
                cr2 = symmetric(transpose(∇u2) ⋅ ∂Nr)
                Fe[r] -= dΩ * (δε ⊡ σEnl + 0.5 * (cr1 ⊡ σε2 + cr2 ⊡ σε1))
            end
        end
        for (r, d) in enumerate(dofs)
            l = get(free_to_local, d, 0)
            l != 0 && (res[l] += Fe[r])
        end
    end
end

function f2_naive(v1, v2)
    A, B = real.(v1), imag.(v1)
    C, D = real.(v2), imag.(v2)
    rr = zeros(n_free); _f2_naive_real!(rr, A, C)
    ri = zeros(n_free); _f2_naive_real!(ri, A, D)
    ir = zeros(n_free); _f2_naive_real!(ir, B, C)
    ii = zeros(n_free); _f2_naive_real!(ii, B, D)
    return complex.(rr .- ii, ri .+ ir)
end

function _f3_naive_real!(res, v1r, v2r, v3r)
    n_dpc = ndofs_per_cell(dh)
    Fe    = zeros(Float64, n_dpc)
    fill!(res, 0.0)
    for element in CellIterator(dh)
        reinit!(cv, element)
        dofs = celldofs(element)
        lv1 = [get(free_to_local, d, 0) == 0 ? 0.0 : v1r[free_to_local[d]] for d in dofs]
        lv2 = [get(free_to_local, d, 0) == 0 ? 0.0 : v2r[free_to_local[d]] for d in dofs]
        lv3 = [get(free_to_local, d, 0) == 0 ? 0.0 : v3r[free_to_local[d]] for d in dofs]
        fill!(Fe, 0.0)
        for q in 1:getnquadpoints(cv)
            dΩ  = getdetJdV(cv, q)
            ∇u1 = function_gradient(cv, q, lv1)
            ∇u2 = function_gradient(cv, q, lv2)
            ∇u3 = function_gradient(cv, q, lv3)
            E23 = symmetric(0.25 * (transpose(∇u2) ⋅ ∇u3 + transpose(∇u3) ⋅ ∇u2))
            E13 = symmetric(0.25 * (transpose(∇u1) ⋅ ∇u3 + transpose(∇u3) ⋅ ∇u1))
            E12 = symmetric(0.25 * (transpose(∇u1) ⋅ ∇u2 + transpose(∇u2) ⋅ ∇u1))
            σ23 = λl * tr(E23) * one(E23) + 2μl * E23
            σ13 = λl * tr(E13) * one(E13) + 2μl * E13
            σ12 = λl * tr(E12) * one(E12) + 2μl * E12
            for r in 1:n_dpc
                ∂Nr = shape_gradient(cv, q, r)
                cr1 = symmetric(transpose(∇u1) ⋅ ∂Nr)
                cr2 = symmetric(transpose(∇u2) ⋅ ∂Nr)
                cr3 = symmetric(transpose(∇u3) ⋅ ∂Nr)
                Fe[r] -= dΩ / 3 * (cr1 ⊡ σ23 + cr2 ⊡ σ13 + cr3 ⊡ σ12)
            end
        end
        for (r, d) in enumerate(dofs)
            l = get(free_to_local, d, 0)
            l != 0 && (res[l] += Fe[r])
        end
    end
end

function f3_naive(v1, v2, v3)
    # expand T(A+iB, C+iD, E+iF): 8 real evaluations
    A, B = real.(v1), imag.(v1)
    C, D = real.(v2), imag.(v2)
    E_, F_ = real.(v3), imag.(v3)
    tmp = zeros(n_free)
    re  = zeros(n_free)
    im_ = zeros(n_free)
    # Real part: T(A,C,E) - T(A,D,F) - T(B,C,F) - T(B,D,E)
    _f3_naive_real!(tmp, A, C, E_); re .+= tmp
    _f3_naive_real!(tmp, A, D, F_); re .-= tmp
    _f3_naive_real!(tmp, B, C, F_); re .-= tmp
    _f3_naive_real!(tmp, B, D, E_); re .-= tmp
    # Imaginary part: T(A,C,F) + T(A,D,E) + T(B,C,E) - T(B,D,F)
    _f3_naive_real!(tmp, A, C, F_); im_ .+= tmp
    _f3_naive_real!(tmp, A, D, E_); im_ .+= tmp
    _f3_naive_real!(tmp, B, C, E_); im_ .+= tmp
    _f3_naive_real!(tmp, B, D, F_); im_ .-= tmp
    return complex.(re, im_)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Comparison 1: MORFE.jl Ferrite vs naive reference  (3 random vectors)
# ─────────────────────────────────────────────────────────────────────────────

rng = MersenneTwister(42)
test_vecs = [complex.(randn(rng, n_free), randn(rng, n_free)) for _ in 1:3]

println("\n", "="^70)
println("Comparison 1: MORFE.jl Ferrite vs naive reference  (beam_h27.msh)")
println("="^70)

println("\n--- Quadratic force f₂(v, v) ---")
println("  (same-vector call; tests accumulate_qp! O5 Re/Im decomposition)")
for k in 1:3
    v = test_vecs[k]
    f_morfe = zeros(ComplexF64, n_free); _f2_morfe!(f_morfe, v)
    f_ref   = f2_naive(v, v)
    err = norm(f_morfe .- f_ref) / norm(f_ref)
    println("  k=$k: ‖f₂_morfe(v,v) - f₂_ref(v,v)‖ / ‖f₂_ref‖ = $(round(err, sigdigits=4))")
end

println("\n--- Quadratic force f₂(v₁, v₂) with v₁ ≠ v₂ ---")
println("  (MORFE via polarisation; tests off-diagonal part)")
for k in 1:3
    v1 = test_vecs[k]
    v2 = test_vecs[mod1(k + 1, 3)]
    f_morfe = f2_morfe(v1, v2)
    f_ref   = f2_naive(v1, v2)
    err = norm(f_morfe .- f_ref) / norm(f_ref)
    println("  k=$k: ‖f₂_morfe(v₁,v₂) - f₂_ref(v₁,v₂)‖ / ‖f₂_ref‖ = $(round(err, sigdigits=4))")
end

println("\n--- Cubic force f₃(v, v, v) ---")
println("  (same-vector call; tests cubic accumulate_qp! O5 Re/Im decomposition)")
for k in 1:3
    v = test_vecs[k]
    f_morfe = zeros(ComplexF64, n_free); _f3_morfe!(f_morfe, v)
    f_ref   = f3_naive(v, v, v)
    err = norm(f_morfe .- f_ref) / norm(f_ref)
    println("  k=$k: ‖f₃_morfe(v,v,v) - f₃_ref(v,v,v)‖ / ‖f₃_ref‖ = $(round(err, sigdigits=4))")
end

println("\n--- Cubic force f₃(v₁, v₂, v₃) with v₁ ≠ v₂ ≠ v₃ ---")
println("  (MORFE via polarisation)")
for k in 1:3
    v1 = test_vecs[k]
    v2 = test_vecs[mod1(k + 1, 3)]
    v3 = test_vecs[mod1(k + 2, 3)]
    f_morfe = f3_morfe(v1, v2, v3)
    f_ref   = f3_naive(v1, v2, v3)
    err = norm(f_morfe .- f_ref) / norm(f_ref)
    println("  k=$k: ‖f₃_morfe(v₁,v₂,v₃) - f₃_ref(v₁,v₂,v₃)‖ / ‖f₃_ref‖ = $(round(err, sigdigits=4))")
end

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Comparison 2: MORFE.jl Ferrite vs Gridap (Legacy)
#     Modal projections φᵣᵀ f(φᵢ, …) are DOF-ordering independent.
# ─────────────────────────────────────────────────────────────────────────────

println("\n", "="^70)
println("Comparison 2: modal projections  Ferrite vs Gridap (Legacy)")
println("  mesh: beam_h27.msh loaded in both backends")
println("="^70)

# Ferrite modal projections (using eigenvectors computed above, complex cast)
println("\n[Ferrite] computing modal force projections …")
φ1c = complex.(φ[1])   # mass-normalised first eigenvector as ComplexF64

f2_ferrite_111 = zeros(ComplexF64, n_free)
_f2_morfe!(f2_ferrite_111, φ1c)
proj_f2_111_ferrite = dot(φ1c, f2_ferrite_111)

f3_ferrite_1111 = zeros(ComplexF64, n_free)
_f3_morfe!(f3_ferrite_1111, φ1c)
proj_f3_1111_ferrite = dot(φ1c, f3_ferrite_1111)

println("  φ₁ᵀ f₂(φ₁,φ₁)     [Ferrite] = ", proj_f2_111_ferrite)
println("  φ₁ᵀ f₃(φ₁,φ₁,φ₁)  [Ferrite] = ", proj_f3_1111_ferrite)

# Gridap backend — load same mesh
println("\n[Gridap] loading beam_h27.msh …")
try
    using Gridap, GridapGmsh

    gmsh_model = GmshDiscreteModel(_msh)

    deg_g  = 2 * 2   # 2× spatial order
    Ω_g    = Triangulation(gmsh_model)
    dΩ_g   = Measure(Ω_g, deg_g)
    reffe_g = ReferenceFE(lagrangian, VectorValue{3, Float64}, 2)
    V_g    = TestFESpace(gmsh_model, reffe_g; conformity = :H1,
                          dirichlet_tags = ["Dirichlet"],
                          dirichlet_masks = [(true, true, true)])
    U_g    = TrialFESpace(V_g, x -> VectorValue(0.0, 0.0, 0.0))

    sym_g(u) = 0.5 * (u + u')
    σ_lin(ε) = λl * tr(ε) * one(ε) + 2μl * ε
    E_nl_g(u1, u2) = 0.25 * ((∇(u1)') ⋅ ∇(u2) + (∇(u2)') ⋅ ∇(u1))
    σ_nln(ε) = λl * tr(ε) * one(TensorValue{3, 3, Float64}) + 2μl * ε

    a_g(u, v) = ∫(ε(v) ⊙ (σ_lin ∘ ε(u)))dΩ_g
    m_g(u, v) = ∫(ρ_mat * u ⋅ v)dΩ_g

    K_g = assemble_matrix((u, v) -> a_g(u, v), U_g, V_g)
    M_g = assemble_matrix((u, v) -> m_g(u, v), U_g, V_g)
    n_g = size(K_g, 1)
    println("  Gridap free DOFs: ", n_g)

    (λ_g_c, V_g_eigs) = eigs(K_g, M_g; nev = 4, which = :SM, tol = 1e-12)
    V_g_eigs = real.(V_g_eigs)
    for j in axes(V_g_eigs, 2)
        V_g_eigs[:, j] ./= sqrt(V_g_eigs[:, j]' * M_g * V_g_eigs[:, j])
    end
    println("  First eigenfrequencies (rad/s): ", sqrt.(real.(λ_g_c)))

    φ1g = V_g_eigs[:, 1]   # mass-normalised first eigenvector (Gridap DOF order)

    function g_quad_gridap(u1, u2, v)
        ∫(ε(v) ⊙ (σ_nln(E_nl_g(u1, u2))) +
          0.5 * (sym_g(∇(u1)' ⋅ ∇(v)) ⊙ σ_nln(ε(u2))
                 + sym_g(∇(u2)' ⋅ ∇(v)) ⊙ σ_nln(ε(u1))))dΩ_g
    end
    function h_cube_gridap(u1, u2, u3, v)
        1 / 3 * ∫(sym_g(∇(u1)' ⋅ ∇(v)) ⊙ σ_nln(E_nl_g(u2, u3)) +
                   sym_g(∇(u2)' ⋅ ∇(v)) ⊙ σ_nln(E_nl_g(u1, u3)) +
                   sym_g(∇(u3)' ⋅ ∇(v)) ⊙ σ_nln(E_nl_g(u1, u2)))dΩ_g
    end

    function f2_gridap(u, w)
        ur = FEFunction(U_g, real(u)); ui = FEFunction(U_g, imag(u))
        wr = FEFunction(U_g, real(w)); wi = FEFunction(U_g, imag(w))
        re = assemble_vector(v -> g_quad_gridap(ur, wr, v) - g_quad_gridap(ui, wi, v), V_g)
        im = assemble_vector(v -> g_quad_gridap(ur, wi, v) + g_quad_gridap(ui, wr, v), V_g)
        return -complex.(re, im)
    end

    function f3_gridap(u, v_in, w)
        ur = FEFunction(U_g, real(u)); ui = FEFunction(U_g, imag(u))
        vr = FEFunction(U_g, real(v_in)); vi_ = FEFunction(U_g, imag(v_in))
        wr = FEFunction(U_g, real(w)); wi = FEFunction(U_g, imag(w))
        re = assemble_vector(
            tv -> h_cube_gridap(ur, vr, wr, tv) - h_cube_gridap(ur, vi_, wi, tv) -
                  h_cube_gridap(ui, vr, wi, tv) - h_cube_gridap(ui, vi_, wr, tv), V_g)
        im_ = assemble_vector(
            tv -> h_cube_gridap(ui, vr, wr, tv) + h_cube_gridap(ur, vi_, wr, tv) +
                  h_cube_gridap(ur, vr, wi, tv) - h_cube_gridap(ui, vi_, wi, tv), V_g)
        return -complex.(re, im_)
    end

    φ1gc = complex.(φ1g)
    f2_g111 = f2_gridap(φ1gc, φ1gc)
    proj_f2_111_gridap = dot(φ1gc, f2_g111)

    f3_g1111 = f3_gridap(φ1gc, φ1gc, φ1gc)
    proj_f3_1111_gridap = dot(φ1gc, f3_g1111)

    println("  φ₁ᵀ f₂(φ₁,φ₁)     [Gridap]  = ", proj_f2_111_gridap)
    println("  φ₁ᵀ f₃(φ₁,φ₁,φ₁)  [Gridap]  = ", proj_f3_1111_gridap)

    println("\n--- Relative differences (Ferrite vs Gridap) ---")
    Δf2 = abs(proj_f2_111_ferrite - proj_f2_111_gridap) / abs(proj_f2_111_gridap)
    Δf3 = abs(proj_f3_1111_ferrite - proj_f3_1111_gridap) / abs(proj_f3_1111_gridap)
    println("  |Δ(φ₁ᵀ f₂(φ₁,φ₁))|  / |ref| = $(round(Δf2, sigdigits=4))")
    println("  |Δ(φ₁ᵀ f₃(φ₁,φ₁,φ₁))| / |ref| = $(round(Δf3, sigdigits=4))")

catch e
    println("  Gridap comparison skipped: ", e)
end

println("\n" * "="^70)
println("Done.")
