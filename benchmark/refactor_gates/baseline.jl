# Stage-0 baseline harness for the parametrise/spectral-layer refactor.
#
# Captures, for a set of models chosen to cover every code path the refactor
# touches: the (W, R) coefficient arrays (bit-identical oracle), wall time, and
# allocation count (performance gate).
#
# Deliberately uses DefaultEigensolver (LAPACK `eigen`) everywhere rather than
# Arpack: the Arpack eigenvector gauge is not reproducible across runs, which
# would make a bit-identical comparison meaningless (see project_svk_gate_env_failure
# and project_karman_demo_validation).
#
# Usage:  julia --project=test baseline.jl <outdir>

using MORFE
using LinearAlgebra, SparseArrays, StaticArrays, Serialization, Printf
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver, SpectralData,
                           left_eigenmode_orders_from_slice
using MORFE.CohomologicalEquations: solve_cohomological_problem
using MORFE.Resonance: build_resonance_set, ResonanceConfig

# The resonance policy every model here uses. `outer_targets` reproduces what the
# eigenproblem-based builder did unconditionally; the solve reads only the inner block,
# so it cannot move a coefficient either way.
const CNF = ResonanceConfig(style = :complex_normal_form, tol = 0.05,
    outer_targets = true, warn_outer = false)

const OUTDIR = length(ARGS) >= 1 ? ARGS[1] : @__DIR__
mkpath(OUTDIR)

# ── model builders ───────────────────────────────────────────────────────────

# A 2-DOF Duffing chain: dense, ORD = 2, no external system.
function duffing2(; sparse_matrices = false)
    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = [1.0 0.0; 0.0 1.0]
    B1 = 0.001 * B2
    cubic = MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), (3, 0))
    lt = sparse_matrices ? (sparse(B0), sparse(B1), sparse(B2)) : (B0, B1, B2)
    return NthOrderModel(lt, (cubic,)), (B0, B1, B2)
end

# An N-DOF tridiagonal chain, used for the sparse solver path (KLU).
function chainN(n::Int; sparse_matrices = true)
    B0 = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
    B2 = Matrix(1.0I, n, n)
    B1 = 0.002 * B2
    cubic = MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), (3, 0))
    lt = sparse_matrices ? (sparse(B0), sparse(B1), sparse(B2)) : (B0, B1, B2)
    return NthOrderModel(lt, (cubic,)), (B0, B1, B2)
end

# The eigenproblem every model here is fed from.  Takes a DENSE model
# (DefaultEigensolver calls `eigen(A, B)`, which needs dense matrices) carrying at least
# one nonlinear term so `ORD` is inferrable.
function eigenproblem(dense_lt)
    cubic = MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), (3, 0))
    return spectrum(NthOrderModel(dense_lt, (cubic,)); solver = DefaultEigensolver())
end

# ── measurement ──────────────────────────────────────────────────────────────

results = Dict{String, Any}()

function record!(name, f)
    f()                                   # warm up / compile
    GC.gc()
    allocs = @allocated ((W, R) = f())
    t = @elapsed f()
    t = min(t, @elapsed f())              # best of two, less scheduler noise
    W, R = f()
    serialize(joinpath(OUTDIR, "$(name)_W.jls"), W.poly.coefficients)
    serialize(joinpath(OUTDIR, "$(name)_R.jls"), R.poly.coefficients)
    results[name] = (; time_s = t, bytes = allocs,
        nW = length(W.poly.coefficients), nR = length(R.poly.coefficients))
    @printf("  %-28s  %8.4f s  %12d bytes  |W|=%d\n", name, t, allocs,
        length(W.poly.coefficients))
    return nothing
end

println("Stage-0 baseline  (", OUTDIR, ")")
println(repeat("-", 78))

# M1 — dense ORD=2, N_EXT=0, no conjugate symmetry.
let ROM = 2, order = 7
    model, dense_lt = duffing2()
    ep = eigenproblem(dense_lt)
    sd = SpectralData(model, ep; master = master_by_sorting(ROM))
    mset = all_multiindices_up_to(ROM, order; min_degree = 1)
    rset = build_resonance_set(model, mset, sd, CNF)
    record!("M1_dense_noconj",
        () -> solve_cohomological_problem(model, mset, sd, rset;
            conjugate_permutation = nothing, show_progress = false))

    # M2 — same model, conjugate-symmetry path active.
    record!("M2_dense_conj",
        () -> solve_cohomological_problem(model, mset, sd, rset;
            conjugate_permutation = [2, 1], show_progress = false))
end

# M3 — sparse solver path (KLU), larger chain.
let ROM = 2, order = 5, n = 30
    model, dense_lt = chainN(n)
    ep = eigenproblem(dense_lt)
    sd = SpectralData(model, ep; master = master_by_sorting(ROM))
    mset = all_multiindices_up_to(ROM, order; min_degree = 1)
    rset = build_resonance_set(model, mset, sd, CNF)
    record!("M3_sparse_chain30",
        () -> solve_cohomological_problem(model, mset, sd, rset;
            conjugate_permutation = [2, 1], show_progress = false))
end

# M4 — external system present (N_EXT = 2), conjugate permutation over ROM+N_EXT.
let ROM = 2, order = 5
    model_eig, dense_lt = duffing2()
    ep = eigenproblem(dense_lt)
    Ω = 1.3
    fvec = [1.0, 0.5]
    force = MultilinearMap((res, r) -> begin
            @inbounds for j in 1:2
                iszero(r[j]) || (res .+= r[j] .* fvec)
            end
            res
        end, (0, 0), 1)
    cubic = MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), (3, 0))
    ext = ExternalSystem((im * Ω, -im * Ω))
    model = NthOrderModel(dense_lt, (cubic, force), ext)
    sd = SpectralData(model, ep; master = master_by_sorting(ROM))
    mset = all_multiindices_up_to(ROM + 2, order; min_degree = 1)
    rset = build_resonance_set(model, mset, sd, CNF)
    record!("M4_external_conj",
        () -> solve_cohomological_problem(model, mset, sd, rset;
            conjugate_permutation = [2, 1, 4, 3], show_progress = false))
end

# M5 — ORD-mismatch path: ORD-3 augmented model fed by ORD-2 eigendata,
#      exactly as MORFEFerrite examples 04/07 do it. This is the case the
#      SpectralData reconstruction rule (b) must reproduce.
let ROM = 2, order = 5
    _, dense_lt = duffing2()
    B0, B1, B2 = dense_lt
    Z = zeros(size(B0))
    # ORD = 3 here (4 linear terms), so the multiindex needs 3 entries.
    cubic = MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), (3, 0, 0))
    model = NthOrderModel((B0, B1, B2, Z), (cubic,))

    ep = eigenproblem(dense_lt)
    # `SpectralData` reconciles the order mismatch: right blocks extended by multiplying
    # the LAST AVAILABLE block by λ (not a fresh λ^{k-1}ψ), left blocks rebuilt against
    # the AUGMENTED linear_terms. Both are dumped so the archive still pins them.
    sd = SpectralData(model, ep; master = master_by_sorting(ROM))
    mset = all_multiindices_up_to(ROM, order; min_degree = 1)
    rset = build_resonance_set(model, mset, sd, CNF)
    serialize(joinpath(OUTDIR, "M5_mmd.jls"),
        Array(MORFE.SpectralDecomposition.right_mode_derivatives(sd)))
    serialize(joinpath(OUTDIR, "M5_lmd.jls"),
        Array(MORFE.SpectralDecomposition.left_mode_blocks(sd)))
    record!("M5_ord3_mismatch",
        () -> solve_cohomological_problem(model, mset, sd, rset;
            conjugate_permutation = [2, 1], show_progress = false))
end

serialize(joinpath(OUTDIR, "summary.jls"), results)
println(repeat("-", 78))
println("wrote ", length(results), " model baselines to ", OUTDIR)
