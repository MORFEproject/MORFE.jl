using Test
using LinearAlgebra
using SparseArrays
using StaticArrays

using MORFE.Multiindices: all_multiindices_up_to
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap, linear_first_order_matrices
using MORFE.Resonance: resonance_set_from_complex_normal_form_style
using MORFE.CohomologicalEquations: solve_cohomological_problem
using MORFE.SpectralDecomposition: left_eigenmode_orders_from_slice, SpectralData

# ── Minimal 2-DOF Duffing model ──────────────────────────────────────────────
const _FOM = 2
const _ROM = 2
const _NVAR = 2   # N_EXT = 0

B0 = [2.0 -1.0; -1.0 2.0]
B2 = [1.0 0.0; 0.0 1.0]
B1 = 0.001 * B2

term_cubic = MultilinearMap(
    (res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3),
    (3, 0)
)

_model = NthOrderModel((B0, B1, B2), (term_cubic,))

# ── Spectrum ──────────────────────────────────────────────────────────────
let
    A_eig, B_eig = linear_first_order_matrices(_model)
    global _eig = eigen(A_eig, B_eig)
end

_sorted_idx = sortperm(abs.(_eig.values))
_master_eigenvalues = SVector{_ROM, ComplexF64}(_eig.values[_sorted_idx[1:_ROM]])
_master_modes = _eig.vectors[1:_FOM, _sorted_idx[1:_ROM]]
_left_eigenmodes = _master_modes

_ORD_model = 2  # second-order system → ORD = 2
_master_modes_derivatives = zeros(ComplexF64, _FOM, _ORD_model - 1, _ROM)
for r in 1:_ROM
    orig = _sorted_idx[r]
    for k in 1:(_ORD_model - 1)
        _master_modes_derivatives[:, k, r] .= _eig.vectors[(_FOM + 1):(2 * _FOM), orig]
    end
end

# Lower-order left eigenvector blocks, generated from the physical slice
# (real symmetric system → the position mode is a valid sesquilinear slice).
_left_modes_derivatives = left_eigenmode_orders_from_slice(
    _model.linear_terms, _left_eigenmodes,
    collect(_master_eigenvalues))[:, 1:(_ORD_model - 1), :]

# The bundle the solve consumes: physical slices plus their companion blocks, which is
# exactly what this file already had. `SpectralData` applies the mirrored convention.
_spectral = SpectralData(; eigenvalues = _master_eigenvalues,
    right_modes = _master_modes, right_derivatives = _master_modes_derivatives,
    left_modes = _left_eigenmodes, left_blocks = _left_modes_derivatives)

# ── Multiindex and resonance sets ─────────────────────────────────────────────
const _max_degree = 5
_mset = all_multiindices_up_to(_NVAR, _max_degree; min_degree = 1)
_super_eigenvalues = Vector{ComplexF64}(_master_eigenvalues)

_resonance_set = resonance_set_from_complex_normal_form_style(
    _mset, _super_eigenvalues, 0.05)

const _conj_perm = [2, 1]

# ── Helper: build (primary, secondary) pairs and self-conjugate indices ───────
function _build_pairs(mset, perm)
    L = length(mset)
    NVAR = length(perm)
    mdict = Dict(mset.exponents[i] => i for i in 1:L)
    pairs = Tuple{Int, Int}[]
    self_conj = Int[]
    seen = falses(L)
    for i in 1:L
        seen[i] && continue
        γ = mset.exponents[i]
        Pγ = SVector{NVAR, Int}(ntuple(k -> γ[perm[k]], Val(NVAR)))
        if Pγ == γ
            push!(self_conj, i)
            seen[i] = true
        elseif haskey(mdict, Pγ)
            j = mdict[Pγ]
            if j != i && !seen[j]
                push!(pairs, (i, j))
                seen[i] = seen[j] = true
            end
        end
    end
    return pairs, self_conj
end

_pairs, _self_conj = _build_pairs(_mset, _conj_perm)

# ── Solve once per strategy ───────────────────────────────────────────────────
W_nosym, R_nosym = solve_cohomological_problem(
    _model, _mset, _spectral, _resonance_set; conjugate_permutation = nothing
)

W_expl, R_expl = solve_cohomological_problem(
    _model, _mset, _spectral, _resonance_set; conjugate_permutation = _conj_perm
)

_sparse_model = NthOrderModel(map(sparse, (B0, B1, B2)), (term_cubic,))
_umfpack_options = ParametrisationOptions(
    backend = :umfpack, show_progress = false, verbose = false)
W_umfpack_nosym, R_umfpack_nosym = solve_cohomological_problem(
    _sparse_model, _mset, _spectral, _resonance_set;
    conjugate_permutation = nothing, options = _umfpack_options)
W_umfpack_expl, R_umfpack_expl = solve_cohomological_problem(
    _sparse_model, _mset, _spectral, _resonance_set;
    conjugate_permutation = _conj_perm, options = _umfpack_options)

# =============================================================================
@testset "ConjugateSymmetry" begin
    @testset "W-symmetry: W[P·γ] = conj(W[γ])" begin
        Wc = W_expl.poly.coefficients   # FOM × ORD × L
        for (src, cj) in _pairs
            for j in axes(Wc, 2)
                @test Wc[:, j, cj] ≈ conj.(Wc[:, j, src]) atol=1e-12
            end
        end
    end

    @testset "R-symmetry: R[r, P·γ] = conj(R[perm[r], γ])" begin
        Rc = R_expl.poly.coefficients   # NVAR × L
        for (src, cj) in _pairs
            for r in 1:_ROM
                pr = _conj_perm[r]
                @test Rc[r, cj] ≈ conj(Rc[pr, src]) atol=1e-12
            end
        end
    end

    @testset "Self-conjugate W is real-valued" begin
        Wc = W_expl.poly.coefficients
        for idx in _self_conj
            @test norm(imag.(Wc[:, :, idx])) < 1e-12
        end
    end

    @testset "Parity: explicit perm ≈ no-sym" begin
        @test W_expl.poly.coefficients ≈ W_nosym.poly.coefficients rtol=1e-8
        @test R_expl.poly.coefficients ≈ R_nosym.poly.coefficients rtol=1e-8
        @test W_umfpack_expl.poly.coefficients≈W_expl.poly.coefficients rtol=1e-10
        @test R_umfpack_expl.poly.coefficients≈R_expl.poly.coefficients rtol=1e-10
        @test W_umfpack_nosym.poly.coefficients≈W_nosym.poly.coefficients rtol=1e-10
        @test R_umfpack_nosym.poly.coefficients≈R_nosym.poly.coefficients rtol=1e-10
    end
end
