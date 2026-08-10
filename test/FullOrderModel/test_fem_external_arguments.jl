using Test
using MORFE
using LinearAlgebra
using StaticArrays: SVector

using MORFE.MultilinearTerms: build_multilinear_terms_cache, compute_multilinear_terms!
using MORFE.ParametrisationMethod: create_parametrisation_method_objects

# ─────────────────────────────────────────────────────────────────────────────
# A minimal FEM backend with an external factor.
#
# Terms with `multiplicity_external > 0` were previously the one shape the two evaluation
# paths disagreed on: `evaluate_term!`'s direct FEM overload passed `r` into the trailing
# argument slots, while the cached solve path (`_replay_fem_split!`) built its qp tuple from
# the internal slots only and applied the external multiplicity as a bare scalar.  A term
# whose integrand actually *depends* on which external direction it is handed therefore got
# different answers from the two paths — and, once external systems could be re-based, would
# never see the change of basis at all.
#
# There is no FEM backend inside MORFE (they live in MORFEFerrite), so the regression test
# needs its own.  This one is as small as it can be: one element, one quadrature point,
# scalar qp values, and an integrand deliberately built to distinguish external directions.
# ─────────────────────────────────────────────────────────────────────────────

const _FEM_FOM = 3
const _FEM_NQP = 1

"""
`g(u, r) = (w · u) * (dir · r) * shape`, with `dir` chosen so the two external directions
give genuinely different results.
"""
struct _StubFEMTerm{N_EXT} <: MORFE.FEMMultilinearMap{2}
    multiindex::NTuple{2, Int}
    multiplicity_external::Int
    deg::Int
    fully_asymmetric::Union{Nothing, Bool}
    w::Vector{ComplexF64}          # qp projection of a W column
    shape::Vector{ComplexF64}      # element residual shape
    dir::Vector{ComplexF64}        # weighting of the external directions
    buffer::Matrix{ComplexF64}     # qp scratch, owned by the term
end

function _stub_fem_term(N_EXT::Int; max_unique::Int = 4)
    w = ComplexF64[1.0, -0.5, 0.25]
    shape = ComplexF64[0.3, 1.0, -0.7]
    # Distinct entries: an integrand that ignored the external direction would make the
    # two directions indistinguishable and the test vacuous.
    dir = ComplexF64[2.0, -3.0, 0.5][1:N_EXT]
    return _StubFEMTerm{N_EXT}(
        (1, 0), 1, 2, false, w, shape, dir,
        zeros(ComplexF64, max_unique, _FEM_NQP))
end

MORFE.fem_elements(::_StubFEMTerm) = 1:1
MORFE.fem_n_qp(::_StubFEMTerm) = _FEM_NQP
MORFE.fem_ndofs_per_cell(::_StubFEMTerm) = _FEM_FOM
MORFE.fem_reinit!(_, ::_StubFEMTerm) = nothing
MORFE.fem_getdetJdV(_, _, ::_StubFEMTerm) = 1.0
MORFE.fem_qp_buffer(t::_StubFEMTerm) = t.buffer

function MORFE.scatter_qp!(∇W_col, W_global, _, t::_StubFEMTerm)
    ∇W_col[1] = dot(conj(t.w), W_global)      # a plain projection, no conjugation intended
    return nothing
end

function MORFE.assemble_element!(accum, Fe, _, ::_StubFEMTerm)
    accum .+= Fe
    return nothing
end

# The load-bearing signature: `∇W_args` must carry the external argument in its trailing
# slot.  Before the fix this method was called with a 1-tuple and would have thrown.
# `Tuple{Any, Any}`, not `NTuple{2}`: the latter means `Tuple{T, T}`, and the internal and
# external slots have genuinely different types.
function MORFE.accumulate_qp!(
        Fe, ∇W_args::Tuple{Any, Any}, mult, _, _, dΩ, t::_StubFEMTerm)
    u_qp, r_ext = ∇W_args
    Fe .+= (mult * dΩ * u_qp * dot(conj(t.dir), r_ext)) .* t.shape
    return nothing
end

"An ordinary closure term computing exactly the same thing, as the reference."
function _stub_closure_term(t::_StubFEMTerm)
    return MultilinearMap(
        (res, u, r) -> (res .+= (dot(conj(t.w), u) *
                                 dot(conj(t.dir), r)) .* t.shape),
        (1, 0), 1; fully_asymmetric = false)
end

"""
Nonlinear right-hand side for every monomial of `mset`, via the cached solve path.

Uses `compute_multilinear_terms!` — the in-place variant — because that is the one the
solve loop calls, and the only one that dispatches FEM terms through `_replay_term!` and
the combined element loop.  Its allocating sibling replays every term through
`_replay_split!`, which assumes a closure.
"""
function _fem_rhs_all_monomials(term, mset, N_EXT, W)
    K = zeros(ComplexF64, _FEM_FOM, _FEM_FOM)
    model = NthOrderModel((K, K, K), (term,),
        ExternalSystem(ntuple(j -> ComplexF64(j) * im, N_EXT)))
    cache = build_multilinear_terms_cache(model, W)
    out = Vector{Vector{ComplexF64}}(undef, length(mset))
    for l in 1:length(mset)
        res = zeros(ComplexF64, _FEM_FOM)
        compute_multilinear_terms!(res, model, l, W, cache)
        out[l] = res
    end
    return out
end

@testset "FEM terms with an external factor" begin
    ROM, N_EXT = 2, 2
    NVAR = ROM + N_EXT
    mset = all_multiindices_up_to(NVAR, 2; min_degree = 1)
    W, _ = create_parametrisation_method_objects(mset, 2, _FEM_FOM, ROM, N_EXT, ComplexF64)
    W.poly.coefficients .= reshape(
        ComplexF64[0.1i + 0.03im * i for i in 1:length(W.poly.coefficients)],
        size(W.poly.coefficients))

    fem_term = _stub_fem_term(N_EXT)
    closure_term = _stub_closure_term(fem_term)

    @testset "cached FEM path agrees with the equivalent closure term" begin
        # The closure path is well covered; making the FEM path reproduce it monomial by
        # monomial is what pins the external argument actually reaching `accumulate_qp!`.
        fem_rhs = _fem_rhs_all_monomials(fem_term, mset, N_EXT, W)
        closure_rhs = _fem_rhs_all_monomials(closure_term, mset, N_EXT, W)

        @test all(isapprox.(fem_rhs, closure_rhs; atol = 1e-12))
        # Not vacuous: some monomial must actually produce a non-zero contribution.
        @test maximum(norm, fem_rhs) > 1e-6
    end

    @testset "the integrand really distinguishes the external directions" begin
        # Guards the test above: with a direction-blind integrand it would pass even if the
        # external argument were never delivered.
        Fe1 = zeros(ComplexF64, _FEM_FOM)
        Fe2 = zeros(ComplexF64, _FEM_FOM)
        MORFE.accumulate_qp!(
            Fe1, (ComplexF64(1.0), SVector{2, Int}(1, 0)), 1, nothing, 1, 1.0, fem_term)
        MORFE.accumulate_qp!(
            Fe2, (ComplexF64(1.0), SVector{2, Int}(0, 1)), 1, nothing, 1, 1.0, fem_term)
        @test norm(Fe1 - Fe2) > 1e-6
    end

    @testset "direct evaluation agrees with the cached path" begin
        # `evaluate_term!` was the path that always passed `r`; the cached one is the path
        # that did not.  They must now agree.
        xs = (ComplexF64[1.0, 2.0, -1.0], ComplexF64[0.0, 0.0, 0.0])
        r = ComplexF64[0.7, -1.1]

        res_fem = zeros(ComplexF64, _FEM_FOM)
        res_closure = zeros(ComplexF64, _FEM_FOM)
        MORFE.MultilinearMaps.evaluate_term!(res_fem, fem_term, xs, r)
        MORFE.MultilinearMaps.evaluate_term!(res_closure, closure_term, xs, r)

        @test res_fem ≈ res_closure
        @test norm(res_fem) > 1e-6
    end
end #@testset "FEM terms with an external factor"
