using Test
using MORFE
using LinearAlgebra
using Random
using StaticArrays: SVector

using MORFE.MultilinearTerms: build_multilinear_terms_cache, compute_multilinear_terms!
using MORFE.ParametrisationMethod: create_parametrisation_method_objects

# ─────────────────────────────────────────────────────────────────────────────
# Flux summation in the single-pass assembly.
#
# Terms that opt in through `fem_flux_zero` have the single pass sum the fluxes of every
# factorisation of every term at a quadrature point and test the sum once with the driver's
# `contract_qp!`, instead of calling `accumulate_qp!` per factorisation. The right-hand side
# must not change.
#
# The stub has three elements sharing dofs, two quadrature points with distinct weights,
# and a two-component flux tested against element- and point-dependent shapes, so a
# weight, shape or point applied to the wrong factorisation shows up. Its `accumulate_qp!`
# is written out on its own rather than through `flux_qp`, so the identity between the two
# routes is tested, not assumed. The single pass scatters, weights, tests and assembles with
# its driver term, so every term of a model shares the mesh data (`_FLUX_W`, `_FLUX_DΩ`,
# `_FLUX_SHAPE`), says so through `fem_element_loop`, and owns only its flux.
# ─────────────────────────────────────────────────────────────────────────────

const _FLUX_FOM = 4
const _FLUX_NQP = 2
const _FLUX_CELLS = ([1, 2, 3], [2, 3, 4], [4, 1, 2])   # overlapping, so assembly sums
const _FLUX_W = randn(Xoshiro(100), ComplexF64, _FLUX_FOM, _FLUX_NQP, length(_FLUX_CELLS))
const _FLUX_DΩ = 0.5 .+ rand(Xoshiro(101), _FLUX_NQP, length(_FLUX_CELLS))
const _FLUX_SHAPE = randn(
    Xoshiro(102), ComplexF64, length(first(_FLUX_CELLS)), 2, _FLUX_NQP,
    length(_FLUX_CELLS))

"`SUM`: whether the term opts in to flux summation. `ME`: its external multiplicity."
struct _FluxStubTerm{SUM, ME} <: MORFE.FEMMultilinearMap{2}
    multiindex::NTuple{2, Int}
    multiplicity_external::Int
    deg::Int
    fully_asymmetric::Union{Nothing, Bool}
    shape::Array{ComplexF64, 4}      # (dof, flux component, qp, element)
    c::Vector{ComplexF64}            # second flux component over the first, per qp
    dir::Vector{ComplexF64}          # weighting of the external directions; empty if ME = 0
    buffer::Matrix{ComplexF64}
    calls::Vector{Int}               # accumulate_qp!, flux_qp, contract_qp!, assemble_element!
end

# Equal seeds give equal terms, whatever `sum_fluxes`.
function _flux_stub(deg, me, sum_fluxes; seed, n_ext = 0)
    rng = Xoshiro(seed)
    return _FluxStubTerm{sum_fluxes, me}((deg - me, 0), me, deg, false, _FLUX_SHAPE,
        randn(rng, ComplexF64, _FLUX_NQP), randn(rng, ComplexF64, n_ext),
        zeros(ComplexF64, 128, _FLUX_NQP), zeros(Int, 4))
end

MORFE.fem_element_loop(t::_FluxStubTerm) = t.shape   # the shape functions of `contract_qp!`
MORFE.fem_elements(::_FluxStubTerm) = eachindex(_FLUX_CELLS)
MORFE.fem_n_qp(::_FluxStubTerm) = _FLUX_NQP
MORFE.fem_ndofs_per_cell(::_FluxStubTerm) = length(first(_FLUX_CELLS))
MORFE.fem_reinit!(_, ::_FluxStubTerm) = nothing
MORFE.fem_getdetJdV(element, q, ::_FluxStubTerm) = _FLUX_DΩ[q, element]
MORFE.fem_qp_buffer(t::_FluxStubTerm) = t.buffer

function MORFE.scatter_qp!(qp_values, coefficient, element, ::_FluxStubTerm)
    for q in eachindex(qp_values)
        qp_values[q] = transpose(@view(_FLUX_W[:, q, element])) * coefficient
    end
end

function MORFE.assemble_element!(result, Fe, element, t::_FluxStubTerm)
    t.calls[4] += 1
    for (r, d) in enumerate(_FLUX_CELLS[element])
        result[d] += Fe[r]
    end
end

# The flux of one factorisation is p (1, c_q), with p the product of the internal qp values
# and of the weighted external arguments.
function _flux_scale(t::_FluxStubTerm{S, ME}, args) where {S, ME}
    n = length(args) - ME
    return prod(args[1:n]) *
           prod((transpose(t.dir) * r for r in args[(n + 1):end]); init = one(ComplexF64))
end

function MORFE.accumulate_qp!(
        Fe, qp_args::Tuple, multiplier, element, q, dΩ, t::_FluxStubTerm)
    t.calls[1] += 1
    p = _flux_scale(t, qp_args)
    for r in eachindex(Fe)
        Fe[r] += multiplier * dΩ * p *
                 (t.shape[r, 1, q, element] + t.c[q] * t.shape[r, 2, q, element])
    end
end

MORFE.fem_flux_zero(::_FluxStubTerm{true}) = zero(SVector{2, ComplexF64})

function MORFE.flux_qp(qp_args::Tuple, element, q, t::_FluxStubTerm{true})
    t.calls[2] += 1
    p = _flux_scale(t, qp_args)
    return SVector(p, t.c[q] * p)
end

function MORFE.contract_qp!(Fe, flux, element, q, dΩ, t::_FluxStubTerm{true})
    t.calls[3] += 1
    for r in eachindex(Fe)
        Fe[r] += dΩ *
                 (flux[1] * t.shape[r, 1, q, element] + flux[2] * t.shape[r, 2, q, element])
    end
end

"The same term as an ordinary closure that assembles over the stub mesh by itself."
function _flux_stub_closure(t::_FluxStubTerm{S, ME}) where {S, ME}
    function f!(res, xs...)
        e = prod((transpose(t.dir) * r for r in xs[(t.deg - ME + 1):end]);
            init = one(ComplexF64))
        for (element, dofs) in enumerate(_FLUX_CELLS), q in 1:_FLUX_NQP

            p = e * prod(transpose(@view(_FLUX_W[:, q, element])) * x
            for x in xs[1:(t.deg - ME)])
            for (r, d) in enumerate(dofs)
                res[d] += _FLUX_DΩ[q, element] * p *
                          (t.shape[r, 1, q, element] + t.c[q] * t.shape[r, 2, q, element])
            end
        end
        return res
    end
    return MultilinearMap(f!; multiindex = t.multiindex, multiplicity_external = ME,
        fully_asymmetric = false)
end

"Nonlinear right-hand side of every monomial of `mset`, through the cached solve path."
function _flux_rhs(terms, mset, W; external = nothing)
    K = zeros(ComplexF64, _FLUX_FOM, _FLUX_FOM)
    model = external === nothing ? NthOrderModel((K, K, K), terms) :
            NthOrderModel((K, K, K), terms, external)
    cache = build_multilinear_terms_cache(model, W)
    rhs = [zeros(ComplexF64, _FLUX_FOM) for _ in 1:length(mset)]
    for l in 1:length(mset)
        compute_multilinear_terms!(rhs[l], model, l, W, cache)
    end
    return rhs, cache
end

_flux_close(a, b) = maximum(norm.(a .- b)) <= 1e-12 * maximum(norm, b)

@testset "FEM flux summation" begin
    # Up to degree 5 in two variables: cubic factorisations with multipliers 1, 3 and 6.
    mset = all_multiindices_up_to(2, 5; min_degree = 1)
    W, _ = create_parametrisation_method_objects(mset, 2, _FLUX_FOM, 2, 0, ComplexF64)
    randn!(Xoshiro(7), W.poly.coefficients)

    summed = (_flux_stub(2, 0, true; seed = 1), _flux_stub(3, 0, true; seed = 2))
    tested = (_flux_stub(2, 0, false; seed = 1), _flux_stub(3, 0, false; seed = 2))

    @testset "the stub satisfies the identity of fem_flux_zero" begin
        # Guards everything below: a stub whose two routes differed would be no reference.
        for t in summed
            args = ntuple(_ -> randn(ComplexF64), t.deg)
            Fe1 = zeros(ComplexF64, 3)
            Fe2 = zeros(ComplexF64, 3)
            MORFE.accumulate_qp!(Fe1, args, 3, 2, 1, 0.7, t)
            MORFE.contract_qp!(Fe2, 3 * MORFE.flux_qp(args, 2, 1, t), 2, 1, 0.7, t)
            @test isapprox(Fe1, Fe2; rtol = 1e-13)
        end
        @test MORFE.fem_flux_zero(tested[1]) === nothing
    end

    @testset "summed fluxes reproduce accumulate_qp! and the closure" begin
        rhs_summed, _ = _flux_rhs(summed, mset, W)
        rhs_tested, _ = _flux_rhs(tested, mset, W)
        rhs_closure, _ = _flux_rhs(map(_flux_stub_closure, tested), mset, W)
        @test maximum(norm, rhs_closure) > 1e-6   # not vacuous
        @test _flux_close(rhs_tested, rhs_closure)
        @test _flux_close(rhs_summed, rhs_closure)
    end

    @testset "one contraction and one assembly per point and element for all terms" begin
        foreach(t -> fill!(t.calls, 0), (summed..., tested...))
        _, cache = _flux_rhs(summed, mset, W)
        _flux_rhs(tested, mset, W)
        active = count(l -> !isempty(only(cache.fem_groups).cols[l]), 1:length(mset))
        @test all(
            t_idx -> any(l -> !isempty(only(cache.fem_groups).entries[l][t_idx]), 1:length(mset)),
            1:2)
        for t_idx in 1:2
            s, a = summed[t_idx], tested[t_idx]
            @test s.calls[1] == 0                  # never accumulate_qp!
            @test s.calls[2] == a.calls[1] > 0     # one flux per factorisation and point
            @test a.calls[2] == a.calls[3] == 0
        end
        # the driver, the first term, tests the summed fluxes of both terms once per element
        # and point, and assembles once per element for both; the second does neither
        for route in (summed, tested)
            @test route[1].calls[4] == length(_FLUX_CELLS) * active
            @test route[2].calls[3] == route[2].calls[4] == 0
        end
        @test summed[1].calls[3] == length(_FLUX_CELLS) * _FLUX_NQP * active
    end

    @testset "summing and non-summing terms share the single pass" begin
        rhs_tested, _ = _flux_rhs(tested, mset, W)
        # a summing driver: the second term integrates itself into the shared element vector
        rhs_mixed, cache = _flux_rhs((summed[1], tested[2]), mset, W)
        @test any(l -> all(!isempty, only(cache.fem_groups).entries[l]), 1:length(mset))
        @test _flux_close(rhs_mixed, rhs_tested)
        # a non-summing driver: no flux is summed, so the second term integrates itself too
        fill!(summed[2].calls, 0)
        rhs_mixed, _ = _flux_rhs((tested[1], summed[2]), mset, W)
        @test _flux_close(rhs_mixed, rhs_tested)
        @test summed[2].calls[1] > 0
        @test summed[2].calls[2] == summed[2].calls[3] == 0
    end

    @testset "direct evaluation keeps accumulate_qp!" begin
        xs = (randn(Xoshiro(5), ComplexF64, _FLUX_FOM), zeros(ComplexF64, _FLUX_FOM))
        for t_idx in 1:2
            fill!(summed[t_idx].calls, 0)
            res = zeros(ComplexF64, _FLUX_FOM)
            reference = zeros(ComplexF64, _FLUX_FOM)
            MORFE.MultilinearMaps.evaluate_term!(res, summed[t_idx], xs, nothing)
            MORFE.MultilinearMaps.evaluate_term!(
                reference, _flux_stub_closure(tested[t_idx]), xs, nothing)
            @test isapprox(res, reference; rtol = 1e-12)
            @test summed[t_idx].calls[1] > 0
            @test summed[t_idx].calls[3] == 0
        end
    end
end #@testset "FEM flux summation"

@testset "FEM flux summation with an external factor" begin
    # Factorisations with an external factor join the same single pass and flux sum.
    ROM, N_EXT = 2, 2
    mset = all_multiindices_up_to(ROM + N_EXT, 3; min_degree = 1)
    W, _ = create_parametrisation_method_objects(
        mset, 2, _FLUX_FOM, ROM, N_EXT, ComplexF64)
    randn!(Xoshiro(11), W.poly.coefficients)
    external = ExternalSystem(ntuple(j -> ComplexF64(j) * im, N_EXT))

    summed = _flux_stub(3, 1, true; seed = 3, n_ext = N_EXT)
    tested = _flux_stub(3, 1, false; seed = 3, n_ext = N_EXT)
    rhs_summed, _ = _flux_rhs((summed,), mset, W; external)
    rhs_tested, _ = _flux_rhs((tested,), mset, W; external)
    rhs_closure, _ = _flux_rhs((_flux_stub_closure(tested),), mset, W; external)

    @test maximum(norm, rhs_closure) > 1e-6
    @test _flux_close(rhs_tested, rhs_closure)
    @test _flux_close(rhs_summed, rhs_closure)
    @test summed.calls[1] == 0
    @test summed.calls[2] == tested.calls[1] > 0
    @test summed.calls[3] > 0

    # with an internal term in the same model: one contraction per element and point, and one
    # assembly per element, for both terms
    internal = _flux_stub(2, 0, true; seed = 6)
    internal_tested = _flux_stub(2, 0, false; seed = 6)
    foreach(t -> fill!(t.calls, 0), (internal, summed))
    rhs_both, cache = _flux_rhs((internal, summed), mset, W; external)
    rhs_closure, _ = _flux_rhs(
        map(_flux_stub_closure, (internal_tested, tested)), mset, W; external)
    @test _flux_close(rhs_both, rhs_closure)
    active = count(l -> !isempty(only(cache.fem_groups).cols[l]), 1:length(mset))
    @test any(l -> all(!isempty, only(cache.fem_groups).entries[l]), 1:length(mset))
    @test internal.calls[3] == length(_FLUX_CELLS) * _FLUX_NQP * active
    @test internal.calls[4] == length(_FLUX_CELLS) * active
    @test summed.calls[2] > 0
    @test summed.calls[1] == summed.calls[3] == summed.calls[4] == 0
end

@testset "FEM flux summation with two external factors" begin
    # Two external slots taking different variables are one split counted twice
    # (`ext_count = 2`), which the entry multiplier must carry.
    ROM, N_EXT = 2, 2
    mset = all_multiindices_up_to(ROM + N_EXT, 3; min_degree = 1)
    W, _ = create_parametrisation_method_objects(
        mset, 2, _FLUX_FOM, ROM, N_EXT, ComplexF64)
    randn!(Xoshiro(12), W.poly.coefficients)
    external = ExternalSystem(ntuple(j -> ComplexF64(j) * im, N_EXT))

    summed = _flux_stub(3, 2, true; seed = 8, n_ext = N_EXT)
    tested = _flux_stub(3, 2, false; seed = 8, n_ext = N_EXT)
    rhs_summed, cache = _flux_rhs((summed,), mset, W; external)
    rhs_tested, _ = _flux_rhs((tested,), mset, W; external)
    rhs_closure, _ = _flux_rhs((_flux_stub_closure(tested),), mset, W; external)

    @test any(l -> any(e -> e.multiplier == 2, only(cache.fem_groups).entries[l][1]), 1:length(mset))
    @test maximum(norm, rhs_closure) > 1e-6
    @test _flux_close(rhs_tested, rhs_closure)
    @test _flux_close(rhs_summed, rhs_closure)
end

# ─────────────────────────────────────────────────────────────────────────────
# Terms on different element loops.
#
# The stub term above restricted to a domain of the stub mesh, A = cells 1–2 or B = cell 3,
# or integrated with one quadrature point, as a term on another element type would be. Its
# label names its cells and quadrature (`LABELLED`), or it keeps the default label. Terms on
# different loops must get separate passes; the reference evaluates each term on its own,
# driving its own loop.
# ─────────────────────────────────────────────────────────────────────────────

struct _LoopStubTerm{LABELLED} <: MORFE.FEMMultilinearMap{2}
    multiindex::NTuple{2, Int}
    multiplicity_external::Int
    deg::Int
    fully_asymmetric::Bool
    stub::_FluxStubTerm{true, 0}
    cells::Vector{Int}
    n_qp::Int
    label::Any
end

function _loop_stub(deg, cells; seed, n_qp = _FLUX_NQP, label = (cells, n_qp),
        labelled = true)
    return _LoopStubTerm{labelled}((deg, 0), 0, deg, false, _flux_stub(deg, 0, true; seed),
        cells, n_qp, label)
end

MORFE.fem_element_loop(t::_LoopStubTerm{true}) = t.label
MORFE.fem_elements(t::_LoopStubTerm) = t.cells
MORFE.fem_n_qp(t::_LoopStubTerm) = t.n_qp
MORFE.fem_ndofs_per_cell(t::_LoopStubTerm) = MORFE.fem_ndofs_per_cell(t.stub)
MORFE.fem_qp_buffer(t::_LoopStubTerm) = MORFE.fem_qp_buffer(t.stub)
MORFE.fem_flux_zero(t::_LoopStubTerm) = MORFE.fem_flux_zero(t.stub)
MORFE.fem_reinit!(_, ::_LoopStubTerm) = nothing
MORFE.fem_getdetJdV(element, q, t::_LoopStubTerm) = MORFE.fem_getdetJdV(element, q, t.stub)
function MORFE.scatter_qp!(qp_values, coefficient, element, t::_LoopStubTerm)
    MORFE.scatter_qp!(qp_values, coefficient, element, t.stub)
end
function MORFE.assemble_element!(result, Fe, element, t::_LoopStubTerm)
    MORFE.assemble_element!(result, Fe, element, t.stub)
end
function MORFE.accumulate_qp!(Fe, qp_args::Tuple, multiplier, element, q, dΩ,
        t::_LoopStubTerm)
    MORFE.accumulate_qp!(Fe, qp_args, multiplier, element, q, dΩ, t.stub)
end
function MORFE.flux_qp(qp_args::Tuple, element, q, t::_LoopStubTerm)
    MORFE.flux_qp(
        qp_args, element, q, t.stub)
end
function MORFE.contract_qp!(Fe, flux, element, q, dΩ, t::_LoopStubTerm)
    MORFE.contract_qp!(Fe, flux, element, q, dΩ, t.stub)
end

"Bytes allocated by the right-hand side of monomial `l`, after a warm-up call."
function _rhs_allocations(terms, l, W)
    K = zeros(ComplexF64, _FLUX_FOM, _FLUX_FOM)
    model = NthOrderModel((K, K, K), terms)
    cache = build_multilinear_terms_cache(model, W)
    result = zeros(ComplexF64, _FLUX_FOM)
    compute_multilinear_terms!(result, model, l, W, cache)
    return @allocated compute_multilinear_terms!(result, model, l, W, cache)
end

@testset "FEM terms on different element loops" begin
    mset = all_multiindices_up_to(2, 5; min_degree = 1)
    W, _ = create_parametrisation_method_objects(mset, 2, _FLUX_FOM, 2, 0, ComplexF64)
    randn!(Xoshiro(13), W.poly.coefficients)
    A, B, AB = [1, 2], [3], [1, 2, 3]
    alone(terms) = reduce((x, y) -> x .+ y, (first(_flux_rhs((t,), mset, W))
    for t in terms))
    loops(terms) = length(last(_flux_rhs(terms, mset, W)).fem_groups)
    active = count(l -> sum(mset.exponents[l]) >= 2, 1:length(mset))

    T1, T2 = _loop_stub(2, AB; seed = 1), _loop_stub(3, AB; seed = 2)
    T3, T4 = _loop_stub(2, A; seed = 3), _loop_stub(3, B; seed = 4)
    T5 = _loop_stub(2, A; seed = 5)
    # T1 and T2 restricted to A and to B, with the physics of T1 and T2
    T1A, T1B = _loop_stub(2, A; seed = 1), _loop_stub(2, B; seed = 1)
    T2A, T2B = _loop_stub(3, A; seed = 2), _loop_stub(3, B; seed = 2)
    # T2 on another element type: one quadrature point
    T2_1qp = _loop_stub(3, AB; seed = 2, n_qp = 1)

    @testset "each term over its own cells, in any order" begin
        @test maximum(norm, alone((T3, T4))) > 1e-6   # not vacuous
        for (terms, n_loops) in (((T1, T2, T3, T4), 3), ((T5, T1, T2), 2),
            ((T1, T2, T5), 2), ((T1, T2_1qp), 2))
            rhs, cache = _flux_rhs(terms, mset, W)
            @test length(cache.fem_groups) == n_loops
            @test _flux_close(rhs, alone(terms))
        end
    end

    @testset "maps split along the domains visit each cell once" begin
        visits(terms) = (foreach(t -> fill!(t.stub.calls, 0), terms);
            _flux_rhs(terms, mset, W);
            sum(t -> t.stub.calls[4], terms))
        split = (T1A, T2A, T3, T1B, T2B, T4)
        @test loops(split) == 2
        @test _flux_close(first(_flux_rhs(split, mset, W)), alone((T1, T2, T3, T4)))
        @test visits(split) == length(AB) * active
        @test visits((T1, T2, T3, T4)) > length(AB) * active
    end

    @testset "the default label gives each term its own pass" begin
        unlabelled = (_loop_stub(2, AB; seed = 1, labelled = false),
            _loop_stub(3, AB; seed = 2, labelled = false))
        @test MORFE.fem_element_loop(unlabelled[1]) === unlabelled[1]
        @test loops(unlabelled) == 2
        @test _flux_close(first(_flux_rhs(unlabelled, mset, W)), alone((T1, T2)))
    end

    @testset "terms with one label must agree on the quadrature" begin
        too_coarse = (_loop_stub(2, AB; seed = 1, label = :mesh),
            _loop_stub(3, AB; seed = 2, n_qp = 1, label = :mesh))
        @test_throws ArgumentError _flux_rhs(too_coarse, mset, W)
    end

    @testset "several element loops allocate nothing" begin
        @test _rhs_allocations((T1, T2, T3, T4), length(mset), W) == 0
    end
end #@testset "FEM terms on different element loops"
