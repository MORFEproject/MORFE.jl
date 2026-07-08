# Tests for the MORFEStructuralSVK extension (high-level SVK + Ferrite UI).
# Gate A (small): high-level parametrise ≡ hand-built ex01-style pipeline.
# Gate B (small): forced(amplitude = 0) ≡ autonomous on shared monomials.
# Requires: Ferrite, FerriteGmsh, Arpack, LinearMaps (test extras).

using MORFE
using Ferrite, FerriteGmsh, Arpack, LinearMaps
using SparseArrays, LinearAlgebra, StaticArrays
using Test

const SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)
@test SVK !== nothing

# ── Tiny in-memory beam (subparametric: Hex8 geometry, quadratic field) ──────
function _test_grid()
    grid = generate_grid(Hexahedron, (4, 1, 1), Vec(0.0, 0.0, 0.0), Vec(100.0, 5.0, 5.0))
    addfacetset!(grid, "Dirichlet",
        x -> abs(x[1]) < 1e-8 || abs(x[1] - 100.0) < 1e-8)
    return grid
end

const _E = 160e3
const _ν = 0.22
const _ρ = 2.32e-3
const _α = 1e-3
const _β = 1e-4
const _order = 3

@testset "high-level vs hand-built (Gate A, small)" begin
    grid = _test_grid()
    beam = SVK.mechanical_model(grid;
        material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
        damping = SVK.RayleighDamping(α = _α, β = _β),
        dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)
    rom = SVK.parametrise(beam; master = [1], order = _order)

    # Hand-built reference: ex01 §§1–8 inline on an identical grid.
    grid2 = _test_grid()
    ip = Lagrange{RefHexahedron, 2}()^3
    qr = QuadratureRule{RefHexahedron}(3)
    cv = CellValues(qr, ip)
    dh = DofHandler(grid2)
    add!(dh, :u, ip)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid2, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
    close!(ch)
    update!(ch, 0.0)

    λm = (_E * _ν) / ((1 + _ν) * (1 - 2_ν))
    μm = _E / (2(1 + _ν))
    K_full = allocate_matrix(dh)
    M_full = allocate_matrix(dh)
    ferrite_assemble_KM!(K_full, M_full, dh, cv, λm, μm, _ρ)
    free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
    free_to_local = Dict(d => i for (i, d) in enumerate(free))
    n_free = length(free)
    K = K_full[free, free]
    M = M_full[free, free]
    C = _α * M + _β * K

    ROM = 2
    mset = all_multiindices_up_to(ROM, _order; min_degree = 1)
    term_quad = ferrite_nonlinearity(
        2, dh, cv, free_to_local, n_free, λm, μm; max_unique_cols = length(mset))
    term_cubic = ferrite_nonlinearity(
        3, dh, cv, free_to_local, n_free, λm, μm; max_unique_cols = length(mset))
    model = NDOrderModel((K, C, M), (term_quad, term_cubic))

    eigenproblem = solve_eigenproblem(model,
        solver = SVK.RayleighEigenSolver(nothing, nothing, 10, _α, _β),
        sorter! = (args...) -> nothing)
    (eigenvalues, Y, X) = get_eigenpairs(eigenproblem)
    select_master_modes_by_sorting(eigenproblem, ROM)
    master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
    master_modes = Y[:, 1, 1:ROM]
    left_eigenmodes = X[:, 1:ROM]
    master_modes_derivatives = zeros(ComplexF64, n_free, 1, ROM)
    for r in 1:ROM
        master_modes_derivatives[:, 1, r] .= Y[:, 2, r]
    end
    left_modes_derivatives = eigenproblem.left_eigenmodes_orders[:, 1:1, 1:ROM]
    resonance_set = resonance_set_from_complex_normal_form_style(
        mset, Vector{ComplexF64}(master_eigenvalues), 0.05)
    W_ref,
    R_ref = solve_cohomological_problem(model, mset, master_eigenvalues,
        master_modes, left_eigenmodes, resonance_set;
        master_modes_derivatives = master_modes_derivatives,
        left_modes_derivatives = left_modes_derivatives,
        conjugate_permutation = [2, 1])

    @test beam.info.n_dofs == n_free
    a = rom.R.poly.coefficients
    b = R_ref.poly.coefficients
    @test size(a) == size(b)
    dev = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-12))
    @info "Gate A (small) max rel dev = $dev"
    @test dev < 1e-10
end

@testset "zero-amplitude forcing consistency (Gate B, small)" begin
    grid = _test_grid()
    beam = SVK.mechanical_model(grid;
        material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
        damping = SVK.RayleighDamping(α = _α, β = _β),
        dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)

    rom0 = SVK.parametrise(beam; master = [1], order = _order)
    romf = SVK.parametrise(beam; master = [1], order = _order,
        forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.0))

    @test romf.info.N_EXT == 2
    @test romf.info.Ω ≈ abs(rom0.eigenvalues[1])

    exps0 = rom0.R.poly.multiindex_set.exponents
    expsf = romf.R.poly.multiindex_set.exponents
    lookup = Dict(e => i for (i, e) in enumerate(expsf))
    maxdev = 0.0
    for (i, e) in enumerate(exps0)
        target = SVector{4, Int}(e[1], e[2], 0, 0)
        j = get(lookup, target, nothing)
        @test j !== nothing
        c0 = rom0.R.poly.coefficients[:, i]
        cf = romf.R.poly.coefficients[1:2, j]
        maxdev = max(maxdev, maximum(abs.(c0 .- cf) ./ max.(abs.(c0), 1e-12)))
    end
    @info "Gate B (small) max rel dev = $maxdev"
    @test maxdev < 1e-10
end

@testset "forced ROM is non-trivial at finite amplitude" begin
    grid = _test_grid()
    beam = SVK.mechanical_model(grid;
        material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
        damping = SVK.RayleighDamping(α = _α, β = _β),
        dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)
    romf = SVK.parametrise(beam; master = [1], order = _order,
        forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.1))
    expsf = romf.R.poly.multiindex_set.exponents
    ext_cols = [i for (i, e) in enumerate(expsf) if e[3] + e[4] > 0]
    @test !isempty(ext_cols)
    # The forcing must inject SOMETHING into the external-monomial coefficients.
    @test maximum(abs.(romf.R.poly.coefficients[:, ext_cols])) > 0
end

@testset "error paths" begin
    grid = _test_grid()
    beam = SVK.mechanical_model(grid;
        material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
        damping = SVK.RayleighDamping(α = _α, β = _β),
        dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)
    @test_throws AssertionError SVK.parametrise(beam; master = [2], order = 3)
    @test_throws AssertionError SVK.parametrise(beam; master = [1], order = 3,
        forcing = SVK.HarmonicForcing(mode = 2, amplitude = 0.1))
end
