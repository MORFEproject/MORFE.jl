# The unified entry point `parametrise(model, spectral, expansion_order)` must reproduce
# the older `parametrise(model, order, spectrum)` path exactly.
#
# Bit-equality, not `≈`: this is a plumbing change, so any difference at all is a bug.

using Test
using LinearAlgebra
using StaticArrays

using MORFE
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver,
                                   select_master_modes_by_sorting, SpectralData
using MORFE.Resonance: ResonanceConfig

@testset "parametrise (unified entry point)" begin
    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = [1.0 0.0; 0.0 1.0]
    B1 = 0.001 * B2
    cubic = MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), (3, 0))
    model = NthOrderModel((B0, B1, B2), (cubic,))
    ROM, order = 2, 5

    sp = spectrum(model; solver = DefaultEigensolver())
    select_master_modes_by_sorting(sp, ROM)
    idx = findall(sp.master_modes)
    sd = SpectralData(model, sp; master = idx)
    cnf = ResonanceConfig(style = :complex_normal_form, tol = 0.05, warn_outer = false)

    @testset "≡ the (model, order, spectrum) path" begin
        Wo, Ro = parametrise(model, order, sp;
            resonance = :complex_normal_form, resonance_tol = 0.05)
        Wn, Rn = parametrise(model, sd, order; resonance = cnf)
        @test Wo.poly.coefficients == Wn.poly.coefficients
        @test Ro.poly.coefficients == Rn.poly.coefficients
    end

    @testset "expansion_order dispatches on its type" begin
        Wi, Ri = parametrise(model, sd, order; resonance = cnf)
        mset = all_multiindices_up_to(ROM, order; min_degree = 1)
        Wm, Rm = parametrise(model, sd, mset; resonance = cnf)
        @test Wm.poly.coefficients == Wi.poly.coefficients
        @test Rm.poly.coefficients == Ri.poly.coefficients

        # A restricted set is a genuine restriction, not silently ignored.
        small = all_multiindices_up_to(ROM, 3; min_degree = 1)
        Ws, _ = parametrise(model, sd, small; resonance = cnf)
        @test size(Ws.poly.coefficients, 3) == length(small)

        # Anything else names what is accepted rather than raising a MethodError.
        @test_throws ArgumentError parametrise(model, sd, "five")
        @test_throws ArgumentError parametrise(model, sd, 2.5)
    end

    @testset "conjugate symmetry is carried by the bundle" begin
        sdc = SpectralData(model, sp; master = idx, conjugate_permutation = :detect)
        Wc, Rc = parametrise(model, sdc, order; resonance = cnf)
        # Identical to passing the literal through the old entry point.
        Wr, Rr = parametrise(model, order, sp;
            resonance = :complex_normal_form, resonance_tol = 0.05,
            conjugate_permutation = [2, 1])
        @test Wc.poly.coefficients == Wr.poly.coefficients
        @test Rc.poly.coefficients == Rr.poly.coefficients

        # ...and overridable per solve without rebuilding the bundle.
        Wn, _ = parametrise(model, sdc, order; resonance = cnf,
            conjugate_permutation = nothing)
        Wp, _ = parametrise(model, sd, order; resonance = cnf)
        @test Wn.poly.coefficients == Wp.poly.coefficients
    end

    @testset "a prebuilt ResonanceSet is used verbatim" begin
        mset = all_multiindices_up_to(ROM, order; min_degree = 1)
        rset = MORFE.Resonance.build_resonance_set(model, mset, sd, cnf)
        Wa, Ra = parametrise(model, sd, mset; resonance = rset)
        Wb, Rb = parametrise(model, sd, mset; resonance = cnf)
        @test Wa.poly.coefficients == Wb.poly.coefficients
        @test Ra.poly.coefficients == Rb.poly.coefficients
    end

    @testset "setup banner" begin
        mset = all_multiindices_up_to(ROM, order; min_degree = 1)
        sdc = SpectralData(model, sp; master = idx, conjugate_permutation = :detect)

        banner = sprint(io -> print_setup(io, model, sdc, mset, cnf))
        @test occursin("MORFE parametrisation", banner)
        @test occursin("FOM = 2", banner)
        @test occursin("ROM = 2,  NVAR = 2", banner)
        @test occursin("$(length(mset)) monomials", banner)
        @test occursin("complex_normal_form", banner)
        @test occursin("master [2, 1]", banner)
        # Conjugate pairs are collapsed rather than listed twice.
        @test count("±", banner) == 1

        # `parametrise` routes through it when a destination is named ...
        buf = IOBuffer()
        parametrise(model, sdc, order; resonance = cnf, setup_io = buf,
            show_progress = false)
        @test occursin("MORFE parametrisation", String(take!(buf)))

        # ... and stays silent when asked to, or when the destination is the default
        # `stderr` under test (never a TTY), which is what keeps existing logs unchanged.
        quiet = IOBuffer()
        parametrise(model, sdc, order; resonance = cnf, setup_io = quiet,
            verbose = false, show_progress = false)
        @test isempty(take!(quiet))
        # The gate: `stderr` follows the TTY rule the progress reporter uses; a named
        # destination is always written to.
        gate = MORFE.ParametrisationMethod._setup_output_enabled
        @test gate(true, stderr) == (stderr isa Base.TTY)
        @test gate(true, IOBuffer())
        @test !gate(false, IOBuffer())
    end
end
