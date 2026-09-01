# Tests for the constant-size bordered cohomological system.
#
# The sparse path used to eliminate L(s) first — forming L(s)⁻¹b and L(s)⁻¹C and
# closing with a dense Schur complement. Inner resonance is flagged by
# |λ_r − s| < tol while det L(λ_r) = 0, so that inverse was being taken precisely
# where L(s) is numerically singular. Both paths now factorise the whole bordered
# matrix instead, at the constant size FOM + ROM.
#
# The dense path has always assembled and factorised the full bordered matrix, so
# "sparse ≡ dense on a model whose L(s) goes singular at its resonances" is the
# sharp test: it is exactly what the old sparse path could not deliver.

using Test
using LinearAlgebra
using SparseArrays
using StaticArrays
using Random: MersenneTwister, randn

using MORFE
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver, SpectralData
using MORFE.ParametrisationSolver: solve_parametrisation
using MORFE.InvarianceError: invariance_error_norms
using MORFE.InvarianceEquation: precompute_sparse_L_template,
                                precompute_sparse_bordered_template,
                                scatter_L_into_bordered!,
                                build_sparse_L_and_rhs!

@testset "Bordered cohomological system" begin

    # ── Fixtures ─────────────────────────────────────────────────────────────
    # Damped 2-DOF Duffing: eigenvalues are off the imaginary axis, so L(s) stays
    # comfortably conditioned. Mirrors test_parametrise_entry.jl.
    function duffing_model(; damping = 0.001, K = [2.0 -1.0; -1.0 2.0])
        B0 = K
        B2 = [1.0 0.0; 0.0 1.0]
        B1 = damping * B2
        term_cubic = MultilinearMap(
            (res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3),
            (3, 0)
        )
        return (B0, B1, B2), (term_cubic,)
    end

    # Solve the same spectral problem through both matrix types. The eigenproblem
    # is solved once and its output fed to both runs, so any difference is due to
    # the linear solver alone and not to eigenvector gauge.
    function solve_both(linear_terms, nl_terms, order, ROM; resonance_style, tol)
        dense_model = NthOrderModel(linear_terms, nl_terms)
        sparse_model = NthOrderModel(map(sparse, linear_terms), nl_terms)
        @test sparse_model.linear_terms[1] isa SparseMatrixCSC
        @test dense_model.linear_terms[1] isa Matrix

        ep = spectrum(dense_model; solver = DefaultEigensolver())
        # Same bundle for both runs: the models differ only in matrix type, and the
        # blocks are sliced from the eigenproblem either way.
        sd = SpectralData(dense_model, ep; master = master_by_sorting(ROM))

        mset = all_multiindices_up_to(ROM, order; min_degree = 1)
        master_eigs = master_eigenvalues(sd)
        rset = MORFE.Resonance.build_resonance_set(dense_model, mset, sd,
            MORFE.Resonance.ResonanceConfig(style = resonance_style, tol = tol,
                outer_targets = true, warn_outer = false))

        args = (mset, sd, rset)
        kwargs = (; options = ParametrisationOptions(show_progress = false))
        Wd, Rd = solve_parametrisation(dense_model, args...; kwargs...)
        Ws, Rs = solve_parametrisation(sparse_model, args...; kwargs...)
        umfpack_options = ParametrisationOptions(
            backend = :umfpack, show_progress = false)
        Wu, Ru = solve_parametrisation(
            sparse_model, args...; options = umfpack_options)
        @test norm(Wu.poly.coefficients - Wd.poly.coefficients) /
              max(norm(Wu.poly.coefficients), norm(Wd.poly.coefficients), eps()) <= 1e-9
        @test norm(Ru.poly.coefficients - Rd.poly.coefficients) /
              max(norm(Ru.poly.coefficients), norm(Rd.poly.coefficients), eps()) <= 1e-9
        return (; Wd, Rd, Ws, Rs, Wu, Ru, mset, rset, master_eigs)
    end

    relerr(a, b) = norm(a .- b) / max(norm(a), norm(b), eps())

    # ── 1. Bordered template structure ───────────────────────────────────────
    @testset "bordered template structure" begin
        FOM, ROM = 6, 3
        Id = SparseMatrixCSC{ComplexF64, Int}(I, FOM, FOM)
        B0 = sprand(ComplexF64, FOM, FOM, 0.3) + Id
        B1 = sprand(ComplexF64, FOM, FOM, 0.2)
        B2 = copy(Id)
        linear_terms = (B0, B1, B2)

        L_tmpl, L_maps = precompute_sparse_L_template(linear_terms)
        M, border_row_base = precompute_sparse_bordered_template(L_tmpl, ROM)
        N = FOM + ROM

        @test size(M) == (N, N)
        @test nnz(M) == nnz(L_tmpl) + 2 * FOM * ROM + ROM^2
        @test length(border_row_base) == FOM
        # CSC invariant: row indices must be sorted within every column.
        for c in 1:N
            rows = M.rowval[M.colptr[c]:(M.colptr[c + 1] - 1)]
            @test issorted(rows)
            @test allunique(rows)
        end
        # Border blocks are structurally dense.
        for c in 1:FOM
            @test border_row_base[c] == M.colptr[c + 1] - ROM
            @test M.rowval[border_row_base[c]:(border_row_base[c] + ROM - 1)] ==
                  collect((FOM + 1):N)
        end
        for q in 1:ROM
            @test M.rowval[M.colptr[FOM + q]:(M.colptr[FOM + q + 1] - 1)] == collect(1:N)
        end

        # The per-column block copy round-trips the (1,1) block for an arbitrary L(s).
        # This is the assertion that pins the affine-per-column layout the copy relies on.
        s = 0.7 - 1.3im
        rhs = zeros(ComplexF64, FOM)
        build_sparse_L_and_rhs!(rhs, L_tmpl, L_maps, linear_terms, s,
            [zeros(ComplexF64, FOM) for _ in 1:2])
        scatter_L_into_bordered!(M, L_tmpl)
        @test Matrix(M)[1:FOM, 1:FOM] ≈ Matrix(L_tmpl)
        # … and leaves the border untouched.
        @test all(iszero, Matrix(M)[(FOM + 1):N, :])
        @test all(iszero, Matrix(M)[:, (FOM + 1):N])

        # The pattern must survive re-evaluation at a different s — this is what
        # makes the cached symbolic factorisation valid for the whole solve.
        colptr_before, rowval_before = copy(M.colptr), copy(M.rowval)
        build_sparse_L_and_rhs!(rhs, L_tmpl, L_maps, linear_terms, -4.2 + 0.1im,
            [zeros(ComplexF64, FOM) for _ in 1:2])
        scatter_L_into_bordered!(M, L_tmpl)
        @test M.colptr == colptr_before
        @test M.rowval == rowval_before
    end

    # ── 2. Sparse ≡ dense, well-conditioned reference ────────────────────────
    @testset "sparse ≡ dense — damped Duffing, complex normal form" begin
        lin, nl = duffing_model()
        r = solve_both(lin, nl, 5, 2;
            resonance_style = :complex_normal_form, tol = 0.05)
        @test relerr(r.Ws.poly.coefficients, r.Wd.poly.coefficients) ≤ 1e-12
        @test relerr(r.Rs.poly.coefficients, r.Rd.poly.coefficients) ≤ 1e-12
    end

    # ── 3. Sparse ≡ dense with L(s) singular at every resonance ──────────────
    # Undamped ⇒ purely imaginary eigenvalues ⇒ for the master pair (λ, λ̄) the
    # monomials α = e_r + (balanced conjugate pair) give s_α = λ_r to machine
    # precision, i.e. det L(s_α) ≈ 0. Graph style makes every degree-≥2 monomial
    # resonant with every master mode, so the border is full width throughout —
    # the maximum-stress case for the old eliminate-L-first path.
    @testset "sparse ≡ dense — conservative, L(s) singular at resonances" begin
        lin, nl = duffing_model(; damping = 0.0)
        r = solve_both(lin, nl, 5, 2; resonance_style = :graph, tol = 0.05)

        # Sanity: the fixture really does put s on the spectrum.
        λ = r.master_eigs
        @test maximum(abs ∘ real, λ) ≤ 1e-12          # undamped
        s_resonant = 2 * λ[1] + λ[2]                   # α = (2,1)
        @test abs(s_resonant - λ[1]) ≤ 1e-12 * abs(λ[1])

        @test all(isfinite, r.Ws.poly.coefficients)
        @test all(isfinite, r.Rs.poly.coefficients)
        @test relerr(r.Ws.poly.coefficients, r.Wd.poly.coefficients) ≤ 1e-10
        @test relerr(r.Rs.poly.coefficients, r.Rd.poly.coefficients) ≤ 1e-10
    end

    # ── 4. 1:3 internal resonance, both pairs as masters ─────────────────────
    @testset "sparse ≡ dense — 1:3 internally resonant, conservative" begin
        # K = diag(1, 9) ⇒ ω = (1, 3), undamped. Both conjugate pairs must be
        # masters: with ROM = 4 the α = (3,0,0,0) monomial has s = 3i, which is a
        # master eigenvalue and therefore carries a border.
        lin, nl = duffing_model(; damping = 0.0, K = [1.0 0.0; 0.0 9.0])
        r = solve_both(lin, nl, 5, 4;
            resonance_style = :complex_normal_form, tol = 0.05)
        @test Set(round.(imag(r.master_eigs); digits = 8)) == Set([-3.0, -1.0, 1.0, 3.0])
        @test all(isfinite, r.Ws.poly.coefficients)
        @test relerr(r.Ws.poly.coefficients, r.Wd.poly.coefficients) ≤ 1e-10
        @test relerr(r.Rs.poly.coefficients, r.Rd.poly.coefficients) ≤ 1e-10
    end

    # ── 5. Outer resonance is reported, not silently absorbed ────────────────
    # Same 1:3 system but with only the ω = 1 pair as master. The monomial
    # α = (3,0) then has s = 3i — exactly a *non-master* eigenvalue. The border
    # spans master directions only, so L(s) is singular in a direction it cannot
    # regularise and the reduction is genuinely ill-posed there. Both paths must
    # say so rather than return a plausible-looking answer.
    @testset "outer resonance raises instead of returning garbage" begin
        lin, nl = duffing_model(; damping = 0.0, K = [1.0 0.0; 0.0 9.0])
        for style in (:complex_normal_form, :graph)
            err = try
                solve_both(lin, nl, 3, 2; resonance_style = style, tol = 0.05)
                nothing
            catch e
                e
            end
            @test err isa MORFE.BorderedLinearSolvers._BorderedSolveFailure
            @test err.category == :outer_resonance
            @test !isempty(err.outer_resonance_targets)
            message = sprint(showerror, err)
            @test occursin("outer resonance", message)
            @test occursin("selected backend: dense", message)
            @test occursin("multiindex:", message)
            @test occursin("superharmonic:", message)
            @test occursin("inner resonance mask:", message)
        end
    end

    # ── 5. Resonance mask ────────────────────────────────────────────────────
    @testset "non-resonant reduced-dynamics coefficients are hard zeros" begin
        lin, nl = duffing_model()
        r = solve_both(lin, nl, 5, 2;
            resonance_style = :complex_normal_form, tol = 0.05)
        ROM = 2
        n_masked = 0
        for idx in 1:length(r.mset), mode in 1:ROM

            MORFE.Resonance.is_resonant(r.rset, idx, mode) && continue
            n_masked += 1
            # Exactly zero, not merely small: the unpack writes these directly
            # rather than reading them out of the trivial rows of the solve.
            @test r.Rs.poly.coefficients[mode, idx] === zero(ComplexF64)
            @test r.Rd.poly.coefficients[mode, idx] === zero(ComplexF64)
        end
        @test n_masked > 0   # the fixture must actually exercise the mask
    end

    # ── 6. FE scale ──────────────────────────────────────────────────────────
    # The fixtures above are 2–4 DOF, where every quantity is O(1) and the
    # conditioning question the bordered formulation exists to answer cannot even
    # arise. This one is a steel bar with consistent mass: K ~ 1e10, M ~ 1e-3, so
    # the operator spans ~1e13 in dynamic range — the regime real FE models live in,
    # and the one where border-vs-L scaling could plausibly cost accuracy.
    function fe_bar_model(n; E = 210e9, ρ = 7850.0, A = 1e-4, Ltot = 1.0, β = 1e12)
        le = Ltot / n
        k = E * A / le
        m = ρ * A * le / 6
        K = spdiagm(-1 => fill(-k, n - 1), 0 => fill(2k, n), 1 => fill(-k, n - 1))
        M = spdiagm(-1 => fill(m, n - 1), 0 => fill(4m, n), 1 => fill(m, n - 1))
        C = 1e-3 .* M
        tc = MultilinearMap((res, x1, x2, x3) -> (@. res += -β * x1 * x2 * x3), (3, 0))
        return (K, C, M), (tc,)
    end

    # One solve serves both assertions below: `solve_both` runs a dense eigensolve
    # plus two full parametrisations, so building the fixture twice was pure waste.
    @testset "FE scale — stiff operator" begin
        lin, nl = fe_bar_model(200)
        # Guard the premise: if this ever stops being FE-scaled the test silently
        # degrades into another O(1) toy and stops checking what it is here to check.
        @test maximum(abs, lin[1]) > 1e9
        @test maximum(abs, lin[1]) / maximum(abs, lin[3]) > 1e10

        r = solve_both(map(Matrix, lin), nl, 5, 2;
            resonance_style = :complex_normal_form, tol = 0.05)

        # Measured 5.3e-10 (W) / 2.3e-10 (R) — orders looser than the 1e-16 seen at
        # FOM = 2, which is the conditioning of the stiff operator showing through,
        # not a defect: it sits at the κ·u floor for this dynamic range.
        @test relerr(r.Ws.poly.coefficients, r.Wd.poly.coefficients) ≤ 1e-8
        @test relerr(r.Rs.poly.coefficients, r.Rd.poly.coefficients) ≤ 1e-8

        # A-posteriori residual. `sparse ≡ dense` cannot scale past the point where the
        # dense path is runnable; this check can, because it evaluates the defining
        # equation ∂W/∂z·R(z) = F(W(z)) on the manifold itself and needs no reference.
        # Normalised by a representative term so the number is dimensionless.
        model = NthOrderModel(lin, nl)
        rng = MersenneTwister(20260731)
        amplitude = 1e-3
        rms = invariance_error_norms(
            model, r.Ws, r.Rs; n_samples = 200, amplitude, rng).rms

        # Scale of a representative term, ‖B₀·W₁(z)‖, at the same amplitude.
        scale = 0.0
        for _ in 1:200
            z = ComplexF64[complex(amplitude / sqrt(2) * randn(rng),
                               amplitude / sqrt(2) * randn(rng)) for _ in 1:2]
            X = MORFE.Polynomials.evaluate(r.Ws.poly, z)
            scale = max(scale, norm(lin[1] * view(X, :, 1)))
        end

        # Measured ≈ 5.2e-8 here and 3.5e-8 on the 4977-DOF Gridap beam, flat across
        # amplitude 1e-4…1e-3 — i.e. the arithmetic floor, not truncation error, which
        # is what "solved as well as the arithmetic allows" looks like.
        @test rms / scale ≤ 1e-6
    end

    # ── 7. Solver state must be finalisable ──────────────────────────────────
    # The Pardiso branch attaches a finaliser to release C-side memory, and Julia
    # refuses to finalise an immutable object. Without Pardiso installed the
    # `ps === nothing ||` guard short-circuits, so no test reaches that line by
    # running a solve — the property has to be asserted directly.
    @testset "solver state is finalisable (Pardiso teardown path)" begin
        n, ROM = 20, 2
        K = spdiagm(-1 => -ones(n - 1), 0 => 2.0 * ones(n), 1 => -ones(n - 1))
        Mass = spdiagm(0 => ones(n))
        L_tmpl, L_maps = precompute_sparse_L_template(
            (complex(K), complex(0.01 .* Mass), complex(Mass)))
        ss = MORFE.BorderedLinearSolvers.SparseLinearSolverState{ComplexF64}(
            L_tmpl, L_maps, n, ROM)
        @test ismutable(ss)
        # The call the Pardiso branch makes; on an immutable state this throws
        # "cannot be finalized because they are not mutable".
        @test (finalizer(_ -> nothing, ss); true)
    end

    # ── 7. The factorisation reads the template in place ─────────────────────
    # Assembly writes into `ss.bordered.nzval` and never copies those values into the
    # factorisation, which is only correct because the two share one array — `klu`
    # takes `nzval` by reference. Breaking that surfaces in the sparse ≡ dense
    # testsets above, but only as an opaque numerical mismatch; this pins the
    # invariant directly, and also catches a rebind that silently costs a copy back
    # without changing results.
    @testset "factorisation aliases the bordered template" begin
        n, ROM = 40, 2
        K = spdiagm(-1 => -ones(n - 1), 0 => 2.0 * ones(n), 1 => -ones(n - 1))
        Mass = spdiagm(0 => ones(n))
        lt = (complex(K), complex(0.01 .* Mass), complex(Mass))
        L_tmpl, L_maps = precompute_sparse_L_template(lt)
        ss = MORFE.BorderedLinearSolvers.SparseLinearSolverState{ComplexF64}(
            L_tmpl, L_maps, n, ROM)

        # Fill the template the way _solve_monomial! does: L(s) in the (1,1) block and
        # a τ = 1 border (all modes non-resonant), which is nonsingular whenever L is.
        function fill_template!(s)
            scratch = zeros(ComplexF64, n)
            build_sparse_L_and_rhs!(scratch, ss.L_template, ss.L_mappings, lt, s,
                [zeros(ComplexF64, n) for _ in 1:2])
            scatter_L_into_bordered!(ss.bordered, ss.L_template)
            for r in 1:ROM
                ss.bordered[n + r, n + r] = one(ComplexF64)
            end
            return ss.bordered
        end

        fill_template!(0.3 + 1.1im)
        F = MORFE.BorderedLinearSolvers._refactorise!(ss, ss.bordered)
        @test issuccess(F)
        @test F.nzval === ss.bordered.nzval
        @test @inferred(MORFE.BorderedLinearSolvers._cached_klu_factor(
            ss, ss.bordered)) === F

        # Behavioural form of the same invariant, independent of *why* it might break:
        # refilling the template at a new s must change what a refactorisation sees.
        b = ones(ComplexF64, n + ROM)
        A_first = copy(ss.bordered)
        x_first = similar(b)
        ldiv!(x_first, F, copy(b))

        fill_template!(-2.7 + 0.4im)
        F2 = MORFE.BorderedLinearSolvers._refactorise!(ss, ss.bordered)
        x_second = similar(b)
        ldiv!(x_second, F2, copy(b))

        @test !(x_first ≈ x_second)                                    # it saw the change
        @test norm(ss.bordered * x_second - b) / norm(b) ≤ 1e-8        # solved the new matrix
        @test norm(A_first * x_first - b) / norm(b) ≤ 1e-8             # and the old one
    end

    @testset "failed SuiteSparse caches are discarded" begin
        BLS = MORFE.BorderedLinearSolvers
        mset = MultiindexSet([SVector(2)])
        unflagged = MORFE.Resonance.empty_resonance_set(mset, 1)
        flagged = MORFE.Resonance.empty_resonance_set(mset, 1, 1)
        MORFE.Resonance.set_resonance!(flagged, 2, 1, true)
        resonance = SVector(false)

        function make_state(backend)
            template = spdiagm(0 => ones(ComplexF64, 2))
            mappings = [collect(eachindex(template.nzval))]
            options = ParametrisationOptions(
                backend = backend, residual_check = :off, show_progress = false)
            return BLS.SparseLinearSolverState{ComplexF64}(
                template, mappings, 2, 1; options)
        end
        function set_diagonal!(state, values)
            fill!(state.bordered.nzval, 0)
            for index in eachindex(values)
                state.bordered[index, index] = values[index]
            end
            return state.bordered
        end

        @test BLS._backend_name(make_state(:auto).backend) == :klu

        for backend in (:klu, :umfpack)
            state = make_state(backend)
            set_diagonal!(state, ComplexF64[2, 3, 1])
            factor = backend == :klu ? BLS._refactorise_klu!(state, state.bordered) :
                     BLS._refactorise_umfpack!(state, state.bordered)
            @test issuccess(factor)
            @test state.fact === factor
            if backend == :umfpack
                @test @inferred(BLS._cached_umfpack_factor(
                    state, state.bordered)) === factor
            end

            set_diagonal!(state, ComplexF64[0, 3, 1])
            failure = try
                BLS._bordered_solve!(state, ones(ComplexF64, 3), 0.5 + 0.2im,
                    1, mset[1], resonance, unflagged)
                nothing
            catch error
                error
            end
            @test failure isa BLS._BorderedSolveFailure
            @test failure.category == :factorisation
            @test failure.backend == backend
            @test failure.index == 1
            @test failure.multiindex == mset[1]
            @test failure.inner_resonance_mask == resonance
            @test failure.recovery_attempted
            @test isempty(failure.outer_resonance_targets)
            @test state.fact === nothing
            message = sprint(showerror, failure)
            @test occursin("not by itself proof", message)
            @test occursin("fresh retry", message)
            if backend == :klu
                @test factor._symbolic == C_NULL
                @test factor._numeric == C_NULL
            end

            set_diagonal!(state, ComplexF64[4, 5, 1])
            recovered = backend == :klu ? BLS._refactorise_klu!(state, state.bordered) :
                        BLS._refactorise_umfpack!(state, state.bordered)
            @test issuccess(recovered)
            @test state.fact === recovered
        end

        outer_state = make_state(:umfpack)
        set_diagonal!(outer_state, ComplexF64[0, 3, 1])
        outer_failure = try
            BLS._bordered_solve!(outer_state, ones(ComplexF64, 3), 0.5 + 0.2im,
                1, mset[1], resonance, flagged)
            nothing
        catch error
            error
        end
        @test outer_failure isa BLS._BorderedSolveFailure
        @test outer_failure.category == :outer_resonance
        @test outer_failure.outer_resonance_targets == [1]
        @test occursin("likely cause", sprint(showerror, outer_failure))
    end
end
