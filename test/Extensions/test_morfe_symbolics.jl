ext = Base.get_extension(MORFE, :MORFESymbolicsExt)
@testset "extraction" begin
    @testset "degree_of_monomial" begin
        @variables x y

        # constant → degree 0
        @test ext.MORFESymbolicsExt.degree_of_monomial(Num(3)) == 0

        # linear terms
        @test ext.MORFESymbolicsExt.degree_of_monomial(x) == 1
        @test ext.MORFESymbolicsExt.degree_of_monomial(2 * x) == 1

        # quadratic
        @test ext.MORFESymbolicsExt.degree_of_monomial(x^2) == 2
        @test ext.MORFESymbolicsExt.degree_of_monomial(x * y) == 2
        @test ext.MORFESymbolicsExt.degree_of_monomial(3 * x * y) == 2

        # cubic
        @test ext.MORFESymbolicsExt.degree_of_monomial(x^3) == 3
        @test ext.MORFESymbolicsExt.degree_of_monomial(x^2 * y) == 3
    end

    @testset "multidegree_of_monomial" begin
        @variables z1 z2 dz1 dz2
        groups = ([z1, z2], [dz1, dz2])

        # purely in group 1
        @test ext.MORFESymbolicsExt.multidegree_of_monomial(z1^2, groups) == (2, 0)
        @test ext.MORFESymbolicsExt.multidegree_of_monomial(z1 * z2, groups) == (2, 0)

        # purely in group 2
        @test ext.MORFESymbolicsExt.multidegree_of_monomial(dz1, groups) == (0, 1)

        # mixed
        @test ext.MORFESymbolicsExt.multidegree_of_monomial(z1 * dz1, groups) == (1, 1)
        @test ext.MORFESymbolicsExt.multidegree_of_monomial(z1^2 * dz2, groups) == (2, 1)

        # coefficients do not affect degree
        @test ext.MORFESymbolicsExt.multidegree_of_monomial(5 * z2^2, groups) == (2, 0)
    end

    @testset "_findgroup — error on unknown symbol" begin
        @variables z1 z2 w  # w is not in any group
        groups = ([z1, z2],)
        @test_throws ErrorException ext.MORFESymbolicsExt._findgroup(Symbolics.value(w), groups)
    end

    @testset "seperate_into_monomials" begin
        @variables x y

        # single monomial
        result = ext.MORFESymbolicsExt.seperate_into_monomials(x^2)
        @test isequal(result, [x^2])
        @test length(result) == 1

        # sum of two monomials
        result = ext.MORFESymbolicsExt.seperate_into_monomials(x + y)
        @test isequal(result, [x, y])
        @test length(result) == 2

        # sum of three monomials
        result = ext.MORFESymbolicsExt.seperate_into_monomials(x^2 + x * y + 3 * y^2)
        @test length(result) == 3
    end

    @testset "extract_linear_matrices — 2-DOF linear system" begin
        @variables z1 z2 dz1 dz2
        groups = ([z1, z2], [dz1, dz2])

        # ż = A z  →  expressed as  -dz + A z = 0
        A = [-1.0 0.5; 0.2 -3.0]
        exprs = [
            -dz1 + A[1, 1] * z1 + A[1, 2] * z2,
            -dz2 + A[2, 1] * z1 + A[2, 2] * z2
        ]

        B = ext.MORFESymbolicsExt.extract_linear_matrices(exprs, groups)

        # B[1] should recover A; B[2] should be identity (coefficient of dz)
        @test B[1] ≈ -A
        @test B[2] ≈ [1.0 0.0; 0.0 1.0]
    end

    @testset "extract_linear_matrices — sign convention" begin
        # exprs = -dz - z  →  B[1] = I, B[2] = I  (after the *-1 inside the function)
        @variables z1 dz1
        groups = ([z1,], [dz1,])
        exprs = [-dz1 - z1]
        B = ext.MORFESymbolicsExt.extract_linear_matrices(exprs, groups)
        @test B[1] ≈ reshape([1.0], 1, 1)
        @test B[2] ≈ reshape([1.0], 1, 1)
    end

    @testset "extract_nonlinear_monomials — quadratic remainder" begin
        @variables z1 z2 dz1 dz2
        groups = ([z1, z2], [dz1, dz2])

        # purely quadratic RHS: F = [z1^2, z1*z2]
        exprs = [-dz1 - z1 + z1^2,
            -dz2 - z2 + z1 * z2]

        linear_terms = ext.MORFESymbolicsExt.extract_linear_matrices(exprs, groups)
        N, monomials,
        deg_monomials,
        multideg_monomials = ext.MORFESymbolicsExt.extract_nonlinear_monomials(exprs, groups, linear_terms)

        @test N == 2
        # Each row should have exactly one nonlinear monomial
        @test length(monomials[1]) == 1
        @test isequal(Num(monomials[1][1]), z1^2)
        @test isequal(Num(monomials[2][1]), z1 * z2)
        @test length(monomials[2]) == 1
        @test deg_monomials[1][1] == 2
        @test deg_monomials[2][1] == 2
        @test multideg_monomials[1][1] == (2,)
        @test multideg_monomials[2][1] == (2,)
    end

    @testset "group_monomials — collects by multiindex" begin
        @variables z1 z2 dz1 dz2
        groups = ([z1, z2], [dz1, dz2])

        exprs = [-dz1 - z1 + z1^2 + z1 * z2,
            -dz2 - z2 + z1^2]

        linear_terms = ext.MORFESymbolicsExt.extract_linear_matrices(exprs, groups)
        N, monomials,
        deg_monomials,
        multideg_monomials = ext.MORFESymbolicsExt.extract_nonlinear_monomials(exprs, groups, linear_terms)

        F_by_multiindex = ext.MORFESymbolicsExt.group_monomials(monomials, multideg_monomials, N)

        # Both quadratic monomials share multiindex (2,) in the F_groups = groups[1:end-1]
        @test haskey(F_by_multiindex, (2,))
        # The grouped vector should be length N
        @test length(F_by_multiindex[(2,)]) == N
        @test isequal(F_by_multiindex[(2,)], [z1^2 + z1 * z2, z1^2])
    end
end #@testset "extraction"

@testset "polarization" begin
    @testset "polarize_monomial — z1^2" begin
        @variables z1 z2
        groups = ([z1, z2],)
        mi = (2,)

        pol, slotvars = ext.MORFESymbolicsExt.polarize_monomial(z1^2, groups, mi)

        # result should be z1_1 * z1_2 (the polarization of z1^2)
        @test length(slotvars) == 1       # one group
        @test length(slotvars[1]) == 2    # two slots (degree 2)
        @test length(slotvars[1][1]) == 2 # two variables per slot
    end

    @testset "polarize_monomial — z1*z2 is already symmetric" begin
        @variables z1 z2
        groups = ([z1, z2],)
        mi = (2,)

        pol, slotvars = ext.MORFESymbolicsExt.polarize_monomial(z1 * z2, groups, mi)

        @test length(slotvars[1]) == 2
    end

    @testset "polarize_monomial — coefficient preserved" begin
        @variables z1 z2
        groups = ([z1, z2],)
        mi = (2,)

        pol_coeff, _ = ext.MORFESymbolicsExt.polarize_monomial(3 * z1^2, groups, mi)
        pol_no_coeff, _ = ext.MORFESymbolicsExt.polarize_monomial(z1^2, groups, mi)

        # Substituting all slot vars to 1: 3*z1_1*z1_2 → 3,  z1_1*z1_2 → 1
        # (just check that the polarized expression with coefficient is 3× the one without)
        sub = Dict(v => Num(1)
        for g in ext.MORFESymbolicsExt.polarize_monomial(z1^2, groups, mi)[2]
        for sv in g for v in sv)
        @test isequal(Symbolics.substitute(pol_coeff, sub),
            3 * Symbolics.substitute(pol_no_coeff, sub))
    end

    @testset "polarize — dictionary round-trip" begin
        @variables z1 z2 dz1 dz2
        groups = ([z1, z2], [dz1, dz2])
        F_groups = groups[1:(end - 1)]  # = ([z1, z2],)

        exprs = [-dz1 - z1 + z1^2,
            -dz2 - z2 + z1 * z2]

        linear_terms = ext.MORFESymbolicsExt.extract_linear_matrices(exprs, groups)
        N, monomials,
        deg_monomials,
        multideg_monomials = ext.MORFESymbolicsExt.extract_nonlinear_monomials(exprs, groups, linear_terms)
        F_by_multiindex = ext.MORFESymbolicsExt.group_monomials(monomials, multideg_monomials, N)

        F_pol, dict_slotvars = ext.MORFESymbolicsExt.polarize(F_by_multiindex, F_groups, N)

        # Same keys
        @test Set(keys(F_pol)) == Set(keys(F_by_multiindex))
        # Polarized vectors have the right length
        for (key, vec) in F_pol
            @test length(vec) == N
        end
    end
end #@testset "polarization"

@testset "toMultilinearMaps" begin
    #Helper function 
    function eval_term(term, xs, r = nothing)
        N = length(xs[1])
        res = zeros(N)
        MORFE.MultilinearMaps.evaluate_term!(res, term, xs, r)
        return res
    end

    @testset "scalar quadratic z1^2" begin
        @variables z1 dz1
        groups = ([z1], [dz1])
        exprs = [-dz1 - z1 + z1^2]
        model = symbolics_to_NDOrdermodel(exprs, groups)

        @test length(model.nonlinear_terms) == 1
        term = model.nonlinear_terms[1]

        # metadata
        @test term.multiindex == (2,)      # degree 2 in x
        @test term.multiplicity_external == 0
        @test term.deg == 2

        # evaluate at x = [3.0]: should give 3^2 = 9
        x = [3.0]
        res = eval_term(term, (x,))
        @test res ≈ [9.0]

        # scaling: f(α·x) = α^2 · f(x)
        α = 2.5
        res_scaled = eval_term(term, (α .* x,))
        @test res_scaled ≈ α^2 .* res

        # linearity in first slot (with second fixed):
        # f(x+y, x+y) = f(x,x) + f(x,y) + f(y,x) + f(y,y)
        # but evaluate_term! passes xs[1] to both slots, so
        # instead verify: result at [1.0] = 1, at [2.0] = 4
        @test eval_term(term, ([1.0],)) ≈ [1.0]
        @test eval_term(term, ([2.0],)) ≈ [4.0]
    end

    @testset "cross term z1*z2" begin
        @variables z1 z2 dz1 dz2
        groups = ([z1, z2], [dz1, dz2])
        exprs = [-dz1 - z1,
            -dz2 - z2 + z1 * z2]
        model = symbolics_to_NDOrdermodel(exprs, groups)

        @test length(model.nonlinear_terms) == 1
        term = model.nonlinear_terms[1]
        @test term.multiindex == (2,)
        @test term.deg == 2

        # at x = [a, b]: row 1 = 0, row 2 = a*b
        a, b = 3.0, 2.0
        x = [a, b]
        res = eval_term(term, (x,))
        @test res[1] ≈ 0.0
        @test res[2] ≈ a * b

        # scaling
        α = 1.5
        res2 = eval_term(term, (α .* x,))
        @test res2 ≈ α^2 .* res
    end
    @testset "grouped quadratic terms — single MultilinearMap" begin
        @variables z1 z2 dz1 dz2
        groups = ([z1, z2], [dz1, dz2])
        exprs = [-dz1 - z1 + z1^2 + z1 * z2,
            -dz2 - z2 + z1^2 + z1 * z2]
        model = symbolics_to_NDOrdermodel(exprs, groups)

        # One multiindex (2,) → one term
        @test length(model.nonlinear_terms) == 1
        term = model.nonlinear_terms[1]

        a, b = 2.0, 3.0
        x = [a, b]
        res = eval_term(term, (x,))

        expected = a^2 + a * b
        @test res[1] ≈ expected
        @test res[2] ≈ expected
    end

    @testset "two multiindices — two MultilinearMaps (ORD=3)" begin
        @variables z1 dz1 ddz1
        groups = ([z1], [dz1], [ddz1])
        exprs = [-ddz1 - dz1 - z1 + z1^2 + dz1^2]
        model = symbolics_to_NDOrdermodel(exprs, groups)

        @test length(model.nonlinear_terms) == 2

        # Find the term with multiindex (2,0) — z1^2
        term_z = findfirst(t -> t.multiindex == (2, 0), model.nonlinear_terms)
        term_dz = findfirst(t -> t.multiindex == (0, 2), model.nonlinear_terms)
        @test !isnothing(term_z)
        @test !isnothing(term_dz)

        a, b = 3.0, 2.0
        x = [a]   # z1  slot
        dx = [b]   # dz1 slot

        # term for z1^2: xs = (x, dx), uses only x → result = a^2
        res_z = eval_term(model.nonlinear_terms[term_z], (x, dx))
        @test res_z ≈ [a^2]

        # term for dz1^2: uses only dx → result = b^2
        res_dz = eval_term(model.nonlinear_terms[term_dz], (x, dx))
        @test res_dz ≈ [b^2]

        # total nonlinear contribution sums correctly
        total = res_z + res_dz
        @test total ≈ [a^2 + b^2]
    end
    @testset "cubic term z1^3" begin
        @variables z1 dz1
        groups = ([z1], [dz1])
        exprs = [-dz1 - z1 - z1^3]
        model = symbolics_to_NDOrdermodel(exprs, groups)

        @test length(model.nonlinear_terms) == 1
        term = model.nonlinear_terms[1]
        @test term.multiindex == (3,)
        @test term.deg == 3

        a = 2.0
        res = eval_term(term, ([a],))
        @test res ≈ [-a^3]

        # scaling: f(α·x) = α^3 · f(x)
        α = 1.5
        @test eval_term(term, ([α * a],)) ≈ [-α^3 * a^3]
    end
    @testset "evaluate_term! accumulates into res" begin
        @variables z1 dz1
        groups = ([z1], [dz1])
        exprs = [-dz1 - z1 + z1^2]
        model = symbolics_to_NDOrdermodel(exprs, groups)
        term = model.nonlinear_terms[1]

        x = [3.0]
        res = [10.0]   # non-zero initial value
        MORFE.MultilinearMaps.evaluate_term!(res, term, (x,), nothing)
        @test res ≈ [10.0 + 9.0]   # 10 + 3^2
    end

    @testset "external forcing — linear in r" begin
        @variables z1 dz1 r1
        groups = ([z1], [dz1])
        ext_var = [r1]
        exprs = [-dz1 - z1 + 5 * r1]
        ext_exprs = [-r1]

        model = symbolics_to_NDOrdermodel(exprs, groups, ext_var, ext_exprs)

        @test length(model.nonlinear_terms) == 1
        term = model.nonlinear_terms[1]
        @test term.multiplicity_external == 1
        @test sum(term.multiindex) == 0   # no state dependence

        x = [0.0]
        r = [2.0]
        res = eval_term(term, (x,), r)
        @test res ≈ [10.0]   # 5 * r[1] = 5 * 2

        # scales linearly with r
        r2 = [3.0]
        res2 = eval_term(term, (x,), r2)
        @test res2 ≈ [15.0]
    end

    @testset "mixed term z1*r1" begin
        @variables z1 dz1 r1
        groups = ([z1], [dz1])
        ext_var = [r1]
        exprs = [-dz1 - z1 + z1 * r1]
        ext_exprs = [-r1]

        model = symbolics_to_NDOrdermodel(exprs, groups, ext_var, ext_exprs)

        term = only(model.nonlinear_terms)
        @test sum(term.multiindex) == 1
        @test term.multiplicity_external == 1
        @test term.deg == 2

        a, b = 3.0, 4.0
        res = eval_term(term, ([a],), [b])
        @test res ≈ [a * b]

        # bilinear: scales as α*β when x→α·x, r→β·r
        α, β = 2.0, 3.0
        res2 = eval_term(term, ([α * a],), [β * b])
        @test res2 ≈ [α * β * a * b]
    end
end #@testset "toMultilinearMaps"

@testset "symbolics_to_NDOrdermodel (no external)" begin
    # @testset "purely linear system returns zero nonlinear terms" begin
    #     @variables z1 z2 dz1 dz2
    #     groups = ([z1, z2], [dz1, dz2])

    #     exprs = [-dz1 - z1,
    #         -dz2 - z2]

    #     model = symbolics_to_NDOrdermodel(exprs, groups)

    #     # Linear matrices: B[1] and B[2] should be identity
    #     @test model.linear_terms[1] ≈ [1.0 0.0; 0.0 1.0]
    #     @test model.linear_terms[2] ≈ [1.0 0.0; 0.0 1.0]

    #     # No nonlinear terms
    #     @test length(model.nonlinear_terms) == 0
    # end

    @testset "scalar Duffing oscillator" begin
        # ẍ + x + ε x³ = 0  written as first-order:
        #   ẋ = v,  v̇ = -x - x³
        # In NDOrderModel form (ORD=2):
        #   exprs = [-dz1 - z1 - z1^3,   (= -v - x - x³, the "v equation" row)
        #            -dz2 - z2]            doesn't matter, padding
        # Actually simpler: treat as single scalar 2nd order:
        @variables x dx
        groups = ([x], [dx])

        # -ẍ - x - x³ = 0  i.e.  ẍ + x + x³ = 0
        exprs = [-dx - x - x^3]

        model = symbolics_to_NDOrdermodel(exprs, groups)

        # Linear: B[1] = [1], B[2] = [1]
        @test model.linear_terms[1] ≈ reshape([1.0], 1, 1)
        @test model.linear_terms[2] ≈ reshape([1.0], 1, 1)

        # One nonlinear term (cubic)
        @test length(model.nonlinear_terms) == 1
        nlterm = model.nonlinear_terms[1]
        @test sum(nlterm.multiindex) == 3   # total degree 3
    end

    @testset "2-DOF system from problem statement" begin
        @variables z1 z2 dz1 dz2
        groups = ([z1, z2], [dz1, dz2])

        exprs = [
            -dz1 - z1 - 1.5 * z1 - 2 * z1^2 + 3 * z1 * z2,
            -1 * dz2 - 3.5 * z2 + z1^2 - 5 * z1 * z2 + 21 // 4 * z2^2
        ]

        model = symbolics_to_NDOrdermodel(exprs, groups)

        # B[2] should be identity (coefficient of dẑ)
        @test model.linear_terms[2] ≈ [1.0 0.0; 0.0 1.0]

        # B[1] encodes the linear stiffness: -dz1 - (1+1.5) z1  → B[1][1,1] = 2.5
        @test model.linear_terms[1][1, 1] ≈ 2.5
        @test model.linear_terms[1][2, 2] ≈ 3.5

        # Three distinct quadratic multiindices: (2,0), (1,1), (0,2)
        @test length(model.nonlinear_terms) == 1
    end

    @testset "vector-syntax gives identical model" begin
        # Flat syntax
        @variables z1 z2 dz1 dz2
        groups_flat = ([z1, z2], [dz1, dz2])
        exprs_flat = [
            -dz1 - z1 - 2 * z1^2 + z1 * z2,
            -dz2 - z2 + z1^2
        ]
        model_flat = symbolics_to_NDOrdermodel(exprs_flat, groups_flat)

        # Array syntax
        @variables z[1:2] dz[1:2]
        groups_arr = (collect(z), collect(dz))
        exprs_arr = [
            -dz[1] - z[1] - 2 * z[1]^2 + z[1] * z[2],
            -dz[2] - z[2] + z[1]^2
        ]
        model_arr = symbolics_to_NDOrdermodel(exprs_arr, groups_arr)

        # Both should produce the same linear matrices
        @test model_flat.linear_terms[1] ≈ model_arr.linear_terms[1]
        @test model_flat.linear_terms[2] ≈ model_arr.linear_terms[2]

        # And the same number of nonlinear terms
        @test length(model_flat.nonlinear_terms) == length(model_arr.nonlinear_terms)
    end

    @testset "third-order ODE (ORD=3)" begin
        @variables x dx ddx
        groups = ([x], [dx], [ddx])

        # -x''' - x'' - x' - x - x^2 = 0
        exprs = [-ddx - dx - x - x^2]

        model = symbolics_to_NDOrdermodel(exprs, groups)

        @test model.linear_terms[1] ≈ reshape([1.0], 1, 1)
        @test model.linear_terms[2] ≈ reshape([1.0], 1, 1)
        @test model.linear_terms[3] ≈ reshape([1.0], 1, 1)
        @test length(model.nonlinear_terms) == 1
        @test sum(model.nonlinear_terms[1].multiindex) == 2
    end

    @testset "assert: groups of unequal length throws" begin
        @variables z1 z2 z3 dz1 dz2
        # groups[2] has one fewer variable than groups[1]
        @test_throws AssertionError symbolics_to_NDOrdermodel(
            [-dz1 - z1, -dz2 - z2],
            ([z1, z2, z3], [dz1, dz2])
        )
    end

    @testset "assert: only one group throws (ORDP1 must be > 1)" begin
        @variables z1 dz1
        @test_throws AssertionError symbolics_to_NDOrdermodel(
            [-z1],
            ([z1],)
        )
    end
end #@testset "symbolics_to_NDOrdermodel (no external)"

@testset "symbolics_to_NDOrdermodel (with external)" begin
    @testset "linear forcing — 2-DOF" begin
        @variables z1 z2 dz1 dz2 r1 r2
        groups = ([z1, z2], [dz1, dz2])
        ext_var = [r1, r2]

        # Purely linear in z and r (no nonlinear cross terms)
        exprs = [-dz1 - z1 + 5 * r1,
            -dz2 - z2 + 2 * r2]

        # External system: ṙ = -r  (harmonic oscillator at amplitude level)
        ext_exprs = [-r1, -r2]

        model = symbolics_to_NDOrdermodel(exprs, groups, ext_var, ext_exprs)

        # Linear state matrices unchanged
        @test model.linear_terms[1] ≈ [1.0 0.0; 0.0 1.0]
        @test model.linear_terms[2] ≈ [1.0 0.0; 0.0 1.0]

        # One nonlinear term: degree (0, 1) in (z, r) — the linear-in-r forcing
        @test length(model.nonlinear_terms) == 1
        @test model.nonlinear_terms[1].multiplicity_external == 1
        @test sum(model.nonlinear_terms[1].multiindex) == 0
    end

    @testset "2-DOF problem statement with external" begin
        @variables z1 z2 dz1 dz2 r1 r2
        groups = ([z1, z2], [dz1, dz2])
        ext_var = [r1, r2]

        exprs = [
            -dz1 - z1 - 1.5 * z1 - 2 * z1^2 + 3 * z1 * z2 * r1 + 5 * r1,
            -1 * dz2 - 3.5 * z2 + z1^2 - 5 * z1 * z2 + 21 // 4 * z2^2 + 2 * r2
        ]
        ext_exprs = [-r1, -r2]

        model = symbolics_to_NDOrdermodel(exprs, groups, ext_var, ext_exprs)

        @test model.linear_terms[2] ≈ [1.0 0.0; 0.0 1.0]
        @test model.linear_terms[1][1, 1] ≈ 2.5
        @test model.linear_terms[1][2, 2] ≈ 3.5

        # Should have nonlinear terms (quadratic in z, linear in r, mixed z*z*r)
        @test length(model.nonlinear_terms) > 0

        # ext_degree of all terms should be 0 or 1 (no r^2 in the problem)
        for term in model.nonlinear_terms
            @test term.multiplicity_external <= 1
        end
    end

    @testset "vector syntax with external" begin
        @variables z[1:2] dz[1:2] r[1:2]
        groups = (collect(z), collect(dz))
        ext_var = collect(r)

        exprs = [
            -dz[1] - z[1] - 1.5 * z[1] - 2 * z[1]^2 + 3 * z[1] * z[2] * r[1] + 5 * r[1],
            -dz[2] - 3.5 * z[2] + z[1]^2 - 5 * z[1] * z[2] + (21 // 4) * z[2]^2 + 2 * r[2]
        ]
        ext_exprs = [-r[1], -r[2]]

        # Should not throw
        model = symbolics_to_NDOrdermodel(exprs, groups, ext_var, ext_exprs)
        @test !isnothing(model)
    end

    @testset "assert: empty ext_var throws" begin
        @variables z1 dz1
        @test_throws AssertionError symbolics_to_NDOrdermodel(
            [-dz1 - z1],
            ([z1], [dz1]),
            Num[],
            Num[]
        )
    end
end #@testset "symbolics_to_NDOrdermodel (with external)"

@testset "symbolics_to_Externalsystem" begin
    @testset "scalar linear system" begin
        @variables r
        ext_exprs = [-2 * r]

        ex_sys = ext.MORFESymbolicsExt.symbolics_to_Externalsystem(ext_exprs, [r])
        @test !isnothing(ex_sys)
    end

    @testset "2D linear system" begin
        @variables r1 r2
        ext_exprs = [-r1, -r2]

        ex_sys = ext.MORFESymbolicsExt.symbolics_to_Externalsystem(ext_exprs, [r1, r2])
        @test !isnothing(ex_sys)
    end

    @testset "nonlinear external system" begin
        @variables r1 r2
        ext_exprs = [-r1 + r1^2, -r2 + r1 * r2]

        ex_sys = ext.MORFESymbolicsExt.symbolics_to_Externalsystem(ext_exprs, [r1, r2])
        @test !isnothing(ex_sys)
    end
end  # testset "symbolics_to_Externalsystem"

@testset "input validation" begin
    @testset "ext.MORFESymbolicsExt.is_polynomial — positive cases" begin
        @variables z1 z2 dz1 dz2

        # purely linear
        @test ext.MORFESymbolicsExt.is_polynomial([-dz1 - z1, -dz2 - z2], [
            z1, z2, dz1, dz2])

        # quadratic
        @test ext.MORFESymbolicsExt.is_polynomial([-dz1 - z1 + z1^2, -dz2 - z2 + z1*z2], [
            z1, z2, dz1, dz2])

        # cubic
        @test ext.MORFESymbolicsExt.is_polynomial([-dz1 - z1 - z1^3], [z1, dz1])

        # mixed degree (linear + quadratic + cubic)
        @test ext.MORFESymbolicsExt.is_polynomial([-dz1 - z1 + z1^2 + z1^3], [z1, dz1])

        # with external variables
        @variables r1 r2
        @test ext.MORFESymbolicsExt.is_polynomial(
            [-dz1 - z1 + z1^2 + 3*z1*r1 + 5*r1],
            [z1, dz1, r1]
        )
    end

    @testset "ext.MORFESymbolicsExt.is_polynomial — negative cases" begin
        @variables z1 dz1

        # transcendental functions
        @test !ext.MORFESymbolicsExt.is_polynomial([-dz1 - sin(z1)], [z1, dz1])
        @test !ext.MORFESymbolicsExt.is_polynomial([-dz1 - exp(z1)], [z1, dz1])
        @test !ext.MORFESymbolicsExt.is_polynomial([-dz1 - cos(z1)], [z1, dz1])
        @test !ext.MORFESymbolicsExt.is_polynomial([-dz1 - log(z1)], [z1, dz1])

        # rational function (not polynomial)
        @test !ext.MORFESymbolicsExt.is_polynomial([-dz1 - 1/z1], [z1, dz1])

        # square root
        @test !ext.MORFESymbolicsExt.is_polynomial([-dz1 - sqrt(z1)], [z1, dz1])

        # mixed: one polynomial row, one not
        @variables z2 dz2
        @test !ext.MORFESymbolicsExt.is_polynomial(
            [-dz1 - z1^2, -dz2 - sin(z2)],
            [z1, z2, dz1, dz2]
        )
    end

    @testset "ext.MORFESymbolicsExt.check_constant_terms — no constant terms" begin
        @variables z1 z2 dz1 dz2

        # all rows vanish at origin
        @test isempty(ext.MORFESymbolicsExt.check_constant_terms(
            [-dz1 - z1 + z1^2, -dz2 - z2 + z1*z2],
            [z1, z2, dz1, dz2]
        ))

        # purely linear
        @test isempty(ext.MORFESymbolicsExt.check_constant_terms(
            [-dz1 - z1, -dz2 - z2],
            [z1, z2, dz1, dz2]
        ))
    end

    @testset "ext.MORFESymbolicsExt.check_constant_terms — detects offending rows" begin
        @variables z1 z2 dz1 dz2

        # constant in row 1 only
        offending = ext.MORFESymbolicsExt.check_constant_terms(
            [-dz1 - z1 + 3, -dz2 - z2],
            [z1, z2, dz1, dz2]
        )
        @test offending == [1]

        # constant in row 2 only
        offending = ext.MORFESymbolicsExt.check_constant_terms(
            [-dz1 - z1, -dz2 - z2 + 7],
            [z1, z2, dz1, dz2]
        )
        @test offending == [2]

        # constant in both rows
        offending = ext.MORFESymbolicsExt.check_constant_terms(
            [-dz1 - z1 + 1, -dz2 - z2 - 5],
            [z1, z2, dz1, dz2]
        )
        @test offending == [1, 2]

        # constant cancels to zero — should not be flagged
        offending = ext.MORFESymbolicsExt.check_constant_terms(
            [-dz1 - z1 + 3 - 3],
            [z1, dz1]
        )
        @test isempty(offending)
    end

    @testset "ext.MORFESymbolicsExt.check_expr — valid input does not throw" begin
        @variables z1 z2 dz1 dz2
        exprs = [-dz1 - z1 + z1^2, -dz2 - z2 + z1*z2]
        @test_nowarn ext.MORFESymbolicsExt.check_expr(exprs, [z1, z2, dz1, dz2])
    end

    @testset "ext.MORFESymbolicsExt.check_expr — throws on non-polynomial" begin
        @variables z1 dz1
        @test_throws AssertionError ext.MORFESymbolicsExt.check_expr([-dz1 - sin(z1)], [
            z1, dz1])
    end

    @testset "ext.MORFESymbolicsExt.check_expr — throws on constant term" begin
        @variables z1 dz1
        @test_throws AssertionError ext.MORFESymbolicsExt.check_expr([-dz1 - z1 + 5], [
            z1, dz1])
    end

    @testset "ext.MORFESymbolicsExt.check_expr — throws on non-polynomial before constant check" begin
        # both errors present: polynomial check fires first
        @variables z1 dz1
        @test_throws AssertionError ext.MORFESymbolicsExt.check_expr([-dz1 - sin(z1) + 5], [
            z1, dz1])
    end

    @testset "symbolics_to_NDOrdermodel — rejects non-polynomial exprs" begin
        @variables z1 dz1
        groups = ([z1], [dz1])
        @test_throws AssertionError symbolics_to_NDOrdermodel(
            [-dz1 - sin(z1)], groups
        )
    end

    @testset "symbolics_to_NDOrdermodel — rejects nonzero constant term" begin
        @variables z1 dz1
        groups = ([z1], [dz1])
        @test_throws AssertionError symbolics_to_NDOrdermodel(
            [-dz1 - z1 + 5], groups
        )
    end

    @testset "symbolics_to_NDOrdermodel with ext — rejects non-polynomial" begin
        @variables z1 dz1 r1
        groups = ([z1], [dz1])
        ext_var = [r1]
        @test_throws AssertionError symbolics_to_NDOrdermodel(
            [-dz1 - z1 + sin(r1)], groups, ext_var, [-r1]
        )
    end

    @testset "symbolics_to_NDOrdermodel with ext — rejects constant term" begin
        @variables z1 dz1 r1
        groups = ([z1], [dz1])
        ext_var = [r1]
        @test_throws AssertionError symbolics_to_NDOrdermodel(
            [-dz1 - z1 + 3], groups, ext_var, [-r1]
        )
    end
    @testset "all variables used — empty result" begin
        @variables z1 z2 dz1 dz2
        @test isempty(ext.MORFESymbolicsExt.check_all_vars_used(
            [-dz1 - z1 + z1*z2, -dz2 - z2],
            [z1, z2, dz1, dz2]
        ))
    end
    @testset "one unused variable" begin
        @variables z1 z2 dz1 dz2
        # z2 never appears
        unused = ext.MORFESymbolicsExt.check_all_vars_used(
            [-dz1 - z1, -dz2 - z1],
            [z1, z2, dz1, dz2]
        )
        @test length(unused) == 1
        @test any(isequal(u, z2) for u in unused)
    end

    @testset "multiple unused variables" begin
        @variables z1 z2 dz1 dz2
        # z2 and dz2 never appear
        unused = ext.MORFESymbolicsExt.check_all_vars_used(
            [-dz1 - z1, -dz1 - z1],
            [z1, z2, dz1, dz2]
        )
        @test length(unused) == 2
        @test any(isequal(u, z2) for u in unused)
        @test any(isequal(u, dz2) for u in unused)
    end
end  # @testset "input validation"

println("tests finished")