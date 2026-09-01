using StaticArrays: SVector

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

    @testset "complex — degree_of_monomial" begin
        @variables x y

        # complex coefficient, same degree as real
        @test ext.MORFESymbolicsExt.degree_of_monomial(im * x) == 1
        @test ext.MORFESymbolicsExt.degree_of_monomial((1 + 2im) * x) == 1
        @test ext.MORFESymbolicsExt.degree_of_monomial(im * x^2) == 2
        @test ext.MORFESymbolicsExt.degree_of_monomial((3 + 4im) * x * y) == 2
        @test ext.MORFESymbolicsExt.degree_of_monomial(im * x^3) == 3
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

    @testset "complex — multidegree_of_monomial" begin
        @variables z1 z2 dz1 dz2
        groups = ([z1, z2], [dz1, dz2])

        # complex coefficient does not affect multidegree
        @test ext.MORFESymbolicsExt.multidegree_of_monomial(im * z1^2, groups) == (2, 0)
        @test ext.MORFESymbolicsExt.multidegree_of_monomial((1 + 2im) * z1 * z2, groups) ==
              (2, 0)
        @test ext.MORFESymbolicsExt.multidegree_of_monomial(im * dz1, groups) == (0, 1)
        @test ext.MORFESymbolicsExt.multidegree_of_monomial((3im) * z1 * dz1, groups) ==
              (1, 1)
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

        # sum of two monomials.  SymbolicUtils keeps a sum as a dictionary of term ⇒
        # coefficient, so `arguments` hands the terms back in hash order — [x, y] on
        # aarch64, [y, x] on the x86 runner.  Assert membership, not order; nothing
        # downstream reads these positionally, each monomial carries its own exponents.
        result = ext.MORFESymbolicsExt.seperate_into_monomials(x + y)
        @test all(m -> any(isequal(m), Num.(result)), [x, y])
        @test length(result) == 2

        # sum of three monomials
        result = ext.MORFESymbolicsExt.seperate_into_monomials(x^2 + x * y + 3 * y^2)
        @test length(result) == 3
    end

    @testset "complex — seperate_into_monomials" begin
        @variables x y

        # purely imaginary coefficient
        result = ext.MORFESymbolicsExt.seperate_into_monomials(im * x)
        @test length(result) == 1

        # mixed real + imaginary coefficients on same variable
        result = ext.MORFESymbolicsExt.seperate_into_monomials((1 + 2im) * x * y)
        @test length(result) == 2   # one real monomial, one imaginary monomial

        # sum: real term + complex term
        result = ext.MORFESymbolicsExt.seperate_into_monomials(x^2 + im * x * y)
        @test length(result) == 2

        # purely real expression returns same as before
        result = ext.MORFESymbolicsExt.seperate_into_monomials(x^2 + x * y)
        @test length(result) == 2
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
        @test B[1] ≈ A
        @test B[2] ≈ [-1.0 0.0; 0.0 -1.0]
    end

    @testset "extract_linear_matrices — sign convention" begin
        # exprs = -dz - z  →  B[1] = I, B[2] = I  (after the *-1 inside the function)
        @variables z1 dz1
        groups = ([z1,], [dz1,])
        exprs = [dz1 + z1]
        B = ext.MORFESymbolicsExt.extract_linear_matrices(exprs, groups)
        @test B[1] ≈ reshape([1.0], 1, 1)
        @test B[2] ≈ reshape([1.0], 1, 1)
    end

    @testset "complex — extract_linear_matrices" begin
        @variables z1 z2 dz1 dz2

        # complex linear system: B z where B has complex entries
        groups = ([z1, z2], [dz1, dz2])
        exprs = [
            dz1 + (1 + 2im) * z1,
            dz2 + (3 - im) * z2
        ]

        B = ext.MORFESymbolicsExt.extract_linear_matrices(exprs, groups)

        @test B[1] ≈ [(1 + 2im) 0; 0 (3 - im)]
        @test B[2] ≈ [1.0 0.0; 0.0 1.0]
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
        @test isequal(Num(monomials[1][1]), -z1^2)
        @test isequal(Num(monomials[2][1]), -z1 * z2)
        @test length(monomials[2]) == 1
        @test deg_monomials[1][1] == 2
        @test deg_monomials[2][1] == 2
        @test multideg_monomials[1][1] == (2,)
        @test multideg_monomials[2][1] == (2,)
    end
    @testset "complex — extract_nonlinear_monomials" begin
        @variables z1 dz1

        groups = ([z1], [dz1])
        exprs = [-dz1 - z1 + im * z1^2]

        linear_terms = ext.MORFESymbolicsExt.extract_linear_matrices(exprs, groups)
        N, monomials,
        deg_monomials,
        multideg_monomials = ext.MORFESymbolicsExt.extract_nonlinear_monomials(exprs, groups, linear_terms)

        @test N == 1
        @test length(monomials[1]) == 1   # one nonlinear monomial: im*z1^2
        @test deg_monomials[1][1] == 2
        @test multideg_monomials[1][1] == (2,)
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
        @test isequal(F_by_multiindex[(2,)], [-z1^2 - z1 * z2, -z1^2])
    end
    @testset "complex — group_monomials" begin
        @variables z1 z2 dz1 dz2
        groups = ([z1, z2], [dz1, dz2])

        exprs = [
            -dz1 - z1 + im * z1^2,
            -dz2 - z2 + (1 + im) * z1 * z2
        ]

        linear_terms = ext.MORFESymbolicsExt.extract_linear_matrices(exprs, groups)
        N, monomials,
        deg_monomials,
        multideg_monomials = ext.MORFESymbolicsExt.extract_nonlinear_monomials(exprs, groups, linear_terms)

        F_by_multiindex = ext.MORFESymbolicsExt.group_monomials(monomials, multideg_monomials, N)

        @test haskey(F_by_multiindex, (2,))
        @test length(F_by_multiindex[(2,)]) == N
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
    @testset "complex — polarize_monomial" begin
        @variables z1 z2
        groups = ([z1, z2],)
        mi = (2,)

        # complex coefficient is preserved in polarization
        pol_im, slotvars = ext.MORFESymbolicsExt.polarize_monomial(im * z1^2, groups, mi)
        pol_re, _ = ext.MORFESymbolicsExt.polarize_monomial(z1^2, groups, mi)

        # substituting all slot vars to 1: im*z1_1*z1_2 → im,  z1_1*z1_2 → 1
        sub = Dict(v => Num(1)
        for g in slotvars
        for sv in g for v in sv)

        val_im = Symbolics.substitute(pol_im isa Complex{Num} ? real(pol_im) : pol_im, sub)
        val_re = Symbolics.substitute(pol_re isa Complex{Num} ? real(pol_re) : pol_re, sub)

        # real part of im*z1^2 polarized should be 0, imaginary should be 1
        if pol_im isa Complex{Num}
            re_val = ComplexF64(Symbolics.value(Symbolics.substitute(real(pol_im), sub)))
            im_val = ComplexF64(Symbolics.value(Symbolics.substitute(imag(pol_im), sub)))
            @test re_val ≈ 0.0
            @test im_val ≈ 1.0
        end

        @test length(slotvars) == 1
        @test length(slotvars[1]) == 2
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
        model = model_from_symbolics(exprs, groups)

        @test length(model.nonlinear_terms) == 1
        term = model.nonlinear_terms[1]

        # metadata
        @test term.multiindex == (2,)      # degree 2 in x
        @test term.multiplicity_external == 0
        @test term.deg == 2

        # evaluate at x = [3.0]: should give 3^2 = 9
        x = [3.0]
        res = eval_term(term, (x,))
        @test res ≈ [-9.0]

        # scaling: f(α·x) = α^2 · f(x)
        α = 2.5
        res_scaled = eval_term(term, (α .* x,))
        @test res_scaled ≈ α^2 .* res

        # linearity in first slot (with second fixed):
        # f(x+y, x+y) = f(x,x) + f(x,y) + f(y,x) + f(y,y)
        # but evaluate_term! passes xs[1] to both slots, so
        # instead verify: result at [1.0] = 1, at [2.0] = 4
        @test eval_term(term, ([1.0],)) ≈ [-1.0]
        @test eval_term(term, ([2.0],)) ≈ [-4.0]
    end

    @testset "cross term z1*z2" begin
        @variables z1 z2 dz1 dz2
        groups = ([z1, z2], [dz1, dz2])
        exprs = [-dz1 - z1,
            -dz2 - z2 + z1 * z2]
        model = model_from_symbolics(exprs, groups)

        @test length(model.nonlinear_terms) == 1
        term = model.nonlinear_terms[1]
        @test term.multiindex == (2,)
        @test term.deg == 2

        # at x = [a, b]: row 1 = 0, row 2 = a*b
        a, b = 3.0, 2.0
        x = [a, b]
        res = eval_term(term, (x,))
        @test res[1] ≈ 0.0
        @test res[2] ≈ - a * b

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
        model = model_from_symbolics(exprs, groups)

        # One multiindex (2,) → one term
        @test length(model.nonlinear_terms) == 1
        term = model.nonlinear_terms[1]

        a, b = 2.0, 3.0
        x = [a, b]
        res = eval_term(term, (x,))

        expected = - a^2 - a * b
        @test res[1] ≈ expected
        @test res[2] ≈ expected
    end

    @testset "two multiindices — two MultilinearMaps (ORD=3)" begin
        @variables z1 dz1 ddz1
        groups = ([z1], [dz1], [ddz1])
        exprs = [-ddz1 - dz1 - z1 + z1^2 + dz1^2]
        model = model_from_symbolics(exprs, groups)

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
        @test res_z ≈ [-a^2]

        # term for dz1^2: uses only dx → result = b^2
        res_dz = eval_term(model.nonlinear_terms[term_dz], (x, dx))
        @test res_dz ≈ [-b^2]

        # total nonlinear contribution sums correctly
        total = res_z + res_dz
        @test total ≈ [-a^2 - b^2]
    end

    @testset "cubic term z1^3" begin
        @variables z1 dz1
        groups = ([z1], [dz1])
        exprs = [-dz1 - z1 - z1^3]
        model = model_from_symbolics(exprs, groups)

        @test length(model.nonlinear_terms) == 1
        term = model.nonlinear_terms[1]
        @test term.multiindex == (3,)
        @test term.deg == 3

        a = 2.0
        res = eval_term(term, ([a],))
        @test res ≈ [a^3]

        # scaling: f(α·x) = α^3 · f(x)
        α = 1.5
        @test eval_term(term, ([α * a],)) ≈ [α^3 * a^3]
    end

    @testset "evaluate_term! accumulates into res" begin
        @variables z1 dz1
        groups = ([z1], [dz1])
        exprs = [-dz1 - z1 + z1^2]
        model = model_from_symbolics(exprs, groups)
        term = model.nonlinear_terms[1]

        x = [3.0]
        res = [10.0]   # non-zero initial value
        MORFE.MultilinearMaps.evaluate_term!(res, term, (x,), nothing)
        @test res ≈ [10.0 - 9.0]   # 10 - 3^2
    end

    @testset "external forcing — linear in r" begin
        @variables z1 dz1 r1
        groups = ([z1], [dz1])
        ext_var = [r1]
        exprs = [-dz1 - z1 + 5 * r1]
        ext_exprs = [-r1]

        model = model_from_symbolics(exprs, groups, ext_var, ext_exprs)

        @test length(model.nonlinear_terms) == 1
        term = model.nonlinear_terms[1]
        @test term.multiplicity_external == 1
        @test sum(term.multiindex) == 0   # no state dependence

        x = [0.0]
        r = [2.0]
        res = eval_term(term, (x,), r)
        @test res ≈ [-10.0]   # -5 * r[1] = -5 * 2

        # scales linearly with r
        r2 = [3.0]
        res2 = eval_term(term, (x,), r2)
        @test res2 ≈ [-15.0]
    end

    @testset "mixed term z1*r1" begin
        @variables z1 dz1 r1
        groups = ([z1], [dz1])
        ext_var = [r1]
        exprs = [-dz1 - z1 + z1 * r1]
        ext_exprs = [-r1]

        model = model_from_symbolics(exprs, groups, ext_var, ext_exprs)

        term = only(model.nonlinear_terms)
        @test sum(term.multiindex) == 1
        @test term.multiplicity_external == 1
        @test term.deg == 2

        a, b = 3.0, 4.0
        res = eval_term(term, ([a],), [b])
        @test res ≈ [-a * b]

        # bilinear: scales as α*β when x→α·x, r→β·r
        α, β = 2.0, 3.0
        res2 = eval_term(term, ([α * a],), [β * b])
        @test res2 ≈ [- α * β * a * b]
    end

    @testset "complex — toMultilinearMaps" begin
        # helper mirroring `eval_term`, but complex-valued
        function eval_term_complex(term, xs, r = nothing)
            N = length(xs[1])
            res = zeros(ComplexF64, N)
            MORFE.MultilinearMaps.evaluate_term!(res, term, xs, r)
            return res
        end

        @testset "real-coefficient monomial, complex argument" begin
            @variables z1 dz1
            groups = ([z1], [dz1])
            exprs = [-dz1 - z1 - z1^2]
            model = model_from_symbolics(exprs, groups)
            term = model.nonlinear_terms[1]

            z = 1.0 + 2.0im
            res = eval_term_complex(term, ([z],))
            @test res ≈ [z^2]                     # (1+2i)^2 = -3+4i

            # scaling with a complex factor
            α = 0.5 - 1.0im
            res2 = eval_term_complex(term, ([α * z],))
            @test res2 ≈ [α^2 * z^2]
        end

        @testset "complex-coefficient monomial, real argument" begin
            @variables z1 dz1
            groups = ([z1], [dz1])
            exprs = [-dz1 - z1 - im * z1^2]
            model = model_from_symbolics(exprs, groups)
            term = model.nonlinear_terms[1]

            a = 3.0
            res = eval_term_complex(term, ([complex(a)],))
            @test res ≈ [im * a^2]
        end

        @testset "complex-coefficient monomial, complex argument" begin
            @variables z1 z2 dz1 dz2
            groups = ([z1, z2], [dz1, dz2])
            exprs = [-dz1 - z1,
                -dz2 - z2 - (1 + 2im) * z1 * z2]
            model = model_from_symbolics(exprs, groups)
            term = only(model.nonlinear_terms)

            z1v, z2v = 1.0 - 1.0im, 2.0 + 0.5im
            res = eval_term_complex(term, ([z1v, z2v],))
            @test res[1] ≈ 0.0
            @test res[2] ≈ (1 + 2im) * z1v * z2v

            # bilinear scaling with complex α, β on the two "slots"
            # (evaluate_term! passes xs[1] to all slots, so scale the single input)
            β = 0.3 + 0.7im
            res2 = eval_term_complex(term, ([β * z1v, β * z2v],))
            @test res2[2] ≈ β^2 * (1 + 2im) * z1v * z2v
        end

        @testset "cubic, complex argument (higher degree sanity check)" begin
            @variables z1 dz1
            groups = ([z1], [dz1])
            exprs = [-dz1 - z1 + z1^3]
            model = model_from_symbolics(exprs, groups)
            term = model.nonlinear_terms[1]

            z = 0.5 + 1.5im
            res = eval_term_complex(term, ([z],))
            @test res ≈ [-z^3]
        end
    end
end #@testset "toMultilinearMaps"

@testset "model_from_symbolics (no external)" begin
    # @testset "purely linear system returns zero nonlinear terms" begin
    #     @variables z1 z2 dz1 dz2
    #     groups = ([z1, z2], [dz1, dz2])

    #     exprs = [-dz1 - z1,
    #         -dz2 - z2]

    #     model = model_from_symbolics(exprs, groups)

    #     # Linear matrices: B[1] and B[2] should be identity
    #     @test model.linear_terms[1] ≈ [1.0 0.0; 0.0 1.0]
    #     @test model.linear_terms[2] ≈ [1.0 0.0; 0.0 1.0]

    #     # No nonlinear terms
    #     @test length(model.nonlinear_terms) == 0
    # end

    @testset "scalar Duffing oscillator" begin
        # ẍ + x + ε x³ = 0  written as first-order:
        #   ẋ = v,  v̇ = -x - x³
        # In NthOrderModel form (ORD=2):
        #   exprs = [-dz1 - z1 - z1^3,   (= -v - x - x³, the "v equation" row)
        #            -dz2 - z2]            doesn't matter, padding
        # Actually simpler: treat as single scalar 2nd order:
        @variables x dx
        groups = ([x], [dx])

        # -ẍ - x - x³ = 0  i.e.  ẍ + x + x³ = 0
        exprs = [dx + x - x^3]

        model = model_from_symbolics(exprs, groups)

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

        model = model_from_symbolics(exprs, groups)

        # B[2] should be identity (coefficient of dẑ)
        @test model.linear_terms[2] ≈ [-1.0 0.0; 0.0 -1.0]

        # B[1] encodes the linear stiffness: -dz1 - (1+1.5) z1  → B[1][1,1] = 2.5
        @test model.linear_terms[1][1, 1] ≈ -2.5
        @test model.linear_terms[1][2, 2] ≈ -3.5

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
        model_flat = model_from_symbolics(exprs_flat, groups_flat)

        # Array syntax
        @variables z[1:2] dz[1:2]
        groups_arr = (collect(z), collect(dz))
        exprs_arr = [
            -dz[1] - z[1] - 2 * z[1]^2 + z[1] * z[2],
            -dz[2] - z[2] + z[1]^2
        ]
        model_arr = model_from_symbolics(exprs_arr, groups_arr)

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
        exprs = [ddx + dx + x - x^2]

        model = model_from_symbolics(exprs, groups)

        @test model.linear_terms[1] ≈ reshape([1.0], 1, 1)
        @test model.linear_terms[2] ≈ reshape([1.0], 1, 1)
        @test model.linear_terms[3] ≈ reshape([1.0], 1, 1)
        @test length(model.nonlinear_terms) == 1
        @test sum(model.nonlinear_terms[1].multiindex) == 2
    end

    @testset "assert: groups of unequal length throws" begin
        @variables z1 z2 z3 dz1 dz2
        # groups[2] has one fewer variable than groups[1]
        @test_throws AssertionError model_from_symbolics(
            [-dz1 - z1, -dz2 - z2],
            ([z1, z2, z3], [dz1, dz2])
        )
    end

    @testset "assert: only one group throws (ORDP1 must be > 1)" begin
        @variables z1 dz1
        @test_throws AssertionError model_from_symbolics(
            [-z1],
            ([z1],)
        )
    end

    @testset "complex — model_from_symbolics (no external)" begin
        @testset "scalar system with purely imaginary nonlinearity" begin
            @variables z1 dz1
            groups = ([z1], [dz1])

            # -ż - z + im*z^2 = 0
            exprs = [-dz1 - z1 + im * z1^2]

            model = model_from_symbolics(exprs, groups)

            # linear matrices: B[1] = [1] (real), B[2] = [1]
            @test model.linear_terms[1] ≈ reshape([-1.0], 1, 1)
            @test model.linear_terms[2] ≈ reshape([-1.0], 1, 1)

            # one nonlinear term of degree 2
            @test length(model.nonlinear_terms) == 1
            @test model.nonlinear_terms[1].deg == 2
        end

        @testset "complex linear coefficient" begin
            @variables z1 dz1
            groups = ([z1], [dz1])

            # -ż + (1+2im)*z + z^2 = 0
            exprs = [-dz1 + (1 + 2im) * z1 + z1^2]

            model = model_from_symbolics(exprs, groups)

            # linear matrix should be complex
            @test model.linear_terms[1] ≈ reshape([(1 + 2im)], 1, 1)
            @test model.linear_terms[2] ≈ reshape([-1.0], 1, 1)
            @test length(model.nonlinear_terms) == 1
        end

        @testset "2-DOF system with complex coefficients" begin
            @variables z1 z2 dz1 dz2
            groups = ([z1, z2], [dz1, dz2])

            exprs = [
                dz1 + (1 + im) * z1 + 2im * z1^2,
                dz2 + (2 - im) * z2 + z1 * z2
            ]

            model = model_from_symbolics(exprs, groups)

            @test model.linear_terms[2] ≈ [1.0 0.0; 0.0 1.0]
            @test model.linear_terms[1][1, 1] ≈ (1 + im)
            @test model.linear_terms[1][2, 2] ≈ (2 - im)
            @test length(model.nonlinear_terms) == 1
        end
    end
end #@testset "model_from_symbolics (no external)"

@testset "model_from_symbolics (with external)" begin
    @testset "linear forcing — 2-DOF" begin
        @variables z1 z2 dz1 dz2 r1 r2
        groups = ([z1, z2], [dz1, dz2])
        ext_var = [r1, r2]

        # Purely linear in z and r (no nonlinear cross terms)
        exprs = [dz1 + z1 + 5 * r1,
            dz2 + z2 + 2 * r2]

        # External system: ṙ = -r  (harmonic oscillator at amplitude level)
        ext_exprs = [-r1, -r2]

        model = model_from_symbolics(exprs, groups, ext_var, ext_exprs)

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
            dz1 + z1 + 1.5 * z1 - 2 * z1^2 + 3 * z1 * z2 * r1 + 5 * r1,
            1 * dz2 + 3.5 * z2 + z1^2 - 5 * z1 * z2 + 21 // 4 * z2^2 + 2 * r2
        ]
        ext_exprs = [-r1, -r2]

        model = model_from_symbolics(exprs, groups, ext_var, ext_exprs)

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
        model = model_from_symbolics(exprs, groups, ext_var, ext_exprs)
        @test !isnothing(model)
    end

    @testset "assert: empty ext_var throws" begin
        @variables z1 dz1
        @test_throws AssertionError model_from_symbolics(
            [-dz1 - z1],
            ([z1], [dz1]),
            Num[],
            Num[]
        )
    end

    @testset "complex — model_from_symbolics (with external)" begin
        @testset "complex external forcing im*Omega0*r" begin
            @variables z1 dz1 r1
            groups = ([z1], [dz1])
            ext_var = [r1]
            Omega0 = 2.5

            exprs = [-dz1 - z1 + 5 * r1]
            ext_exprs = [im * Omega0 * r1]   # the motivating use case

            model = model_from_symbolics(exprs, groups, ext_var, ext_exprs)

            @test length(model.nonlinear_terms) == 1
            @test model.nonlinear_terms[1].multiplicity_external == 1
            @test sum(model.nonlinear_terms[1].multiindex) == 0
            @test !isnothing(model.external_system)
        end

        @testset "complex coefficient on mixed term" begin
            @variables z1 dz1 r1
            groups = ([z1], [dz1])
            ext_var = [r1]

            exprs = [-dz1 - z1 + (1 + im) * z1 * r1]
            ext_exprs = [-r1]

            model = model_from_symbolics(exprs, groups, ext_var, ext_exprs)

            term = only(model.nonlinear_terms)
            @test term.multiplicity_external == 1
            @test sum(term.multiindex) == 1
            @test term.deg == 2
        end

        @testset "fully complex external system" begin
            @variables z1 dz1 r1 r2
            groups = ([z1], [dz1])
            ext_var = [r1, r2]
            Omega0 = 1.5

            exprs = [-dz1 - z1 + r1]
            # ṙ = im*Omega0*r  written as 2x2 real system
            ext_exprs = [im * Omega0 * r1, im * Omega0 * r2]

            model = @test_warn "The following variables appear in `groups` but not in `exprs`: r2" begin
                model_from_symbolics(exprs, groups, ext_var, ext_exprs)
            end
            @test !isnothing(model.external_system)
        end
    end
end #@testset "model_from_symbolics (with external)"

@testset "externalsystem_from_symbolics" begin
    @testset "scalar linear system" begin
        @variables r
        ext_exprs = [-2 * r]

        ex_sys = ext.MORFESymbolicsExt.externalsystem_from_symbolics(ext_exprs, [r])
        @test ex_sys.first_order_dynamics.multiindex_set ==
              MultiindexSet(SVector{1, Int64}[[1]])
        @test ex_sys.first_order_dynamics.coefficients == ComplexF64[-2.0 + 0.0im;;]
    end

    @testset "2D linear system" begin
        @variables r1 r2
        ext_exprs = [-r1, -r2]

        ex_sys = ext.MORFESymbolicsExt.externalsystem_from_symbolics(ext_exprs, [r1, r2])
        @test ex_sys.first_order_dynamics.multiindex_set ==
              MultiindexSet(SVector{2, Int64}[[1, 0], [0, 1]])
        @test ex_sys.first_order_dynamics.coefficients ==
              ComplexF64[-1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im -1.0 + 0.0im]
    end

    @testset "nonlinear external system" begin
        @variables r1 r2
        ext_exprs = [-r1 + r1^2, -r2 + r1 * r2]

        ex_sys = ext.MORFESymbolicsExt.externalsystem_from_symbolics(ext_exprs, [r1, r2])
        @test ex_sys.first_order_dynamics.multiindex_set ==
              MultiindexSet(SVector{2, Int64}[[1, 0], [0, 1], [2, 0], [1, 1]])
        @test ex_sys.first_order_dynamics.coefficients ==
              ComplexF64[-1.0 + 0.0im 0.0 + 0.0im 1.0 + 0.0im 0.0 + 0.0im;
                         0.0 + 0.0im -1.0 + 0.0im 0.0 + 0.0im 1.0 + 0.0im]
    end

    @testset "complex — externalsystem_from_symbolics" begin
        @testset "purely imaginary linear system" begin
            @variables r1
            Omega0 = 2.0
            ext_exprs = [im * Omega0 * r1]

            ex_sys = ext.MORFESymbolicsExt.externalsystem_from_symbolics(ext_exprs, [r1])
            @test ex_sys.first_order_dynamics.multiindex_set ==
                  MultiindexSet(SVector{1, Int64}[[1]])
            @test ex_sys.first_order_dynamics.coefficients ==
                  ComplexF64[0.0 + 2.0im;;]
        end

        @testset "complex coefficient nonlinear external" begin
            @variables r1 r2
            ext_exprs = [im * r1 + r1^2, (1 + im) * r2]

            ex_sys = ext.MORFESymbolicsExt.externalsystem_from_symbolics(ext_exprs, [
                r1, r2])
            @test ex_sys.first_order_dynamics.multiindex_set ==
                  MultiindexSet(SVector{2, Int64}[[1, 0], [0, 1], [2, 0]])
            @test ex_sys.first_order_dynamics.coefficients ==
                  ComplexF64[0.0 + 1.0im 0.0 + 0.0im 1.0 + 0.0im;
                             0.0 + 0.0im 1.0 + 1.0im 0.0 + 0.0im]
        end
    end
end  # testset "externalsystem_from_symbolics"

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

    @testset "model_from_symbolics — rejects non-polynomial exprs" begin
        @variables z1 dz1
        groups = ([z1], [dz1])
        @test_throws AssertionError model_from_symbolics(
            [-dz1 - sin(z1)], groups
        )
    end

    @testset "model_from_symbolics — rejects nonzero constant term" begin
        @variables z1 dz1
        groups = ([z1], [dz1])
        @test_throws AssertionError model_from_symbolics(
            [-dz1 - z1 + 5], groups
        )
    end

    @testset "model_from_symbolics with ext — rejects non-polynomial" begin
        @variables z1 dz1 r1
        groups = ([z1], [dz1])
        ext_var = [r1]
        @test_throws AssertionError model_from_symbolics(
            [-dz1 - z1 + sin(r1)], groups, ext_var, [-r1]
        )
    end

    @testset "model_from_symbolics with ext — rejects constant term" begin
        @variables z1 dz1 r1
        groups = ([z1], [dz1])
        ext_var = [r1]
        @test_throws AssertionError model_from_symbolics(
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

@testset "DifferentialEquations.jl formulation of ODEs" begin
    @testset "_differential_equations_helper — order 1 linear system" begin
        function linear1!(du, u, p, t)
            du[1] = -u[1] + 2 * u[2]
            du[2] = -3 * u[2]
        end

        exprs, groups = ext._differential_equations_helper(linear1!, 1, 2)

        @test length(groups) == 2 # (u, du)
        u, du = groups
        expected = [du[1] - (-u[1] + 2u[2]), du[2] - (-3u[2])]

        @test all(isequal.(Symbolics.expand.(exprs), Symbolics.expand.(expected)))
    end

    @testset "_differential_equations_helper — order 3 linear system" begin
        function order3!(dddu, ddu, du, u, p, t)
            dddu[1] = -du[1] + 2 * u[2] + ddu[1]*ddu[2]
            dddu[2] = ddu[2] - 3 * u[2]
        end

        exprs, groups = ext._differential_equations_helper(order3!, 3, 2)

        @test length(groups) == 4
        u, du, ddu, dddu = groups
        expected = [
            dddu[1] - (-du[1] + 2 * u[2] + ddu[1]*ddu[2]), dddu[2] - (ddu[2] - 3 * u[2])]

        @test all(isequal.(Symbolics.expand.(exprs), Symbolics.expand.(expected)))
    end

    @testset "_differential_equations_helper_external — retrun" begin
        function f(r, p, t)
            dx = 10.0 * (r[2] - r[1])
            dy = r[1] * (28.0 - r[3]) - r[2]
            dz = r[1] * r[2] - (8 / 3) * r[3]
            return [dx, dy, dz]
        end

        exprs, r = ext._differential_equations_helper_external(f, 3)
        expected = [
            10.0 * (r[2] - r[1]), r[1] * (28.0 - r[3]) - r[2], r[1] * r[2] - (8 / 3) * r[3]]
        @test all(isequal.(Symbolics.expand.(exprs), Symbolics.expand.(expected)))
    end

    @testset "_differential_equations_helper_external — inplace" begin
        function f(dr, r, p, t)
            dr[1] = 10.0 * (r[2] - r[1])
            dr[2] = r[1] * (28.0 - r[3]) - r[2]
            dr[3] = r[1] * r[2] - (8 / 3) * r[3]
        end

        exprs, r = ext._differential_equations_helper_external(f, 3)
        expected = [
            10.0 * (r[2] - r[1]), r[1] * (28.0 - r[3]) - r[2], r[1] * r[2] - (8 / 3) * r[3]]
        @test all(isequal.(Symbolics.expand.(exprs), Symbolics.expand.(expected)))
    end

    @testset "wrong arity for the declared order" begin
        # this is a valid order-1 signature, but we ask for order=2
        wrong_order!(du, u, p, t) = (du[1] = -u[1])
        @test_throws AssertionError ext._differential_equations_helper(wrong_order!, 2, 1)
    end

    @testset "out-of-place function passed to the in-place helper" begin
        oop_f(u, p, t) = -u
        @test_throws Exception ext._differential_equations_helper(oop_f, 1, 1)
    end

    @testset "function with more than one method" begin
        # a generic function picks up 2 methods once both are defined
        ambiguous!(du, u, p, t) = (du[1] = -u[1])
        ambiguous!(du, u, v, p, t) = (du[1] = -u[1] - v[1])
        @test_throws AssertionError ext._differential_equations_helper(ambiguous!, 1, 1)
    end

    @testset "non-autonomous main system is rejected" begin
        nonautonomous!(du, u, p, t) = (du[1] = -u[1] + t)
        @test_throws AssertionError ext._differential_equations_helper(nonautonomous!, 1, 1)
    end

    @testset "non-autonomous external system is rejected" begin
        nonautonomous_ext!(dr, r, p, t) = (dr[1] = -r[1] + t)
        @test_throws AssertionError ext._differential_equations_helper_external(nonautonomous_ext!, 1)
    end

    @testset "external out-of-place function returns wrong length" begin
        bad_length(r, p, t) = [r[1]]        # nvars will be told as 2, only returns 1
        @test_throws AssertionError ext._differential_equations_helper_external(bad_length, 2)
    end

    @testset "external function with unrecognized signature" begin
        bad_sig(r, p) = -r                  # missing t
        @test_throws Exception ext._differential_equations_helper_external(bad_sig, 1)
    end

    # ── the public DifferentialEquations-shaped methods ────────────────────────────────
    # Only the private helpers above were covered before, which is how the argument-order
    # and dropped-parameter bugs in these three methods survived.

    @testset "model_from_symbolics(f!, order, nvars) ≡ the symbolic form" begin
        c, k, g = 0.1, 2.0, 6.0
        duffing!(ddu, du, u, p, t) = (ddu[1] = -c * du[1] - k * u[1] - g * u[1]^3)

        model_fn = model_from_symbolics(duffing!, 2, 1)

        # the helper writes exprs as dᵏu − f(...), so the equivalent expression form is
        @variables u[1:1] du[1:1] ddu[1:1]
        u, du, ddu = collect(u), collect(du), collect(ddu)
        exprs = [ddu[1] + c * du[1] + k * u[1] + g * u[1]^3]
        model_sym = model_from_symbolics(exprs, (u, du, ddu))

        for j in 1:3
            @test model_fn.linear_terms[j] ≈ model_sym.linear_terms[j]
        end
        @test length(model_fn.nonlinear_terms) == length(model_sym.nonlinear_terms) == 1
        @test model_fn.nonlinear_terms[1].multiindex ==
              model_sym.nonlinear_terms[1].multiindex
    end

    @testset "externalsystem_from_symbolics(f, nvars) ≡ the symbolic form" begin
        Ω = 1.3
        # real rotation — the function layout builds a Vector{Num}, so the RHS stays real
        rot!(dr, r, p, t) = (dr[1] = Ω * r[2]; dr[2] = -Ω * r[1])
        rot_oop(r, p, t) = [Ω * r[2], -Ω * r[1]]

        @variables r[1:2]
        r = collect(r)
        sys_sym = externalsystem_from_symbolics([Ω * r[2], -Ω * r[1]], r)

        for sys in (externalsystem_from_symbolics(rot!, 2),
            externalsystem_from_symbolics(rot_oop, 2))
            @test sys.first_order_dynamics.multiindex_set ==
                  sys_sym.first_order_dynamics.multiindex_set
            @test sys.first_order_dynamics.coefficients ≈
                  sys_sym.first_order_dynamics.coefficients
        end
    end

    @testset "coupled model_from_symbolics(f!, order, nvars, f_ext, nvars_ext)" begin
        Ω, c, k = 1.3, 0.2, 1.0
        # f! takes the external state as an extra argument, after u and before p
        forced!(ddu, du, u, r, p, t) = (ddu[1] = -c * du[1] - k * u[1] + r[1])
        rot!(dr, r, p, t) = (dr[1] = Ω * r[2]; dr[2] = -Ω * r[1])

        model_fn = model_from_symbolics(forced!, 2, 1, rot!, 2)

        @variables u[1:1] du[1:1] ddu[1:1] r[1:2]
        u, du, ddu, r = collect(u), collect(du), collect(ddu), collect(r)
        exprs = [ddu[1] + c * du[1] + k * u[1] - r[1]]
        model_sym = model_from_symbolics(
            exprs, (u, du, ddu), r, [Ω * r[2], -Ω * r[1]])

        for j in 1:3
            @test model_fn.linear_terms[j] ≈ model_sym.linear_terms[j]
        end
        @test model_fn.external_system.first_order_dynamics.coefficients ≈
              model_sym.external_system.first_order_dynamics.coefficients

        # the forcing term must actually read the external state
        @test length(model_fn.nonlinear_terms) == length(model_sym.nonlinear_terms)
        @test any(t -> t.multiplicity_external == 1, model_fn.nonlinear_terms)
    end

    @testset "p and p_ext reach the user's function" begin
        # each of these reads p[1]; before the fix p was replaced by (), so they threw
        scaled!(dr, r, p, t) = (dr[1] = p[1] * r[2]; dr[2] = -p[1] * r[1])
        sys = externalsystem_from_symbolics(scaled!, 2; p = (2.5,))
        @variables r[1:2]
        r = collect(r)
        ref = externalsystem_from_symbolics([2.5 * r[2], -2.5 * r[1]], r)
        @test sys.first_order_dynamics.coefficients ≈ ref.first_order_dynamics.coefficients

        stiff!(ddu, du, u, p, t) = (ddu[1] = -p[1] * u[1] - 0.3 * du[1])
        model = model_from_symbolics(stiff!, 2, 1; p = (7.0,))
        @test model.linear_terms[1] ≈ reshape([7.0], 1, 1)
        @test model.linear_terms[2] ≈ reshape([0.3], 1, 1)

        # both parameter sets on the coupled method, each reaching its own function
        forced!(ddu, du, u, r, p, t) = (ddu[1] = -p[1] * u[1] - 0.3 * du[1] + r[1] + r[2])
        coupled = model_from_symbolics(forced!, 2, 1, scaled!, 2; p = (7.0,), p_ext = (2.5,))
        @test coupled.linear_terms[1] ≈ reshape([7.0], 1, 1)
        @test coupled.external_system.first_order_dynamics.coefficients ≈
              ref.first_order_dynamics.coefficients
    end

    @testset "coupled f! must carry the external-state argument" begin
        rot!(dr, r, p, t) = (dr[1] = r[2]; dr[2] = -r[1])
        no_ext!(ddu, du, u, p, t) = (ddu[1] = -u[1])   # arity order+1, not order+2
        @test_throws AssertionError model_from_symbolics(no_ext!, 2, 1, rot!, 2)
    end
end  # "DifferentialEquations.jl formulation of ODEs"
