# using Test
# using MORFE
using MORFE.Resonance: empty_resonance_set, is_resonant, set_resonance!, resonant_targets,
                       resonant_multiindices
using MORFE.Multiindices: find_in_set
# using LinearAlgebra
# using SparseArrays

@testset "Resonance" begin
    @testset "empty_resonance_set / set_resonance! / is_resonant" begin
        mset = all_multiindices_up_to(2, 3)
        nmodes = 2
        rs = empty_resonance_set(mset, nmodes, nmodes)

        # Everything must start false
        for k in 1:length(mset), t in 1:nmodes
            @test !is_resonant(rs, k, t)
        end

        # Mark monomial 3 as resonant with target 1
        set_resonance!(rs, 1, 3, true)
        @test is_resonant(rs, 3, 1)
        @test !is_resonant(rs, 3, 2)

        # resonant_targets returns correct column
        tgt = resonant_targets(rs, 3)
        @test tgt[1] == true
        @test tgt[2] == false

        # resonant_multiindices returns the right index set
        ri = resonant_multiindices(rs, 1)
        @test 3 ∈ ri
        @test length(ri) == 1

        # Toggle back to false
        set_resonance!(rs, 1, 3, false)
        @test !is_resonant(rs, 3, 1)

        #Test set_resonance with multiindex
        idx = 7
        set_resonance!(rs, 2, Vector(mset.exponents[idx]), true)
        @test is_resonant(rs, idx, 2)

        # Test set_resonance for outer target
        set_resonance!(rs, nmodes + 1, 5, true)
        @test is_resonant(rs, 5, nmodes + 1)
        @test !is_resonant(rs, 5, nmodes + 2)
        ri_out = resonant_multiindices(rs, nmodes + 1)
        @test 5 ∈ ri_out

        # is_resonant returns false for out-of-range outer target when outer is nothing
        rs_no_outer = empty_resonance_set(mset, nmodes, 0)
        @test !is_resonant(rs_no_outer, 1, nmodes + 1)
    end

    @testset "resonance_set_from_graph_style — internal resonances" begin
        λ = ComplexF64[-0.1 + 1.0im, -0.1 - 1.0im]

        mset = all_multiindices_up_to(2, 4)
        rs = resonance_set_from_graph_style(mset, λ, ComplexF64[], ComplexF64[], 1e-6)

        for k in 1:length(mset)
            mi = mset.exponents[k]
            deg = sum(mi)
            if deg >= 2
                # Every degree ≥ 2 monomial must be resonant with ALL master modes
                for t in 1:n_internal(rs)
                    @test is_resonant(rs, k, t)
                end
            elseif deg == 1
                # Linear monomial eᵣ is resonant only with its own mode r
                r = findfirst(!iszero, mi)
                for t in 1:n_internal(rs)
                    @test is_resonant(rs, k, t) == (t == r)
                end
            end
        end
    end

    @testset "resonance_set_from_graph_style — forced resonance" begin
        # Master mode λ₁ = 1im, forcing Ω = 1im (exact resonance)
        master_eigenvalues = ComplexF64[1.0im, -1.0im]
        external_eigenvalues = ComplexF64[1.0im]
        n_internal = 2
        tol = 1e-8

        mset = all_multiindices_up_to(3, 2)  # 2 master + 1 forcing variable
        rs = resonance_set_from_graph_style(
            mset, master_eigenvalues, external_eigenvalues, ComplexF64[], tol)

        # The monomial z₁ (exponent [1,0,0]) has s = λ₁ = 1im ≈ Ω
        idx_z1 = find_in_set(mset, [0, 0, 1])
        @test idx_z1 !== nothing
        @test rs.inner_resonances[1, idx_z1]
        @test is_resonant(rs, idx_z1, 1)
    end

    @testset "resonance_set_from_graph_style — outer resonances" begin
        master_eigenvalues = ComplexF64[1.0im, -1.0im]
        outer_eigenvalues = ComplexF64[1.0im]

        tol = 1e-8
        n_int = length(master_eigenvalues)

        mset = all_multiindices_up_to(n_int, 3)

        rs = resonance_set_from_graph_style(
            mset, master_eigenvalues, ComplexF64[], outer_eigenvalues, tol)

        @test n_internal(rs) == n_int

        # Monomial [1,0,0]: s = λ₁ = 1im ≈ Ω  → outer target (index n_int+1) resonant
        idx_z1 = find_in_set(mset, [1, 0])
        @test idx_z1 !== nothing
        @test is_resonant(rs, idx_z1, n_int + 1)

        # Monomial [0,1,0]: s = λ₂ = -1im ≠ Ω → outer target NOT resonant
        idx_z2 = find_in_set(mset, [0, 1])
        @test idx_z2 !== nothing
        @test !is_resonant(rs, idx_z2, n_int + 1)

        # All degree ≥ 2 monomials must still be flagged for inner targets
        for k in 1:length(mset)
            deg = sum(mset.exponents[k])
            if deg >= 2
                for t in 1:n_int
                    @test is_resonant(rs, k, t)
                end
            end
        end
    end

    @testset "resonance_set_from_complex_normal_form_style" begin
        master_eigenvalues = ComplexF64[-0.05 + 2.0im, -0.05 - 2.0im]
        mset = all_multiindices_up_to(2, 4)
        n_internal = 2
        tol = 1e-6

        rs = resonance_set_from_complex_normal_form_style(
            mset, master_eigenvalues, tol)

        # The near-identity monomials e₁ = [1,0] and e₂ = [0,1] are linear;
        # CNF style should flag them as resonant with the matching target.
        idx_e1 = find_in_set(mset, [1, 0])
        idx_e2 = find_in_set(mset, [0, 1])
        @test idx_e1 !== nothing && idx_e2 !== nothing

        @test is_resonant(rs, idx_e1, 1)   # [1,0] resonant with λ₁
        @test !is_resonant(rs, idx_e1, 2)   # [1,0] not resonant with λ₂
        @test is_resonant(rs, idx_e2, 2)   # [0,1] resonant with λ₂
        @test !is_resonant(rs, idx_e2, 1)   # [0,1] not resonant with λ₁

        # Degree-2 monomial [2,0] with s = 2λ₁: NOT resonant unless 2λ₁ ≈ λⱼ
        idx_20 = find_in_set(mset, [2, 0])
        if idx_20 !== nothing
            # 2(-0.05+2i) = -0.1+4i ≠ any λⱼ → neither target resonant
            @test !is_resonant(rs, idx_20, 1)
            @test !is_resonant(rs, idx_20, 2)
        end

        idx_31 = find_in_set(mset, [3, 1])
        if idx_31 !== nothing
            # 3(-0.05+2i) + 1(-0.05 +2i)  = λ₁
            @test !is_resonant(rs, idx_31, 1)
            @test !is_resonant(rs, idx_31, 2)
        end
    end
    @testset "real_normal_form_style" begin
        # Conjugate pair: λ₁ = -0.05+2im, λ₂ = conj(λ₁)
        # conjugacy_map: [2, 1]  (each mode points to the other)
        master_eigenvalues = ComplexF64[-0.05 + 2.0im, -0.05 - 2.0im]
        conj_map = [2, 1]
        tol = 1e-6
        mset = all_multiindices_up_to(2, 3)
        n_int = 2

        rs = resonance_set_from_real_normal_form_style(
            mset, master_eigenvalues, conj_map, tol)
        @test n_internal(rs) == n_int

        # [1,0]: s = λ₁. |λ₁-s|=0 < tol → resonant with target 1.
        # Via conjugacy: also checks λ₂ for target 1; |λ₂-λ₁| ≠ 0, irrelevant.
        idx_e1 = find_in_set(mset, [1, 0])
        @test is_resonant(rs, idx_e1, 1)

        # [0,1]: s = λ₂. Resonant with target 2, AND via conjugacy with target 1
        # because conj(1)=2 and |λ₂ - λ₂| = 0 < tol.
        idx_e2 = find_in_set(mset, [0, 1])
        @test is_resonant(rs, idx_e2, 2)
        @test is_resonant(rs, idx_e2, 1)   # conjugacy symmetry

        # [2,0]: s = 2λ₁ → large imaginary part, not near any eigenvalue
        idx_20 = find_in_set(mset, [2, 0])
        @test !is_resonant(rs, idx_20, 1)
        @test !is_resonant(rs, idx_20, 2)

        # ── With outer eigenvalues ──
        Ω = ComplexF64[-0.05 + 2.0im, -0.05 - 2.0im]
        outer_conj = [2, 1]
        full_conj = vcat(conj_map, outer_conj .+ n_int)  # global map
        rs2 = resonance_set_from_real_normal_form_style(
            mset, master_eigenvalues, full_conj, tol; outer_eigenvalues = Ω)
        # [1,0]: s = λ₁ ≈ Ω₁ → outer target 1 resonant
        @test is_resonant(rs2, idx_e1, n_int + 1)
    end

    @testset "condition_number_estimate_style" begin
        master_eigenvalues = ComplexF64[-0.1 + 1.0im, -0.1 - 1.0im]
        n_int = 2
        ρ = 1.0          # spectral radius
        κ = [10.0, 10.0] # condition numbers for master modes
        max_c = 100.0        # max acceptable condition number
        mset = all_multiindices_up_to(2, 3)

        rs = resonance_set_from_condition_number_estimate(
            mset, master_eigenvalues, ρ, κ, max_c)

        @test n_internal(rs) == n_int
        @test rs.outer_resonances === nothing

        # [1,0]: s = λ₁. |λ₁-s|=0 → criterion satisfied → resonant with target 1
        idx_e1 = find_in_set(mset, [1, 0])
        @test is_resonant(rs, idx_e1, 1)

        # [2,0]: s = 2λ₁ = -0.2+2im. |λ₁-s| = |-0.1+1im - (-0.2+2im)| = |0.1-im| ≈ 1.005
        # criterion: 1.005 * 100 = 100.5 < 1.0 * 10 = 10? → false → not resonant
        idx_20 = find_in_set(mset, [2, 0])
        @test !is_resonant(rs, idx_20, 1)

        # ── With outer eigenvalues ──
        Ω = ComplexF64[-0.1 + 1.0im]
        κ_all = vcat(κ, [10.0])
        rs2 = resonance_set_from_condition_number_estimate(
            mset, master_eigenvalues, ρ, κ_all, max_c; outer_eigenvalues = Ω)
        # [1,0]: s = λ₁ ≈ Ω₁ → outer target resonant
        @test is_resonant(rs2, idx_e1, n_int + 1)

        # ── With conjugacy_map ──
        conj_map = [2, 1]
        rs3 = resonance_set_from_condition_number_estimate(
            mset, master_eigenvalues, ρ, κ, max_c; conjugacy_map = conj_map)
        # [0,1]: s = λ₂; |λ₁ - λ₂| = |2im| = 2; 2*100 = 200 < 1*10? → false
        # but |λ₂ - λ₂| = 0 → 0 < 10 → resonant via conjugacy with target 1
        idx_e2 = find_in_set(mset, [0, 1])
        @test is_resonant(rs3, idx_e2, 1)
    end
end# @testset "Resonances"
