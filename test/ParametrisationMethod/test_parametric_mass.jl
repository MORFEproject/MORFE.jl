# A parametric MASS acts on ü, so it only exists on an augmented ORD = 3 model — and that
# is the one place the stored higher derivative levels of W are read back into the solve.
# They used to be propagated with the level-1 eigenmodes Φ at every level instead of
# Φ Λ^(j-1), so the acceleration level was wrong and R picked up errors from second order
# in the parameter on, including real parts in a conservative system's normal form.
#
# With M(θ) = M₀(1 + θ) and K fixed, every eigenvalue is λ₀(1 + θ)^(-1/2) EXACTLY, so the
# normal form's z₁θᶜ coefficients are known in closed form: λ₀ · binom(-1/2, c).
using StaticArrays

@testset "parametric mass on an augmented ORD = 3 model" begin
    K = [2.0 -1.0; -1.0 2.0]
    M0 = Matrix{Float64}(I, 2, 2)
    C = zeros(2, 2)
    Z = zeros(2, 2)                       # the augmented model's fourth linear block
    MAXθ = 4
    resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05, warn_outer = false)
    options = ParametrisationOptions(show_progress = false, verbose = false)

    # R's coefficients of z₁θᶜ, c = 0…MAXθ, for M(θ) = M₀ + θ M₁. The spectrum is the
    # second-order one, reconciled against the augmented order, as MORFEFerrite does it.
    function linear_coefficients(M1)
        ep = spectrum(NthOrderModel((K, C, M0)); solver = DefaultEigensolver())
        mass = MultilinearMap((res, x, r) -> (res .-= r[1] .* (M1 * x)), (0, 0, 1), 1)
        model = NthOrderModel((K, C, M0, Z), (mass,), ExternalSystem((0.0,)))
        spectral = SpectralData(model, ep; master = master_by_sorting(2))
        mset = MultiindexSet([SVector(a, b, c) for a in 0:1 for b in 0:1
                              for c in 0:MAXθ
                              if a + b ≤ 1 && a + b + c ≥ 1])
        _, R = parametrise(model, spectral, mset; resonance, options)
        exps = R.poly.multiindex_set.exponents
        return [R.poly.coefficients[1, findfirst(==(SVector(1, 0, c)), exps)]
                for c in 0:MAXθ]
    end
    binom(x, k) = prod((x - i) / (i + 1) for i in 0:(k - 1); init = 1.0)

    @testset "uniform mass: the exact binomial series" begin
        g = linear_coefficients(M0)
        for c in 0:MAXθ
            # Before the fix c = 2 gave 0.125 + 0.25i against the exact 0.375.
            @test g[c + 1] / g[1]≈binom(-0.5, c) atol=1e-12
        end
        @test maximum(abs ∘ real, g) < 1e-12        # conservative: no growth or decay
    end

    @testset "non-uniform mass: against the generalised eigenvalues" begin
        M1 = [0.7 0.2; 0.2 0.1]                     # couples the two modes
        g = linear_coefficients(M1)
        for θ in (0.02, 0.05)
            λ = sum(g[c + 1] * θ^c for c in 0:MAXθ)
            ω = sqrt.(real(eigvals(K, M0 + θ * M1)))
            # Truncation alone leaves O(θ^(MAXθ+1)); the bug left 3.5e-5 at θ = 0.02.
            @test minimum(abs.(abs(λ) .- ω)) < 10θ^(MAXθ + 1)
            @test abs(real(λ)) < 1e-12
        end
    end
end
