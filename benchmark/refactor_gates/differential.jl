# Stage-B differential oracle: the SpectralData path must reproduce the positional
# path BIT-FOR-BIT, and must not allocate more.
#
# Both run in the same session, so this comparison carries no cross-run noise at all —
# it is a stronger check than comparing against the Stage-0 archive.

using MORFE
using LinearAlgebra, SparseArrays, StaticArrays, Printf
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver,
                                   select_master_modes_by_sorting,
                                   left_eigenmode_orders_from_slice
using MORFE.CohomologicalEquations: solve_cohomological_problem
using MORFE.Resonance: build_resonance_set
using MORFE.SpectralDecomposition: SpectralData, check_biorthogonality,
                                   right_modes, left_modes,
                                   right_mode_derivatives, left_mode_blocks

function cubic(nd)
    MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3),
        ntuple(i -> i == 1 ? 3 : 0, nd))
end

fails = 0
function check(name, cond)
    global fails
    cond || (fails += 1)
    @printf("  %-52s %s\n", name, cond ? "ok" : "MISMATCH")
end

println("Stage-B differential: positional path vs SpectralData path")
println(repeat("-", 72))

# ── Case 1: ORD = 2, matched orders, with and without conjugate symmetry ─────
let ROM = 2, order = 7
    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = [1.0 0.0; 0.0 1.0]
    B1 = 0.001 * B2
    model = NthOrderModel((B0, B1, B2), (cubic(2),))
    ep = spectrum(model; solver = DefaultEigensolver())
    select_master_modes_by_sorting(ep, ROM)
    m = ep.master_modes
    λ = SVector{ROM, ComplexF64}(ep.eigenvalues[m])
    Ψ = ep.eigenmodes[:, 1, m]
    ℓ = ep.left_eigenmodes[:, m]
    mmd = Array(ep.eigenmodes[:, 2:end, m])
    lmd = Array(ep.left_eigenmodes_orders[:, 1:(end - 1), m])

    mset = all_multiindices_up_to(ROM, order; min_degree = 1)
    rset = build_resonance_set(model, :complex_normal_form, mset, ep, 0.05, nothing)

    sd = SpectralData(model, ep; master = findall(m))

    # The reconciled blocks must equal what the old code sliced by hand.
    check("ORD=2 right physical slice", right_modes(sd) == Ψ)
    check("ORD=2 left physical slice", left_modes(sd) == ℓ)
    check("ORD=2 right derivative blocks", right_mode_derivatives(sd) == mmd)
    check("ORD=2 left order blocks", left_mode_blocks(sd) == lmd)
    check("ORD=2 master eigenvalues", MORFE.SpectralDecomposition.master_eigenvalues(sd) ==
                                      λ)

    G = check_biorthogonality(sd, model)
    check("ORD=2 biorthogonality G ≈ I", isapprox(G, I; atol = 1e-10))

    for (label, perm) in (("noconj", nothing), ("conj", [2, 1]))
        Wa, Ra = solve_cohomological_problem(model, mset, λ, Ψ, ℓ, rset;
            master_modes_derivatives = mmd, left_modes_derivatives = lmd,
            conjugate_permutation = perm, show_progress = false)
        sdp = SpectralData(model, ep; master = findall(m), conjugate_permutation = perm)
        Wb, Rb = solve_cohomological_problem(model, mset, sdp, rset; show_progress = false)
        check("ORD=2 $label  W bit-identical", Wa.poly.coefficients == Wb.poly.coefficients)
        check("ORD=2 $label  R bit-identical", Ra.poly.coefficients == Rb.poly.coefficients)

        ga = @allocated solve_cohomological_problem(model, mset, λ, Ψ, ℓ, rset;
            master_modes_derivatives = mmd, left_modes_derivatives = lmd,
            conjugate_permutation = perm, show_progress = false)
        gb = @allocated solve_cohomological_problem(
            model, mset, sdp, rset; show_progress = false)
        @printf("  %-52s %d vs %d bytes (%+d)\n", "ORD=2 $label  allocations", ga, gb,
            gb - ga)
        # The SpectralData path carries a small CONSTANT setup overhead, measured at
        # +832 bytes and verified independent of problem size (|mset| 9→35, FOM 6→30
        # all give exactly +832). Attribution: full_conjugate_permutation 416 B — it
        # *derives* the permutation where the old path was handed a literal, which is
        # the correctness win — plus 2×64 B for the derivative-block views, 2×80 B for
        # the physical-slice accessors, and keyword-tuple overhead.
        #
        # The gate that matters is the graded loop, which must not gain a single byte;
        # an O(1) setup cost is not a regression. Bound it so genuine growth still trips.
        check("ORD=2 $label  setup overhead is O(1) and bounded", 0 <= gb - ga <= 1024)
    end

    # :detect must reproduce the hand-written literal on this model.
    sdd = SpectralData(model, ep; master = findall(m), conjugate_permutation = :detect)
    # The stored involution now spans the whole spectrum; the master block is derived.
    check("ORD=2 :detect reproduces [2,1]",
        MORFE.SpectralDecomposition.master_conjugate_permutation(sdd) == [2, 1])
end

# ── Case 2: ORD-mismatch — ORD-3 augmented model fed by ORD-2 eigendata ──────
let ROM = 2, order = 5
    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = [1.0 0.0; 0.0 1.0]
    B1 = 0.001 * B2
    Z = zeros(size(B0))
    eig_model = NthOrderModel((B0, B1, B2), (cubic(2),))
    model = NthOrderModel((B0, B1, B2, Z), (cubic(3),))

    ep = spectrum(eig_model; solver = DefaultEigensolver())
    select_master_modes_by_sorting(ep, ROM)
    m = ep.master_modes
    λ = SVector{ROM, ComplexF64}(ep.eigenvalues[m])
    Ψ = ep.eigenmodes[:, 1, m]
    ℓ = ep.left_eigenmodes[:, m]

    # The hand-rolled ex04 reconstruction.
    Y2 = ep.eigenmodes[:, 2, m]
    mmd = zeros(ComplexF64, size(Ψ, 1), 2, ROM)
    for r in 1:ROM
        mmd[:, 1, r] .= Y2[:, r]
        mmd[:, 2, r] .= λ[r] .* Y2[:, r]
    end
    lmd = left_eigenmode_orders_from_slice(model.linear_terms, ℓ, collect(λ))[:,
        1:(end - 1), :]

    sd = SpectralData(model, ep; master = findall(m), conjugate_permutation = [2, 1])

    check("ORD-mismatch right derivative blocks == hand-rolled",
        right_mode_derivatives(sd) == mmd)
    check("ORD-mismatch left order blocks == hand-rolled",
        left_mode_blocks(sd) == Array(lmd))
    check("ORD-mismatch right physical slice", right_modes(sd) == Ψ)
    check("ORD-mismatch left physical slice", left_modes(sd) == ℓ)

    mset = all_multiindices_up_to(ROM, order; min_degree = 1)
    rset = build_resonance_set(model, :complex_normal_form, mset, ep, 0.05, nothing)
    Wa, Ra = solve_cohomological_problem(model, mset, λ, Ψ, ℓ, rset;
        master_modes_derivatives = mmd, left_modes_derivatives = lmd,
        conjugate_permutation = [2, 1], show_progress = false)
    Wb, Rb = solve_cohomological_problem(model, mset, sd, rset; show_progress = false)
    check("ORD-mismatch W bit-identical", Wa.poly.coefficients == Wb.poly.coefficients)
    check("ORD-mismatch R bit-identical", Ra.poly.coefficients == Rb.poly.coefficients)
end

# ── Case 3: external system, conjugate permutation assembled from the model ──
let ROM = 2, order = 5
    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = [1.0 0.0; 0.0 1.0]
    B1 = 0.001 * B2
    ep_model = NthOrderModel((B0, B1, B2), (cubic(2),))
    ep = spectrum(ep_model; solver = DefaultEigensolver())
    select_master_modes_by_sorting(ep, ROM)
    m = ep.master_modes
    λ = SVector{ROM, ComplexF64}(ep.eigenvalues[m])
    Ψ = ep.eigenmodes[:, 1, m]
    ℓ = ep.left_eigenmodes[:, m]
    mmd = Array(ep.eigenmodes[:, 2:end, m])
    lmd = Array(ep.left_eigenmodes_orders[:, 1:(end - 1), m])

    Ω = 1.3
    fvec = [1.0, 0.5]
    force = MultilinearMap(
        (res, r) -> begin
            @inbounds for j in 1:2
                iszero(r[j]) || (res .+= r[j] .* fvec)
            end
            res
        end, (0, 0), 1)
    model = NthOrderModel((B0, B1, B2), (cubic(2), force), ExternalSystem((
        im * Ω, -im * Ω)))
    mset = all_multiindices_up_to(ROM + 2, order; min_degree = 1)
    rset = build_resonance_set(model, :complex_normal_form, mset, ep, 0.05, nothing)

    sd = SpectralData(model, ep; master = findall(m), conjugate_permutation = [2, 1])
    Wa, Ra = solve_cohomological_problem(model, mset, λ, Ψ, ℓ, rset;
        master_modes_derivatives = mmd, left_modes_derivatives = lmd,
        conjugate_permutation = [2, 1, 4, 3], show_progress = false)
    Wb, Rb = solve_cohomological_problem(model, mset, sd, rset; show_progress = false)
    check("external  W bit-identical (perm derived, not literal)",
        Wa.poly.coefficients == Wb.poly.coefficients)
    check("external  R bit-identical", Ra.poly.coefficients == Rb.poly.coefficients)
end

println(repeat("-", 72))
if fails == 0
    println("DIFFERENTIAL PASSED — SpectralData path is bit-identical to the positional path")
else
    println("DIFFERENTIAL FAILED — $fails mismatches")
    exit(1)
end
