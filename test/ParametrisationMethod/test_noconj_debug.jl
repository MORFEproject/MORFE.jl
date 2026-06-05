"""
Debug script: compare NoConjugatePermutation vs conjugate_permutation paths
for a small analytical second-order system with N_EXT = 2.

Run with:
    julia --project test/ParametrisationMethod/test_noconj_debug.jl

or interactively:
    include("test/ParametrisationMethod/test_noconj_debug.jl")
"""

using LinearAlgebra
using StaticArrays
using Printf
using MORFE

# ─────────────────────────────────────────────────────────────────────────────
# System parameters
# ─────────────────────────────────────────────────────────────────────────────
const _FOM        = 4
const _ROM        = 2
const _N_EXT      = 2
const _NVAR       = _ROM + _N_EXT
const _max_degree = 3
const _ζ          = 0.01

# Natural frequencies; first one is the master mode (conjugate pair λ₁, conj(λ₁))
const _ω_nat = [1.0, 3.0, 5.0, 7.0]

# ─────────────────────────────────────────────────────────────────────────────
# Linear matrices  (second-order system: M ẍ + C ẋ + K x = f, M = I)
# ─────────────────────────────────────────────────────────────────────────────
const _K = Matrix{Float64}(Diagonal(_ω_nat .^ 2))
const _C = Matrix{Float64}(Diagonal(2_ζ .* _ω_nat))
const _M = Matrix{Float64}(I(_FOM))

# ─────────────────────────────────────────────────────────────────────────────
# Eigenpairs (analytical; diagonal → mode shapes are standard basis vectors)
#   λ² + 2ζω λ + ω² = 0   →   λ = -ζω ± iω√(1-ζ²)
# ─────────────────────────────────────────────────────────────────────────────
const _λ₁ = complex(-_ζ * _ω_nat[1], _ω_nat[1] * sqrt(1 - _ζ^2))
const _λ₂ = conj(_λ₁)
const _master_eigenvalues = SVector{_ROM, ComplexF64}(_λ₁, _λ₂)

# Position component of right eigenvectors (DOF-1 mode shape = e₁, real)
const _master_modes = let
    m = zeros(ComplexF64, _FOM, _ROM)
    m[1, 1] = 1.0   # mode 1 position component
    m[1, 2] = 1.0   # mode 2 position component (conj = same for real mode shape)
    m
end

# Velocity component: λ × position for uncoupled diagonal system
const _master_modes_derivatives = let
    d = zeros(ComplexF64, _FOM, 1, _ROM)
    d[1, 1, 1] = _λ₁
    d[1, 1, 2] = _λ₂
    d
end

# For a symmetric real system, left eigenmodes = right eigenmodes
const _left_eigenmodes = _master_modes

# ─────────────────────────────────────────────────────────────────────────────
# Multiindex set (shared; all monomials of total degree 1…max_degree in NVAR vars)
# ─────────────────────────────────────────────────────────────────────────────
const _mset = all_multiindices_up_to(_NVAR, _max_degree; min_degree = 1)

# External forcing frequency = imag(λ₁)  → exact resonance with e₃
const _Ω_force = imag(_λ₁)

# ─────────────────────────────────────────────────────────────────────────────
# Build model for a given f_vec
# ─────────────────────────────────────────────────────────────────────────────
function _make_model(f_vec::AbstractVector{ComplexF64})
    f_vec_r      = real(f_vec)   # physical forcing is a real spatial vector
    term_quad    = MultilinearMap((res, x, y) -> (res .+= x .* y), (2, 0))
    term_forcing = MultilinearMap((res, r) -> (res .+= f_vec_r * sum(r)), (0, 0), 1)
    ext_sys      = ExternalSystem((im * _Ω_force, -im * _Ω_force))
    return NDOrderModel((_K, _C, _M), (term_quad, term_forcing), ext_sys)
end

function _make_resonance()
    super_eigs  = Vector{ComplexF64}([_master_eigenvalues..., im * _Ω_force, -im * _Ω_force])
    target_eigs = Vector{ComplexF64}(_master_eigenvalues)
    return resonance_set_from_complex_normal_form_style(
        _ROM, _mset, super_eigs, target_eigs, 0.1)
end

# ─────────────────────────────────────────────────────────────────────────────
# Run and compare both solve paths, printing every diverging monomial
# ─────────────────────────────────────────────────────────────────────────────
function compare_paths(case_label::String, f_vec::AbstractVector{ComplexF64})
    println("\n", "─"^72)
    println(case_label)
    @printf "  f_vec = %s\n" repr(f_vec)
    println("─"^72)

    model   = _make_model(f_vec)
    res_set = _make_resonance()
    shared  = (master_modes_derivatives = _master_modes_derivatives, show_progress = false)

    W_conj, R_conj = solve_cohomological_problem(
        model, _mset, _master_eigenvalues, _master_modes, _left_eigenmodes, res_set;
        conjugate_permutation = [2, 1, 4, 3],
        shared...,
    )
    W_noconj, R_noconj = solve_cohomological_problem(
        model, _mset, _master_eigenvalues, _master_modes, _left_eigenmodes, res_set;
        shared...,
    )

    Wc1 = W_conj.poly.coefficients    # FOM × ORD × L
    Wc2 = W_noconj.poly.coefficients
    Rc1 = R_conj.poly.coefficients    # NVAR × L
    Rc2 = R_noconj.poly.coefficients

    first_bad = nothing
    n_diverge = 0

    for (l, α) in enumerate(_mset.exponents)
        dW = norm(Wc1[:, :, l] - Wc2[:, :, l])
        dR = norm(Rc1[:, l]    - Rc2[:, l])
        (dW > 1e-10 || dR > 1e-10) || continue

        n_diverge += 1
        first_bad === nothing && (first_bad = (l, α, dW, dR))

        n_diverge <= 6 || continue   # limit terminal output
        @printf "  [l=%3d] α=%-22s deg=%d  ΔW=%.2e  ΔR=%.2e\n" l repr(Tuple(α)) sum(α) dW dR

        if n_diverge <= 3
            for j in axes(Wc1, 2)
                norm(Wc1[:, j, l] - Wc2[:, j, l]) > 1e-14 || continue
                @printf "          W_conj [:,j=%d]  = %s\n" j repr(round.(Wc1[:, j, l]; sigdigits = 6))
                @printf "          W_noconj[:,j=%d]  = %s\n" j repr(round.(Wc2[:, j, l]; sigdigits = 6))
            end
            @printf "          R_conj   = %s\n" repr(round.(Rc1[:, l]; sigdigits = 6))
            @printf "          R_noconj = %s\n" repr(round.(Rc2[:, l]; sigdigits = 6))
        end
    end

    if first_bad === nothing
        println("  ✓ PASS: both paths agree to 1e-10 on all $(length(_mset)) monomials.")
    else
        l, α, dW, dR = first_bad
        println()
        @printf "  ✗ FAIL: first divergence at l=%d  α=%s  degree=%d\n" l repr(Tuple(α)) sum(α)
        @printf "  Total diverging monomials: %d / %d\n" n_diverge length(_mset)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
println("MORFE.jl — NoConjugatePermutation vs conjugate_permutation debug")
@printf "System: FOM=%d, ROM=%d, N_EXT=%d, max_degree=%d\n" _FOM _ROM _N_EXT _max_degree
@printf "λ₁ = %s,  Ω_force = %.6f\n" repr(_λ₁) _Ω_force
@printf "%d monomials in multiindex set\n" length(_mset)

# Case A: real f_vec — physically the demo-like scenario (StructureModalDampingEigensolver)
# Both paths should agree because mode shapes and f_vec are both real.
compare_paths("Case A — real f_vec (demo-like)", ComplexF64[1.0, 0.0, 0.0, 0.0])

# Case B: complex f_vec input, coerced to real(f_vec) inside _make_model.
# Models a general eigensolver where mode shapes may be complex.
# With real(f_vec) applied before building the forcing term, both paths must agree.
compare_paths("Case B — complex input coerced to real(f_vec) (correct)", ComplexF64[1.0 + 0.3im, 0.0, 0.0, 0.0])
