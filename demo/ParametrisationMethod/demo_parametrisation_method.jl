# demo_parametrisation_method.jl
# ==============================================================================
# Complete pipeline for computing a Spectral Submanifold (SSM) via the
# parametrisation method:
#
#   1. Define system matrices (M, C, K) and nonlinear terms
#   2. Build the NDOrderModel (embeds ExternalSystem for harmonic forcing)
#   3. Derive A and B for the generalised eigenproblem directly from the model
#   4. Solve the eigenproblem
#   5. Select master modes and build the generalised eigenmode matrix
#   6. Build the multiindex set and the resonance set
#   7. Solve cohomological equations → W (parametrisation) and R (reduced dynamics)
# ==============================================================================

include(joinpath(@__DIR__, "../../src/MORFE.jl"))

using .MORFE.Eigensolvers: generalised_eigenpairs
using .MORFE.Multiindices: all_multiindices_up_to
using .MORFE.Resonance: resonance_set_from_graph_style, resonance_set_from_complex_normal_form_style
using .MORFE.FullOrderModel: NDOrderModel, MultilinearMap, linear_first_order_matrices
using .MORFE.ExternalSystems: ExternalSystem
using .MORFE.ParametrisationMethod: Parametrisation, ReducedDynamics, coefficients
using .MORFE.CohomologicalEquations: solve_cohomological_problem

using LinearAlgebra
using Random
using StaticArrays

# ------------------------------------------------------------------------------
# 1. Define system matrices
#    Second-order ODE:  M ẍ + C ẋ + K x = F_nl(x, ẋ) + f_ext(t)
# ------------------------------------------------------------------------------
FOM = 2

# NDOrderModel stores linear terms as (B₀, B₁, …, B_ORD)
B0 = [2.0 -1.0; -1.0 2.0]   # stiffness
B1 = [0.01 0.0; 0.0 0.01]   # light damping
B2 = [1.0 0.0; 0.0 1.0]   # mass (highest-order coefficient)

# ------------------------------------------------------------------------------
# 2. Nonlinear terms and external system
# ------------------------------------------------------------------------------

# Cubic stiffness:  β * x³  (Duffing-type, β = 0.5)
term_cubic = MultilinearMap(
	(res, x1, x2, x3) -> (@. res += 1.0 * x1 * x2 * x3),
	(3, 0),
)

# Quadratic damping:  γ * ẋ²  (γ = 0.1)
term_drag = MultilinearMap(
	(res, xd1, xd2) -> (@. res += 0.1 * xd1 * xd2),
	(0, 2),
)

# External harmonic forcing
F_ext = ComplexF64[1.0, 1.0]
term_forcing = MultilinearMap(
	(res, r) -> (@. res += F_ext * r),
	(0, 0), 1,   # one external variable
)

# ExternalSystem: harmonic forcing ṙ = iΩ·r with Ω = 2.5
external_system = ExternalSystem((ComplexF64(1.0im),))

# ------------------------------------------------------------------------------
# 3. Build the full-order model
#    Embedding the ExternalSystem lets solve_cohomological_problem read the
#    external eigenvalues directly from model.external_system.eigenvalues.
# ------------------------------------------------------------------------------
model = NDOrderModel(
	(B0, B1, B2),
	(term_cubic, term_forcing), # term_drag
	external_system,
)

# ------------------------------------------------------------------------------
# 4. Derive generalised-eigenproblem matrices from the model
#    linear_first_order_matrices returns (A, B) for the companion system
#        B Ẋ = A X,    X = [x; ẋ]   (size 2*FOM × 2*FOM)
#    so the position part of any eigenvector is v[1:FOM].
# ------------------------------------------------------------------------------
A_eig, B_eig = linear_first_order_matrices(model)

# ------------------------------------------------------------------------------
# 5. Solve the generalised eigenproblem
# ------------------------------------------------------------------------------

# Compute the generalized eigenproblem (A - λB) φ = 0
eig_result = eigen(A_eig, B_eig)

# Extract the position part of eigenvectors (first FOM rows)
# Ensure FOM is defined and within matrix dimensions
@assert size(eig_result.vectors, 1) >= FOM "FOM exceeds eigenvector matrix rows"
eigenvectors_pos = eig_result.vectors[1:FOM, :]

# ------------------------------------------------------------------------------
# 6. Select master modes and build the reduced-variable structure
# ------------------------------------------------------------------------------
ROM   = 2          # number of master (dominant) modes
N_EXT = 1          # number of external forcing modes (for future use)
NVAR  = ROM + N_EXT

# Sort eigenvalues and corresponding eigenvectors by increasing magnitude
# (common choice for mode selection; can be replaced by, e.g., least damping)
sorted_idx = sortperm(abs.(eig_result.values))
sorted_vals = eig_result.values[sorted_idx]
sorted_vecs = eigenvectors_pos[:, sorted_idx]

println("\nAll eigenpairs (eigen):")
for (i, λ) in enumerate(sorted_vals)
	println("  mode $i: \t λ = $(round(λ, digits=6)) \t y = ", round.(sorted_vecs[:, i]; digits = 6))
end

# Select the first ROM eigenvalues/vectors as master modes
master_eigenvalues = SVector{ROM, ComplexF64}(sorted_vals[1:ROM])
master_modes       = sorted_vecs[:, 1:ROM]          # size: FOM × ROM

# Left eigenmodes for the master modes (needed for the orthogonality conditions)
# In a properly implemented pipeline these come from the left eigenproblem;
# here we use the right eigenmodes as a placeholder for illustration.
left_eigenmodes = master_modes   # FOM × ROM matrix

# Higher-order master mode derivatives W^(k)[e_r], k = 2 … ORD.
# For the companion-form eigenproblem with state ẑ = [x; ẋ; …], the k-th
# block of ẑ (rows (k-1)*FOM+1 : k*FOM) gives W^(k)[e_r] directly.
ORD_model = length(model.linear_terms) - 1   # = 2 for this second-order system
master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
for r in 1:ROM
    orig_idx = sorted_idx[r]
    for k in 1:(ORD_model - 1)   # k = 1 only for ORD = 2
        master_modes_derivatives[:, k, r] .= eig_result.vectors[(k*FOM + 1):((k+1)*FOM), orig_idx]
    end
end

println("\nSelected eigenpairs:")
for (i, λ) in enumerate(master_eigenvalues)
	println("  mode $i: \t λ = $(round(λ, digits=6)) \t y = ", round.(master_modes[:, i]; digits = 6))
end

# ------------------------------------------------------------------------------
# 7. Build multiindex set and resonance set
# ------------------------------------------------------------------------------

outer_eigenvalues = sorted_vals[(ROM+1):end]
println("\nOuter eigenvalues (non-master modes):")
for (i, λ) in enumerate(outer_eigenvalues)
	println("  mode $(ROM + i): \t λ = $(round(λ, digits=6))")
end
# super_eigenvalues must cover all NVAR variables: [master | external]
super_eigenvalues = vcat(Vector{ComplexF64}(master_eigenvalues), Vector{ComplexF64}(external_system.eigenvalues))
println("\nSuper-eigenvalues (master + external):")
for (i, λ) in enumerate(super_eigenvalues)
	println("  var $i: λ = $(round(λ, digits=6))")
end

max_degree = 3
mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)
println("\nMultiindex set: degree ≤ $max_degree in $NVAR variables → $(length(mset)) monomials")

resonance_set = resonance_set_from_graph_style(
	ROM, mset, super_eigenvalues, outer_eigenvalues, 0.05,
)

println("\nResonance set (graph style):")
for (idx, mi) in enumerate(mset.exponents)
	res_str = join(findall(resonance_set.resonances[:, idx]), ", ")
	isempty(res_str) && (res_str = "none")
	println("  $mi → [$res_str]")
end


# ------------------------------------------------------------------------------
# 8. Solve cohomological equations
#    External eigenvalues are read from model.external_system automatically.
# ------------------------------------------------------------------------------
W, R = solve_cohomological_problem(
	model, mset,
	master_eigenvalues,
	master_modes, left_eigenmodes,
	resonance_set;
	master_modes_derivatives = master_modes_derivatives,
)

# ------------------------------------------------------------------------------
# 9. Display results
# ------------------------------------------------------------------------------
n_monomials = min(20, length(mset))
println("\n=== Parametrisation W (first $n_monomials monomials) ===")
for idx in 1:n_monomials
	pos = W.poly.coefficients[:, 1, idx]
	vel = W.poly.coefficients[:, 2, idx]
	println("  $(mset.exponents[idx]) → \tpos = $pos\n\t\tvel = $vel\n")
end

println("\n=== Reduced dynamics R (first $n_monomials monomials) ===")
for idx in 1:n_monomials
	coeffs = R.poly.coefficients[:, idx]
	println("  $(mset.exponents[idx]) → $coeffs")
end

println("\n" * "="^80)
println("Demo finished successfully.")
