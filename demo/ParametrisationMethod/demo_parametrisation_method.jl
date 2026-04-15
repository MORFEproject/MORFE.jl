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
using .MORFE.Resonance: resonance_set_from_graph_style
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
B0 = [3.0 -1.0; -1.0 2.0]   # stiffness
B1 = 0.02 * B0   # proportional (light) damping
B2 = [2.0 0.0; 0.0 1.0]   # mass (highest-order coefficient)

# ------------------------------------------------------------------------------
# 2. Nonlinear terms and external system
# ------------------------------------------------------------------------------

# Cubic stiffness:  β * x³  (Duffing-type, β = 0.5)
term_cubic = MultilinearMap(
	(res, x1, x2, x3) -> (@. res += 0.5 * x1 * x2 * x3),
	(3, 0),
)

# Quadratic damping:  γ * ẋ²  (γ = 0.1)
term_drag = MultilinearMap(
	(res, xd1, xd2) -> (@. res += 0.1 * xd1 * xd2),
	(0, 2),
)

# External harmonic forcing applied to the first DOF
F_ext = ComplexF64[1.0, 0.0]
term_forcing = MultilinearMap(
	(res, r) -> (@. res += F_ext * r),
	(0, 0), 1,   # one external variable
)

# ExternalSystem: harmonic forcing ṙ = iΩ·r with Ω = 2.5
external_system = ExternalSystem((ComplexF64(2.5im),))

# ------------------------------------------------------------------------------
# 3. Build the full-order model
#    Embedding the ExternalSystem lets solve_cohomological_problem read the
#    external eigenvalues directly from model.external_system.eigenvalues.
# ------------------------------------------------------------------------------
model = NDOrderModel(
	(B0, B1, B2),
	(term_cubic, term_drag, term_forcing),
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
# The companion system has size ORD * FOM × ORD * FOM; nev must be < that.
nev    = 2 * FOM - 1   # request all but one eigenvalue
result = generalised_eigenpairs(
A_eig, B_eig;
nev = nev,
sigma = 0.1 + 1.0im,
which = :LM,
tol = 1e-12,
ncv = max(nev + 10, 20),
v0 = randn(MersenneTwister(42), 2 * FOM),
sort_largest_real = true
)

println("Converged eigenvalues: ", result.nconv)
println("Eigenvalues (first $(min(nev, result.nconv))):")
for (i, λ) in enumerate(result.values[1:min(nev, result.nconv)])
	println("  $i: ", round(λ; digits = 6))
end

# Position part of eigenvectors: first FOM rows of each eigenvector
eigenvectors_pos = result.vectors[1:FOM, :]

# ------------------------------------------------------------------------------
# 6. Select master modes and build the reduced-variable structure
# ------------------------------------------------------------------------------
ROM   = 2                   # number of master modes
N_EXT = 1                   # number of external forcing modes
NVAR  = ROM + N_EXT

master_eigenvalues = SVector{ROM, ComplexF64}(result.values[1:ROM])
master_modes       = eigenvectors_pos[:, 1:ROM]   # FOM × ROM

# Forcing direction (spatial distribution of the external load)
external_direction = ComplexF64[1.0, 0.5]
external_direction ./= norm(external_direction)

# Generalised right eigenmode matrix:  columns = [master modes | external direction]
generalised_right_eigenmodes = hcat(master_modes, external_direction)  # FOM × NVAR

# Left eigenmodes for the master modes (needed for the orthogonality conditions)
# In a properly implemented pipeline these come from the left eigenproblem;
# here we use the right eigenmodes as a placeholder for illustration.
left_eigenmodes = SVector{ROM, Vector{ComplexF64}}([master_modes[:, r] for r in 1:ROM]...)

# ------------------------------------------------------------------------------
# 7. Build multiindex set and resonance set
# ------------------------------------------------------------------------------
max_degree = 3
mset = all_multiindices_up_to(NVAR, max_degree)
println("\nMultiindex set: degree ≤ $max_degree in $NVAR variables → $(length(mset)) monomials")

outer_eigenvalues = Vector{ComplexF64}(external_system.eigenvalues)
# super_eigenvalues must cover all NVAR variables: [master | external]
all_eigenvalues = vcat(Vector{ComplexF64}(master_eigenvalues), outer_eigenvalues)
resonance_set = resonance_set_from_graph_style(
	ROM, mset, all_eigenvalues, outer_eigenvalues, 1e-8,
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
	generalised_right_eigenmodes, left_eigenmodes,
	resonance_set,
)

# ------------------------------------------------------------------------------
# 9. Display results
# ------------------------------------------------------------------------------
println("\n=== Parametrisation W (first $(min(10, length(mset))) monomials) ===")
for idx in 1:min(10, length(mset))
	pos = W.poly.coefficients[:, 1, idx]
	vel = W.poly.coefficients[:, 2, idx]
	println("  $(mset.exponents[idx]) → pos = $pos,  vel = $vel")
end

println("\n=== Reduced dynamics R (first $(min(10, length(mset))) monomials) ===")
for idx in 1:min(10, length(mset))
	coeffs = R.poly.coefficients[:, idx]
	println("  $(mset.exponents[idx]) → $coeffs")
end

println("\n" * "="^80)
println("Demo finished successfully.")
