include(joinpath(@__DIR__, "../../src/MORFE.jl"))
using .MORFE.InvarianceEquation: precompute_column_polynomials,
	evaluate_column!,
	evaluate_external_rhs!,
	evaluate_system_matrix_and_lower_order_rhs!,
	assemble_cohomological_matrix_and_rhs
using LinearAlgebra
using StaticArrays: SVector

# -------------------------------------------------------------------
# 1.  Define problem dimensions and parameters
# -------------------------------------------------------------------
FOM = 5          # full‑order model dimension
ROM = 2          # number of master modes (reduced dynamics)
N_EXT = 1         # number of external forcing modes (NVAR = ROM + N_EXT)
NVAR = ROM + N_EXT

ORD = 2          # polynomial order of the ODE (B matrices: B₀, B₁, B₂)
ORDP1 = ORD + 1   # number of linear matrices

# Linear matrices B₀, B₁, B₂ (each FOM×FOM)
# For a simple second‑order system: B₂ x'' + B₁ x' + B₀ x = ...
B0 = 2.0 * Matrix{ComplexF64}(I, FOM, FOM)
B1 = 3.0 * Matrix{ComplexF64}(I, FOM, FOM)
B2 = 5.0 * Matrix{ComplexF64}(I, FOM, FOM)

linear_terms = (B0, B1, B2)   # NTuple{ORDP1} = (B₀, B₁, B₂)

# Reduced dynamics linear part (NVAR × NVAR) – Jordan form
# For demo, a matrix with eigenvalues λ₁, λ₂ = -0.1±1i (master) and λ₃ = 3.0 (external)
reduced_dynamics_linear = [
	-0.1+1.0im   0.0+0.0im   1.0+0.0im; # forcing excites master mode 1
	 0.0+0.0im  -0.1-1.0im   0.0+0.0im; # forcing does not excite master mode 2
	 0.0+0.0im   0.0+0.0im   0.0+1.0im
]  # Example matrix
println("\nReduced dynamics linear part (NVAR × NVAR):\n", repr("text/plain", reduced_dynamics_linear))

# Generalised eigenvectors (FOM × NVAR)
# Columns 1:ROM = master modes, ROM+1:NVAR = external forcing modes
# Only store the position part
generalised_eigenmodes = Matrix{ComplexF64}(I, FOM, NVAR)  # identity for simplicity
for i in 1:ORD
	println("\nGeneralised eigenmodes at order $i (FOM × NVAR) = ",
		repr("text/plain", generalised_eigenmodes * (reduced_dynamics_linear)^(i-1)))
end

# -------------------------------------------------------------------
# 2.  Precompute coefficient arrays C_coeffs and E_coeffs
# -------------------------------------------------------------------
C_coeffs, E_coeffs = precompute_column_polynomials(
	linear_terms, generalised_eigenmodes, reduced_dynamics_linear, ROM,
)

println("\n=== Precomputed coefficients ===")
for j in 1:ROM
	println("\nC_coeffs[$j] (master mode $j):\n", repr("text/plain", C_coeffs[j]))
end
for e in 1:N_EXT
	println("\nE_coeffs[$e] (external mode $e):\n", repr("text/plain", E_coeffs[e]))
end

# -------------------------------------------------------------------
# 3.  Data for a specific multiindex
# -------------------------------------------------------------------
# Frequency s = Σᵢ kᵢ λᵢ – here we pick an arbitrary value
s = -2.0 + 0.0im

# Resonance pattern for master modes (SVector{ROM, Bool})
resonance = SVector{ROM, Bool}(true, false)   # the single master mode is resonant

# Lower‑order couplings ξ_j (j = 1…ORD) – each is a FOM‑vector
# In a real simulation these come from lower‑order multiindices.
lower_order_couplings = SVector{ORD, Vector{ComplexF64}}(
	[0.1, 0.1, 0.1, 0.1, 0.1],       # position
	[10.0, 10.0, 10.0, 10.0, 10.0],  # velocity
)

# External dynamics amplitudes (length N_EXT)
external_dynamics = [-1000.0+0.0im]   # only one external mode

# -------------------------------------------------------------------
# 4.  Assemble the cohomological matrix and right‑hand side
# -------------------------------------------------------------------
M, rhs = assemble_cohomological_matrix_and_rhs(
	s, linear_terms, C_coeffs, E_coeffs,
	resonance, lower_order_couplings, external_dynamics,
)

println("\n=== Assembled cohomological system ===")
println("\nSystem matrix M (FOM × (FOM + nR)):\n", repr("text/plain", M))
println("\nRight‑hand side vector rhs:\n", rhs)

# -------------------------------------------------------------------
# 5.  (Optional) demonstrate low‑level functions independently
# -------------------------------------------------------------------
println("\n=== Low‑level function demonstrations ===")

# Evaluate a single column C_j(s) for j = 1 (master mode)
c1 = zeros(ComplexF64, FOM)
evaluate_column!(c1, s, 1, C_coeffs)

c2 = zeros(ComplexF64, FOM)
evaluate_column!(c2, s, 2, C_coeffs)

# Fused L(s) + lower‑order RHS evaluation
L = Matrix{ComplexF64}(undef, FOM, FOM)
lower_rhs = zeros(ComplexF64, FOM)
evaluate_system_matrix_and_lower_order_rhs!(
	L, lower_rhs, s, lower_order_couplings, linear_terms,
)

println("\nC₁($s) =\n", c1)
println("\nC₂($s) =\n", c2)
println("\nL($s) = \n", repr("text/plain", L))
println("\nLower‑order RHS contribution:\n", lower_rhs)

# Evaluate the external RHS contribution (should match the part from assembly)
rhs_ext = zeros(ComplexF64, FOM)
evaluate_external_rhs!(rhs_ext, s, external_dynamics, E_coeffs)
println("\nExternal RHS contribution:\n", rhs_ext)

println("\n=== Low‑level function manual computations ===")

C = B2 * generalised_eigenmodes * s +
	B2 * generalised_eigenmodes * reduced_dynamics_linear +
	B1 * generalised_eigenmodes

println("\nC₁($s) =\n", C * [1.0, 0.0, 0.0])

println("\nC₂($s) =\n", C * [0.0, 1.0, 0.0])

println("\nL($s) = \n", repr("text/plain", B2 * (s^2) + B1 * s + B0))

println("\nLower‑order RHS contribution:\n",
	-(B2 * s + B1) * lower_order_couplings[1] - B2 * lower_order_couplings[2],
)

println("\nExternal RHS contribution:\n", - C * [0.0, 0.0, external_dynamics[1]])


println("\n" * "="^80)
println("Demo finished successfully.")
