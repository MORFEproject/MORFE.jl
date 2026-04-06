include(joinpath(@__DIR__, "../../src/MORFE.jl"))
using .MORFE.Multiindices: all_multiindices_up_to, find_in_set
using .MORFE.Resonance:
	SingleResonance,
	ResonanceSet,
	resonance_set_from_graph_style,
	resonance_set_from_complex_normal_form_style,
	update_resonance!,
	tolerance_from_local_estimate
using StaticArrays

# ============================================================================
# Setup: Distinguish internal, outer, external modes
# ============================================================================
internal_modes = 3      # number of internal modes (targets 1:internal_modes)
outer_count = 2         # number of outer modes (targets internal_modes+1 : internal_modes+outer_count)
external_count = 1      # number of external variables (only appear in multiindices, not as targets)

NVAR = internal_modes + external_count    # dimension of multiindices (internal + external)
N_TARGETS = internal_modes + outer_count  # total number of targets (internal + outer)

# Build multiindex set: all exponents up to degree 3 in NVAR variables
multiindexset = all_multiindices_up_to(NVAR, 3)
println("Multiindex set (first 10 entries) — dimension = $NVAR:")
for i in 1:min(10, length(multiindexset))
	println("  $i: $(multiindexset.exponents[i])")
end
println("...")

# ============================================================================
# 1. Building a ResonanceSet from SingleResonance objects
# ============================================================================
println("\n=== 1. Building with SingleResonance ===")

# Define some resonances. Targets are in 1..N_TARGETS.
res1 = SingleResonance(1, SVector{NVAR, Int}(1, 0, 0, 0))          # λ₁ ≈ e₁·λ
res2 = SingleResonance(2, SVector{NVAR, Int}(0, 1, 0, 0))          # λ₂ ≈ e₂·λ
res3 = SingleResonance(3, SVector{NVAR, Int}(0, 0, 1, 0), false)   # deactivate λ₃ resonance
res4 = SingleResonance(4, SVector{NVAR, Int}(0, 0, 0, 1))          # outer mode λ₄ ≈ e₄·λ
res5 = SingleResonance(5, SVector{NVAR, Int}(1, 1, 0, 0))          # outer mode λ₅ ≈ (e₁+e₂)·λ

# This multiindex does not exist (degree > 3) → will issue a warning.
res6 = SingleResonance(1, SVector{NVAR, Int}(1, 0, 1, 2))

# Build the ResonanceSet. The new constructor automatically applies the
# first‑order internal resonances (degree‑1 monomials with index ≤ internal_modes).
res_set1 = ResonanceSet(multiindexset, N_TARGETS, internal_modes, [res1, res2, res3, res4, res5, res6])

println("Resonance set after construction (first 10 monomials):")
for i in 1:min(10, length(res_set1.resonances))
	println("  $(multiindexset.exponents[i]) → $(res_set1.resonances[i])")
end

# ============================================================================
# 2. Editing the resonance set with update_resonance!
# ============================================================================
println("\n=== 2. Editing with update_resonance! ===")

# Activate the previously deactivated resonance for λ₃ (multiindex e₃)
γ = SVector{NVAR, Int}(0, 0, 1, 0)
update_resonance!(res_set1, [SingleResonance(3, γ, true)])
println("Activated λ₃ resonance for $γ")

# Add a new resonance: e₁+e₂ → λ₃
γ = SVector{NVAR, Int}(1, 1, 0, 0)
update_resonance!(res_set1, [SingleResonance(3, γ, true)])
println("Activated λ₃ resonance for $γ")

println("Resonance set after updates (first 10 monomials):")
for i in 1:min(10, length(res_set1.resonances))
	println("  $(multiindexset.exponents[i]) → $(res_set1.resonances[i])")
end

# ============================================================================
# 3. Graph style resonance set (heuristic)
# ============================================================================
println("\n=== 3. Graph style resonance set ===")

# Eigenvalues for the internal+external variables (used in the superharmonic)
super_eigenvalues = SVector{NVAR, ComplexF64}(
	1.0+0.0im,   # internal 1
	2.0 + 0.0im,   # internal 2
	3.0 + 0.0im,   # internal 3
	4.0 + 0.0im,   # external 1
)

# Eigenvalues for the outer modes
outer_eigenvalues = SVector{outer_count, ComplexF64}(
	5.0+0.0im,   # outer 1 (target 4)
	6.0 + 0.0im,   # outer 2 (target 5)
)

tol_scalar = 1e-8

# Now n_internal is passed directly as an integer (no Val)
res_set_graph = resonance_set_from_graph_style(
	internal_modes,
	multiindexset,
	super_eigenvalues,
	outer_eigenvalues,
	tol_scalar,
)

println("First 10 entries of graph style set:")
for i in 1:min(10, length(res_set_graph.resonances))
	println("  $(multiindexset.exponents[i]) → $(res_set_graph.resonances[i])")
end

# ============================================================================
# 4. Tolerance‑based resonance detection (complex normal form style)
# ============================================================================
println("\n=== 4. Tolerance‑based resonance detection ===")

# For the complex normal form, we need:
# - super_eigenvalues (internal+external) – same as above
# - target_eigenvalues (internal+outer) – eigenvalues of the targets
target_eigenvalues = SVector{N_TARGETS, ComplexF64}(
	1.0+0.0im,   # internal 1 (target 1)
	2.0 + 0.0im,   # internal 2 (target 2)
	3.0 + 0.0im,   # internal 3 (target 3)
	5.0 + 0.0im,   # outer 1 (target 4)
	6.0 + 0.0im,   # outer 2 (target 5)
)

# 4a. Simple scalar tolerance
res_set_tol_scalar = resonance_set_from_complex_normal_form_style(
	internal_modes,
	multiindexset,
	super_eigenvalues,
	target_eigenvalues,
	1e-9,
)

γ = SVector{NVAR, Int}(2, 0, 0, 0)   # 2·λ₁ (internal)
idx = find_in_set(multiindexset, γ)
println("Scalar tolerance (1e-9):")
println("  $γ → $(res_set_tol_scalar.resonances[idx])")

# 4b. Tolerance derived from eigenvalue condition numbers
# Condition numbers for all eigenvalues (internal, external, outer)
all_κ = SVector{NVAR + outer_count, Float64}(1.0, 1.2, 100.0, 1.0, 1.5, 1.0)   # λ₃ is ill‑conditioned
max_cond = 1e6   # capture resonances where condition number exceeds this

# Extract the condition numbers for the targets (internal+outer)
target_κ = SVector{N_TARGETS, Float64}(
	all_κ[1:internal_modes]...,                         # internal modes
	all_κ[(internal_modes+external_count+1):end]...,       # outer modes
)

# Compute the spectral radius (maximum absolute eigenvalue among all eigenvalues)
all_eigenvalues = SVector{NVAR + outer_count, ComplexF64}(
	1.0+0.0im,   # internal 1
	2.0 + 0.0im,   # internal 2
	3.0 + 0.0im,   # internal 3
	4.0 + 0.0im,   # external 1
	5.0 + 0.0im,   # outer 1
	6.0 + 0.0im,   # outer 2
)
spectral_radius = maximum(abs, all_eigenvalues)

# Compute the tolerance matrix for the internal+outer targets (all N_TARGETS)
tol_matrix = tolerance_from_local_estimate(
	multiindexset,
	super_eigenvalues,          # λ_super
	spectral_radius,
	target_κ,                   # condition numbers for the targets
	max_cond,
	1:N_TARGETS,                # target indices within target_κ (all)
)

res_set_tol_cond = resonance_set_from_complex_normal_form_style(
	internal_modes,
	multiindexset,
	super_eigenvalues,
	target_eigenvalues,
	tol_matrix,
)

println("\nTolerance based on condition number (κ₃ = 100, max_cond = 1e6):")
println("  $γ → $(res_set_tol_cond.resonances[idx])")
println("  For γ = (1,0,0,0) (first‑order internal):")
γ1 = SVector{NVAR, Int}(1, 0, 0, 0)
idx1 = find_in_set(multiindexset, γ1)
println("    → $(res_set_tol_cond.resonances[idx1])")

# ============================================================================
# 5. Show the difference between scalar and condition‑based tolerance
# ============================================================================
println("\n=== 5. Comparison: scalar vs condition‑based tolerance ===")
γ3 = SVector{NVAR, Int}(0, 0, 1, 0)   # multiindex for internal mode 3 (ill‑conditioned)
idx3 = find_in_set(multiindexset, γ3)
println("  γ = $γ3")
println("    Scalar tolerance:      $(res_set_tol_scalar.resonances[idx3])")
println("    Condition‑based tol:   $(res_set_tol_cond.resonances[idx3])")
# The condition‑based tolerance is larger for λ₃, so more resonances involving λ₃ will be flagged.

println("\n" * "="^80 * "\n")
println("Demo finished successfully.")
