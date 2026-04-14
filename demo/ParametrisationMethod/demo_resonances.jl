include(joinpath(@__DIR__, "../../src/MORFE.jl"))  # adjust path
using .MORFE.Multiindices: all_multiindices_up_to, find_in_set
using .MORFE.Resonance:
	resonance_set_from_graph_style,
	resonance_set_from_complex_normal_form_style,
	resonance_set_from_real_normal_form_style,
	resonance_set_from_condition_number_estimate,
	EigenvalueCondition, ConditionNumberEstimateCondition,
	empty_resonance_set, set_resonance!, is_resonant, resonant_targets, resonant_multiindices

format_resonances(vec) = isempty(findall(vec)) ? "none" : join(findall(vec), ", ")

# Setup
internal_modes = 4
outer_count = 3
external_count = 2
NVAR = internal_modes + external_count
N_TARGETS = internal_modes + outer_count
multiindexset = all_multiindices_up_to(NVAR, 3)
println("Multiindex set: degree ≤ 3 in $NVAR variables → $(length(multiindexset)) multiindices")
println("Targets: 1..$internal_modes = internal, $(internal_modes+1)..$N_TARGETS = outer")

super_eigenvalues = ComplexF64[1.0im, -1.0im, -1.0+1.0im, -1.0-1.0im, 1.1im, -1.1im]
outer_eigenvalues = ComplexF64[2.0im, -2.0im, 0.0]
target_eigenvalues = ComplexF64[1.0im, -1.0im, -1.0+1.0im, -1.0-1.0im, 2.0im, -2.0im, 0.0]

# 1. Graph style (classic)
println("\n=== 1. Graph style (classic) ===")
res_graph = resonance_set_from_graph_style(internal_modes, multiindexset, super_eigenvalues, outer_eigenvalues, 1e-8)

for (idx, mi) in enumerate(multiindexset.exponents)
	println("  $mi → [$(format_resonances(res_graph.resonances[:, idx]))]")
end

# 2. Graph style with condition number estimate (outer modes only)
println("\n=== 2. Graph style with condition number estimate (outer only) ===")
target_κ = [1.0, 1.2, 100.0, 1.0, 1.5, 1.0, 1.0]  # κ for all targets (local indices 1..7)
outer_κ = target_κ[(internal_modes+1):end]   # κ for outer targets (local indices 1,2)
# target_indices are the *global* target indices for the outer modes (4 and 5)
ext_cond_local = ConditionNumberEstimateCondition(outer_eigenvalues, 6.0, outer_κ, 1e6, (internal_modes+1):(internal_modes+outer_count))
res_graph_local = resonance_set_from_graph_style(internal_modes, multiindexset, super_eigenvalues, ext_cond_local)
for (idx, mi) in enumerate(multiindexset.exponents)
	println("  $mi → [$(format_resonances(res_graph.resonances[:, idx]))]")
end

# 3. Complex normal form
println("\n=== 3. Complex normal form ===")
res_cnf = resonance_set_from_complex_normal_form_style(internal_modes, multiindexset, super_eigenvalues, target_eigenvalues, 0.2)
for (idx, mi) in enumerate(multiindexset.exponents)
	println("  $mi → [$(format_resonances(res_cnf.resonances[:, idx]))]")
end

# 4. Real normal form
println("\n=== 4. Real normal form ===")
conjugacy_map = [2, 1, 4, 3, 6, 5, 7]  # pairs of conjugate eigenvalues (1↔2, 3↔4, 5↔6, 7 is real)
res_real = resonance_set_from_real_normal_form_style(internal_modes, multiindexset, super_eigenvalues, target_eigenvalues, conjugacy_map, 1e-9)
for (idx, mi) in enumerate(multiindexset.exponents)
	println("  $mi → complex NF: [$(format_resonances(res_cnf.resonances[:, idx]))]   ",
		"\treal NF:   [$(format_resonances(res_real.resonances[:, idx]))]")
end


# 5. condition number estimate (normal form style, all targets)
println("\n=== 5. condition number estimate (all targets) ===")
res_local = resonance_set_from_condition_number_estimate(internal_modes, multiindexset, super_eigenvalues, target_eigenvalues, 6.0, target_κ, 1e6, 1:N_TARGETS)
for (idx, mi) in enumerate(multiindexset.exponents)
	println("  $mi → [$(format_resonances(res_local.resonances[:, idx]))]")
end

println("\n" * "="^60)
println("Demo finished successfully.")
