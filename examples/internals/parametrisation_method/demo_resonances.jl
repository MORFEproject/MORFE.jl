using MORFE.Multiindices: all_multiindices_up_to, find_in_set
using MORFE.Resonance:
                       resonance_set_from_graph_style,
                       resonance_set_from_complex_normal_form_style,
                       resonance_set_from_real_normal_form_style,
                       resonance_set_from_condition_number_estimate,
                       EigenvalueCondition, ConditionNumberEstimateCondition,
                       empty_resonance_set, set_resonance!, is_resonant, resonant_targets,
                       resonant_multiindices, n_internal

format_resonances(vec) = isempty(findall(vec)) ? "none" : join(findall(vec), ", ")

# Setup — three eigenvalue groups, explicit roles
master_eigenvalues = ComplexF64[1.0im, -1.0im, -1.0 + 1.0im, -1.0 - 1.0im]   # ROM = 4
external_eigenvalues = ComplexF64[1.1im, -1.1im]                             # N_EXT = 2, enter s only
outer_eigenvalues = ComplexF64[2.0im, -2.0im, 0.0]                        # N_OUT = 3, outer targets
ROM = length(master_eigenvalues)
N_EXT = length(external_eigenvalues)
N_OUT = length(outer_eigenvalues)
NVAR = ROM + N_EXT
multiindexset = all_multiindices_up_to(NVAR, 3)
println("Multiindex set: degree ≤ 3 in $NVAR variables → $(length(multiindexset)) multiindices")
println("Inner targets: 1..$ROM (master modes),  outer targets: $(ROM+1)..$(ROM+N_OUT)")

# 1. Graph style
println("\n=== 1. Graph style ===")
res_graph = resonance_set_from_graph_style(
    multiindexset, master_eigenvalues, external_eigenvalues, outer_eigenvalues, 1e-8)

for (idx, mi) in enumerate(multiindexset.exponents)
    println("  $mi → [$(format_resonances(resonant_targets(res_graph, idx)))]")
end

# 2. Graph style with condition number estimate (outer modes only)
println("\n=== 2. Graph style with condition number estimate (outer only) ===")
outer_κ = [1.5, 1.0, 1.0]   # κ for outer targets (3 entries, local 1:N_OUT)
ext_cond_local = ConditionNumberEstimateCondition(outer_eigenvalues, 6.0, outer_κ, 1e6,
    1:N_OUT)
res_graph_local = resonance_set_from_graph_style(
    multiindexset, master_eigenvalues, external_eigenvalues, ext_cond_local)
for (idx, mi) in enumerate(multiindexset.exponents)
    println("  $mi → [$(format_resonances(resonant_targets(res_graph_local, idx)))]")
end

# 3. Complex normal form (with outer targets)
println("\n=== 3. Complex normal form ===")
res_cnf = resonance_set_from_complex_normal_form_style(
    multiindexset, master_eigenvalues, 0.2;
    external_eigenvalues = external_eigenvalues,
    outer_eigenvalues = outer_eigenvalues)
for (idx, mi) in enumerate(multiindexset.exponents)
    println("  $mi → [$(format_resonances(resonant_targets(res_cnf, idx)))]")
end

# 4. Real normal form (with outer targets; conjugacy_map covers all ROM+N_OUT targets)
println("\n=== 4. Real normal form ===")
# global indices: inner pairs (1↔2, 3↔4), outer pairs (5↔6, 7 self-conjugate)
conjugacy_map = [2, 1, 4, 3, 6, 5, 7]   # length = ROM + N_OUT = 7, global 1-based
res_real = resonance_set_from_real_normal_form_style(
    multiindexset, master_eigenvalues, conjugacy_map, 1e-9;
    external_eigenvalues = external_eigenvalues,
    outer_eigenvalues = outer_eigenvalues)
for (idx, mi) in enumerate(multiindexset.exponents)
    println(
        "  $mi → complex NF: [$(format_resonances(resonant_targets(res_cnf, idx)))]   ",
        "\treal NF:   [$(format_resonances(resonant_targets(res_real, idx)))]")
end

# 5. Condition number estimate (all targets: inner + outer)
println("\n=== 5. Condition number estimate (all targets) ===")
target_κ = [1.0, 1.2, 100.0, 1.0, 1.5, 1.0, 1.0]  # length = ROM + N_OUT = 7
res_local = resonance_set_from_condition_number_estimate(
    multiindexset, master_eigenvalues, 6.0, target_κ, 1e6;
    external_eigenvalues = external_eigenvalues,
    outer_eigenvalues = outer_eigenvalues)
for (idx, mi) in enumerate(multiindexset.exponents)
    println("  $mi → [$(format_resonances(resonant_targets(res_local, idx)))]")
end

println("\n" * "="^60)
println("Demo finished successfully.")
