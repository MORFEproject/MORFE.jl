# High-level `parametrise(model, order, eigenproblem; …)` entry point.
#
# The generic function `parametrise` is owned by `ParametrisationMethod`
# (declared there). This method body lives in a separate file because it calls
# `solve_cohomological_problem` from `CohomologicalEquations`, which is included
# *after* `ParametrisationMethod` in `src/MORFE.jl`. This file is therefore
# included at MORFE top-level scope after the `using .CohomologicalEquations`
# re-export block, so the bare names below resolve.

using StaticArrays: SVector

function ParametrisationMethod.parametrise(
        model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, LT, MT},
        order::Int,
        eigenproblem::Eigenproblem;
        resonance::Union{Symbol, ResonanceSet} = :graph,
        resonance_tol::Float64 = 1e-2,
        conjugacy_map = nothing,
        mset::Union{Nothing, MultiindexSet} = nothing,
        conjugate_permutation::Union{Nothing, Vector{Int}} = nothing,
        validate_mset::Bool = true,
        external_eigenvalues::Union{Nothing, Vector{ComplexF64}} = nothing,
        master_modes_derivatives = nothing,
        left_modes_derivatives = nothing) where {ORD, ORDP1, N_NL, N_EXT, LT, MT}

    # Extract eigenproblem
    master_mask = eigenproblem.master_modes
    @assert !isnothing(master_mask) "master_modes not set on Eigenproblem"
    ROM = sum(master_mask)
    @assert ROM > 0 "No modes chosen as master modes in Eigenproblem"
    NVAR = ROM + N_EXT

    # Copies, not views: a Bool-mask view is not strided, so downstream
    # BLAS/sparse products would fall back to slow generic matmul.
    # eigenproblem.eigenmodes is FOM × ORD × n_eigs; physical slice is [:, 1, :]
    master_modes = eigenproblem.eigenmodes[:, 1, master_mask]   # FOM × ROM
    left_eigenmodes = eigenproblem.left_eigenmodes[:, master_mask] # FOM × ROM

    # SVector is required by the type signature
    master_eigs_vec = eigenproblem.eigenvalues[master_mask]
    ROM = length(master_eigs_vec)
    master_eigenvalues = SVector{ROM, ComplexF64}(master_eigs_vec)

    # For ORD > 1: derivatives live in higher slices [:, 2:end, master_mask].
    # An explicit kwarg overrides the slicing (lower-order Eigenproblem storage
    # feeding a higher-order model — see docstring).
    if master_modes_derivatives === nothing
        master_modes_derivatives = ORD > 1 ?
                                   @view(eigenproblem.eigenmodes[:, 2:end, master_mask]) :   # FOM × (ORD-1) × ROM
                                   nothing
    end

    # For ORD > 1: lower-order left eigenvector blocks φ_1 … φ_{ORD-1} feed the
    # orthogonality row operators directly (no eigenvalue folding).
    if left_modes_derivatives === nothing
        left_modes_derivatives = if ORD > 1
            @assert eigenproblem.left_eigenmodes_orders !== nothing """
               Eigenproblem stores only the physical-space left eigenmode slice, but
               ORD > 1 orthogonality solves need the full left eigenvector order-blocks.
               Use an eigensolver path that supplies them (solve_left returns FOM × ORD × n).
               """
            @view(eigenproblem.left_eigenmodes_orders[:, 1:(ORD - 1), master_mask])   # FOM × (ORD-1) × ROM
        else
            nothing
        end
    end

    # Multiindex set: default graded-total, or a validated custom set.
    @assert order > 0 "order must be an integer bigger than zero"
    if mset === nothing
        # Graded-total sets are downward closed, contain every unit multiindex and are
        # symmetric under any coordinate permutation — nothing to check.
        mset = all_multiindices_up_to(NVAR, order; min_degree = 1)
    elseif validate_mset
        # Check here rather than only in the solve, so a malformed set is rejected
        # before the eigenproblem slicing and resonance-set construction.
        validate_multiindex_set(mset, NVAR, ROM;
            conjugate_permutation = conjugate_permutation)
    end

    # Generate ResonanceSet
    resonance_set = resonance isa ResonanceSet ? resonance :
                    build_resonance_set(model, resonance, mset,
        eigenproblem, resonance_tol, conjugacy_map;
        external_eigenvalues = external_eigenvalues)

    # Solve cohomological equation
    W,
    R = solve_cohomological_problem(
        model, mset,
        master_eigenvalues,
        master_modes, left_eigenmodes,
        resonance_set;
        master_modes_derivatives = master_modes_derivatives,
        left_modes_derivatives = left_modes_derivatives,
        conjugate_permutation = conjugate_permutation,
        validate_mset = false   # already checked above; don't walk the set twice
    )

    return W, R
end
