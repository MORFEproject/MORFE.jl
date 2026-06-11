include(joinpath(@__DIR__, "..", "..", "..", "ext", "FerriteBackend", "ferrite_assembly.jl"))

function deform_grid(grid, dh::DofHandler,
                     θ₁::Float64, θ₂::Float64,
                     J₁::Tensor{2,3,Float64},
                     arch_mode_free::Vector{Float64},
                     free_to_local::Dict{Int,Int})
    # Expand free-DOF vector to full ndofs length (Dirichlet nodes → 0)
    arch_full = zeros(Float64, ndofs(dh))
    for (gdof, lidx) in free_to_local
        arch_full[gdof] = arch_mode_free[lidx]
    end

    # Extract nodal arch displacement.
    # For Lagrange^3 on a Q2 hex: local node l (1-indexed) has DOFs at
    # positions 3l-2, 3l-1, 3l within celldofs(cell).  All cells agree
    # at shared nodes, so repeated assignment is safe.
    n_nodes   = Ferrite.getnnodes(grid)
    node_disp = zeros(Vec{3,Float64}, n_nodes)
    for cell in CellIterator(dh)
        dofs  = celldofs(cell)
        nodes = grid.cells[Ferrite.cellid(cell)].nodes
        for l in eachindex(nodes)
            gn = nodes[l]
            node_disp[gn] = Vec{3,Float64}((arch_full[dofs[3l-2]],
                                             arch_full[dofs[3l-1]],
                                             arch_full[dofs[3l]]))
        end
    end

    # Displace coordinates; deepcopy preserves facetsets for Dirichlet BCs
    new_grid = deepcopy(grid)
    for n in 1:n_nodes
        X0 = new_grid.nodes[n].x
        new_grid.nodes[n] = Ferrite.Node(X0 + θ₁ * (J₁ ⋅ X0) + θ₂ * node_disp[n])
    end
    return new_grid
end

function assemble_exact_KM(grid_def, ip, geo_ip, qr,
                            λ::Float64, μ::Float64, ρ::Float64,
                            α::Float64, β::Float64)
    dh = DofHandler(grid_def)
    add!(dh, :u, ip)
    close!(dh)

    cv = CellValues(qr, ip, geo_ip)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid_def, "Dirichlet"),
                       (x, t) -> zeros(3), [1, 2, 3]))
    close!(ch)
    update!(ch, 0.0)

    free          = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
    free_to_local = Dict(d => i for (i, d) in enumerate(free))
    n_free        = length(free)

    K_full = allocate_matrix(dh)
    M_full = allocate_matrix(dh)
    assemble_KM!(K_full, M_full, dh, cv, λ, μ, ρ)

    K = K_full[free, free]
    M = M_full[free, free]
    C = α * M + β * K
    return (; dh, cv, free, free_to_local, n_free, K, M, C)
end
