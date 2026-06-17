# Ferrite cells are parametrised by their reference shape: `AbstractCell{refshape}`.
_refshape(::Ferrite.AbstractCell{RS}) where {RS} = RS

"""
    mechanical_model(grid::Ferrite.Grid, constrained_nodes::Set{Int};
                     material, damping, fe_order = 2, quad_order = fe_order + 1)

Build an `AssembledMechanicalModel` from a Ferrite grid and a pre-computed set
of constrained node indices (all three displacement components clamped).
"""
function mechanical_model(grid::Ferrite.Grid, constrained_nodes::Set{Int};
        material::SVKMaterial,
        damping::RayleighDamping,
        fe_order::Int = 2,
        quad_order::Int = fe_order + 1)
    cell1 = Ferrite.getcells(grid, 1)
    RefShape = _refshape(cell1)
    ip = Lagrange{RefShape, fe_order}()^3
    qr = QuadratureRule{RefShape}(quad_order)
    geo_ip = Ferrite.geometric_interpolation(typeof(cell1))
    cv = CellValues(qr, ip, geo_ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, constrained_nodes, (x, t) -> zeros(3), [1, 2, 3]))
    close!(ch)
    update!(ch, 0.0)

    K_full = allocate_matrix(dh)
    M_full = allocate_matrix(dh)
    MORFE.ferrite_assemble_KM!(K_full, M_full, dh, cv, material.λ, material.μ, material.ρ)

    free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
    free_to_local = Dict(d => i for (i, d) in enumerate(free))
    n_free = length(free)

    K = K_full[free, free]
    M = M_full[free, free]
    C = damping.α * M + damping.β * K

    factory(deg::Int, max_cols::Int) = MORFE.ferrite_nonlinearity(deg, dh, cv,
        free_to_local, n_free, material.λ, material.μ; max_unique_cols = max_cols)

    return AssembledMechanicalModel(K, M, C, factory, (2, 3), material, damping,
        (n_dofs = n_free, n_dofs_total = ndofs(dh), backend = "Ferrite",
            fe_order = fe_order, quad_order = quad_order,
            dirichlet = "$(length(constrained_nodes)) constrained nodes"))
end

"""
    mechanical_model(grid::Ferrite.Grid; material, damping, dirichlet,
                     fe_order = 2, quad_order = fe_order + 1)
    mechanical_model(mesh_path::AbstractString; material, damping, dirichlet, ...)

Build an `AssembledMechanicalModel` (K, M, C on free DOFs + lazy SVK
nonlinearity factory) from a Ferrite grid or a mesh file.

- Gmsh `.msh`: pass `dirichlet` as a **String** naming the clamped facetset.
- COMSOL `.mphtxt`: pass `dirichlet` as a `Set{Int}` of boundary entity IDs
  (1-indexed, i.e. raw COMSOL ID + 1).
"""
function mechanical_model(grid::Ferrite.Grid;
        material::SVKMaterial,
        damping::RayleighDamping,
        dirichlet::String,
        fe_order::Int = 2,
        quad_order::Int = fe_order + 1)
    cell1 = Ferrite.getcells(grid, 1)
    RefShape = _refshape(cell1)
    ip = Lagrange{RefShape, fe_order}()^3
    qr = QuadratureRule{RefShape}(quad_order)
    geo_ip = Ferrite.geometric_interpolation(typeof(cell1))
    cv = CellValues(qr, ip, geo_ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, dirichlet), (x, t) -> zeros(3), [1, 2, 3]))
    close!(ch)
    update!(ch, 0.0)

    K_full = allocate_matrix(dh)
    M_full = allocate_matrix(dh)
    MORFE.ferrite_assemble_KM!(K_full, M_full, dh, cv, material.λ, material.μ, material.ρ)

    free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
    free_to_local = Dict(d => i for (i, d) in enumerate(free))
    n_free = length(free)

    K = K_full[free, free]
    M = M_full[free, free]
    C = damping.α * M + damping.β * K

    factory(deg::Int, max_cols::Int) = MORFE.ferrite_nonlinearity(deg, dh, cv,
        free_to_local, n_free, material.λ, material.μ; max_unique_cols = max_cols)

    return AssembledMechanicalModel(K, M, C, factory, (2, 3), material, damping,
        (n_dofs = n_free, n_dofs_total = ndofs(dh), backend = "Ferrite",
            fe_order = fe_order, quad_order = quad_order, dirichlet = dirichlet))
end

function mechanical_model(mesh_path::AbstractString; dirichlet, kwargs...)
    if endswith(mesh_path, ".mphtxt")
        grid, constrained = load_comsol_grid(mesh_path, dirichlet)
        return mechanical_model(grid, constrained; kwargs...)
    else
        return mechanical_model(FerriteGmsh.togrid(mesh_path); dirichlet = dirichlet, kwargs...)
    end
end
