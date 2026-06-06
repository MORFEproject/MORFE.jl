module ComsolToGmsh

using Gmsh

export comsol_to_gmsh, comsol_to_gmsh_linear

const VE1n = 1

# Number of nodes of a linear line element (2)
const BE2n = 2

# Number of nodes of a quadratic line element (3)
const BE3n = 3

# Number of nodes of a linear triangular element (3)
const TR3n = 3

# Number of nodes of a quadratic triangular element (6)
const TR6n = 6

# Number of nodes of a linear tetrahedral element (4)
const TE4n = 4

# Number of nodes of a quadratic tetrahedral element (10)
const T10n = 10

# Number of nodes of a linear wedge element (6)
const PE6n = 6

# Number of nodes of a quadratic 15-nodes wedge element (15)
const P15n = 15

# Number of nodes of a quadratic 18-nodes wedge element (18)
const P18n = 18

# Number of nodes of a linear square element (4)
const QU4n = 4

# Number of nodes of a quadratic serendipity square element (8)
const QU8n = 8

# Number of nodes of a quadratic Lagrange square element (9)
const QU9n = 9

# Number of nodes of a linear hexahedral element (8)
const HE8n = 8

# Number of nodes of a quadratic serendipity hexahedral element (20)
const H20n = 20

# Number of nodes of a quadratic Lagrange hexahedral element (27)
const H27n = 27

# Number of supported boundary element
const Snse = 5

# Number of supported volume element
const Vnse = 8

# Maximum number of elements connected one to the other
const ncv_max = 30

# Maximum number nodes for an element
const nne_max = 27

"""
    _read_mesh( mesh_file::String)

Reads a Comsol .mphtxt file and extracts
- `n2c`:   node coordinates
- `e2nT6`: node list of elements of type T6 (triangle)
- `e2qT6`: label of elements of type T6 (triangle)
- `e2nQ9`: node list of elements of type Q9 (rectangle)
- `e2qQ9`: label of elements of type Q9 (rectangle)
- `e2nT10`: node list of elements of type T10 (triangle)
- `e2qT10`: label of elements of type T10 (triangle)
- `e2nP18`: node list of elements of type P18 (prism)
- `e2qP18`: label of elements of type P18 (prism)
- `e2nH27`: node list of elements of type H27 (hex)
- `e2qH27`: label of elements of type H27 (hex)

Based on the mesh.jl file from Morfe2.0.

"""
function _read_mesh(mesh_file::String)
    neT6 = 0
    e2nT6 = Int64[]
    e2gT6 = Int64[]
    neQ9 = 0
    e2nQ9 = Int64[]
    e2gQ9 = Int64[]
    neT10 = 0
    e2nT10 = Int64[]
    e2gT10 = Int64[]
    neP18 = 0
    e2nP18 = Int64[]
    e2gP18 = Int64[]
    neH27 = 0
    e2nH27 = Int64[]
    e2gH27 = Int64[]
    # e2n stores nodes
    # e2g stores geometric identity
    nn = 0
    n2c = 0
    dim = 3
    #
    fname = mesh_file
    # extract mesh infos
    open(fname, "r") do fhand
        while !eof(fhand)
            line = readline(fhand)
            if (contains(line, "# number of mesh vertices"))
                line = split(line)
                nn = Meta.parse(line[1])
                n2c = Vector{Float64}(undef, nn * dim)
            elseif (contains(line, "# Mesh vertex coordinates"))
                for i in 1:(nn)
                    line = split(readline(fhand))
                    for j in 1:dim
                        n2c[j + (i - 1) * dim] = Meta.parse(line[j])
                    end
                end
            elseif (contains(line, "tri2 # type name"))
                line = readline(fhand)
                line = readline(fhand)
                line = readline(fhand)
                line = split(readline(fhand))
                neT6 = Meta.parse(line[1])
                line = readline(fhand)
                e2nT6 = Vector{Int64}(undef, neT6 * TR6n)
                e2gT6 = Vector{Int64}(undef, neT6)
                for i in 1:neT6
                    line = split(readline(fhand))
                    for j in 1:TR6n
                        e2nT6[j + (i - 1) * TR6n] = Meta.parse(line[j]) + 1
                    end
                end
                while (true)
                    line = readline(fhand)
                    if (contains(line, "# Geometric entity indices"))
                        for i in 1:neT6
                            line = readline(fhand)
                            e2gT6[i] = Meta.parse(line) + 1
                        end
                        break
                    end
                end
            elseif (contains(line, "quad2 # type name"))
                line = readline(fhand)
                line = readline(fhand)
                line = readline(fhand)
                line = split(readline(fhand))
                neQ9 = Meta.parse(line[1])
                line = readline(fhand)
                e2nQ9 = Vector{Int64}(undef, neQ9 * QU9n)
                e2gQ9 = Vector{Int64}(undef, neQ9)
                for i in 1:neQ9
                    line = split(readline(fhand))
                    for j in 1:QU9n
                        e2nQ9[j + (i - 1) * QU9n] = Meta.parse(line[j]) + 1
                    end
                end
                while (true)
                    line = readline(fhand)
                    if (contains(line, "# Geometric entity indices"))
                        for i in 1:neQ9
                            line = readline(fhand)
                            e2gQ9[i] = Meta.parse(line) + 1
                        end
                        break
                    end
                end
            elseif (contains(line, "tet2 # type name"))
                line = readline(fhand)
                line = readline(fhand)
                line = readline(fhand)
                line = split(readline(fhand))
                neT10 = Meta.parse(line[1])
                line = readline(fhand)
                e2nT10 = Vector{Int64}(undef, neT10 * T10n)
                e2gT10 = Vector{Int64}(undef, neT10)
                for i in 1:neT10
                    line = split(readline(fhand))
                    for j in 1:T10n
                        e2nT10[j + (i - 1) * T10n] = Meta.parse(line[j]) + 1
                    end
                end
                while (true)
                    line = readline(fhand)
                    if (contains(line, "# Geometric entity indices"))
                        for i in 1:neT10
                            line = readline(fhand)
                            e2gT10[i] = Meta.parse(line)
                        end
                        break
                    end
                end
            elseif (contains(line, "prism2 # type name"))
                line = readline(fhand)
                line = readline(fhand)
                line = readline(fhand)
                line = split(readline(fhand))
                neP18 = Meta.parse(line[1])
                line = readline(fhand)
                e2nP18 = Vector{Int64}(undef, neP18 * P18n)
                e2gP18 = Vector{Int64}(undef, neP18)
                for i in 1:neP18
                    line = split(readline(fhand))
                    for j in 1:P18n
                        e2nP18[j + (i - 1) * P18n] = Meta.parse(line[j]) + 1
                    end
                end
                while (true)
                    line = readline(fhand)
                    if (contains(line, "# Geometric entity indices"))
                        for i in 1:neP18
                            line = readline(fhand)
                            e2gP18[i] = Meta.parse(line)
                        end
                        break
                    end
                end
            elseif (contains(line, "hex2 # type name"))
                line = readline(fhand)
                line = readline(fhand)
                line = readline(fhand)
                line = split(readline(fhand))
                neH27 = Meta.parse(line[1])
                line = readline(fhand)
                e2nH27 = Vector{Int64}(undef, neH27 * H27n)
                e2gH27 = Vector{Int64}(undef, neH27)
                for i in 1:neH27
                    line = split(readline(fhand))
                    for j in 1:H27n
                        e2nH27[j + (i - 1) * H27n] = Meta.parse(line[j]) + 1
                    end
                end
                while (true)
                    line = readline(fhand)
                    if (contains(line, "# Geometric entity indices"))
                        for i in 1:neH27
                            line = readline(fhand)
                            e2gH27[i] = Meta.parse(line)
                        end
                        break
                    end
                end
            end
        end
    end

    return nn, n2c, e2nT6, e2gT6, e2nQ9, e2gQ9, e2nT10, e2gT10, e2nP18, e2gP18, e2nH27,
    e2gH27
end

"""
    comsol_to_gmsh(comsol_file::String, gmsh_file::String)

Reads `comsol_file` (.mphtxt) and returns a Gmsh (.msh) file.
Warning: The Node Re-Ordering is done by hand comparing with the old comsol files from Morfe2.0. 
         So be aware that Comsol may have changed the structure or numbering of nodes.
         Here is the permutation only implemented for T6, Q9 and P18
"""
function comsol_to_gmsh(comsol_file::String, gmsh_file::String)
    (nn, n2c, e2nT6, e2gT6, e2nQ9, e2gQ9, e2nT10, e2gT10,
        e2nP18, e2gP18, e2nH27, e2gH27) = _read_mesh(comsol_file)

    gmsh.initialize()
    gmsh.model.add("mesh")

    vol_tag = 1
    dim_vol = 3
    gmsh.model.addDiscreteEntity(dim_vol, vol_tag)
    surf_tag = 2
    dim_surf = 2
    gmsh.model.addDiscreteEntity(dim_surf, surf_tag)

    # -------------------------
    # 1. NODES
    # -------------------------

    nodeTags = collect(1:nn)
    gmsh.model.mesh.addNodes(3, 1, nodeTags, n2c)

    # -------------------------
    # 2. ELEMENTS (DOMAINS Ω)
    # -------------------------
    # Comsol:T6 (tri2) ->  Gmsh:9
    elemType = 9  # second-order tetra
    # elemTags = collect(1:neT6)
    elemTags = e2gT6
    #permute
    perm_T6 = [1, 2, 3, 4, 6, 5]
    ne = length(e2nT6) ÷ 6
    for i in 1:ne
        idx = ((i - 1) * 6 + 1):(i * 6)
        elem = e2nT6[idx]
        e2nT6[idx] = elem[perm_T6]
    end
    gmsh.model.mesh.addElements(
        dim_surf,
        surf_tag,
        [elemType],
        [elemTags],
        [e2nT6]
    )

    # Comsol:Quad9 (quad2)  ->  Gmsh:10
    elemType = 10
    nn_Q9 = 9
    elemTags = e2gQ9
    #permute
    perm_Q9 = [1, 2, 4, 3, 5, 8, 9, 6, 7]
    ne = length(e2nQ9) ÷ nn_Q9
    for i in 1:ne
        idx = ((i - 1) * nn_Q9 + 1):(i * nn_Q9)
        elem = e2nQ9[idx]
        e2nQ9[idx] = elem[perm_Q9]
    end
    gmsh.model.mesh.addElements(
        dim_surf,
        surf_tag,
        [elemType],
        [elemTags],
        [e2nQ9]
    )

    # Comsol:P18 (prism2) -> Gmsh:13
    elemType = 13
    nn_P18 = 18
    elemTags = e2gP18
    #permute
    perm_P18 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 12, 15, 16, 17, 18, 11, 13, 14]
    ne = length(e2nP18) ÷ nn_P18
    for i in 1:ne
        idx = ((i - 1) * nn_P18 + 1):(i * nn_P18)
        elem = e2nP18[idx]
        e2nP18[idx] = elem[perm_P18]
    end
    gmsh.model.mesh.addElements(
        dim_vol,
        vol_tag,
        [elemType],
        [elemTags],
        [e2nP18]
    )

    # Comsol:T10 (tet2) -> Gmsh:11
    if length(e2nT10) > 0
        elemType = 11
        elemTags = e2gT10
        # COMSOL tet2: corners 3↔4 swapped and edge-mids 6↔7 swapped vs Gmsh type 11.
        # Derived from FEconv module_utils_mphtxt.f90 and verified against Gmsh API coords.
        perm_T10 = [1, 2, 4, 3, 5, 7, 6, 8, 9, 10]
        ne = length(e2nT10) ÷ T10n
        for i in 1:ne
            idx = ((i - 1) * T10n + 1):(i * T10n)
            elem = e2nT10[idx]
            e2nT10[idx] = elem[perm_T10]
        end
        gmsh.model.mesh.addElements(dim_vol, vol_tag, [elemType], [elemTags], [e2nT10])
    end

    # Comsol:H27 (hex2) -> Gmsh:12
    if length(e2nH27) > 0
        elemType = 12
        elemTags = e2gH27
        # perm_H27[j] = Legacy position corresponding to Gmsh type-12 position j.
        # Derived from actual Gmsh type-12 node coordinates vs legacy shape_functions.jl ordering.
        # Inverse of INV_PERM_H27 in GmshToComsol.jl.
        perm_H27 = [7, 5, 1, 3, 8, 6, 2, 4,
            24, 20, 27, 14, 23, 10, 9, 13, 26, 22, 16, 12,
            17, 25, 21, 15, 11, 19, 18]
        ne = length(e2nH27) ÷ H27n
        for i in 1:ne
            idx = ((i - 1) * H27n + 1):(i * H27n)
            elem = e2nH27[idx]
            e2nH27[idx] = elem[perm_H27]
        end
        gmsh.model.mesh.addElements(dim_vol, vol_tag, [elemType], [elemTags], [e2nH27])
    end

    gmsh.write(gmsh_file)
    gmsh.finalize()
end

"""
    comsol_to_gmsh_linear(comsol_file::String, gmsh_file::String)

Like `comsol_to_gmsh` but downgrades quadratic elements to their linear
counterparts before writing the Gmsh `.msh` file.
"""
function comsol_to_gmsh_linear(comsol_file::String, gmsh_file::String)
    (nn, n2c, e2nT6, e2gT6, e2nQ9, e2gQ9, e2nT10, e2gT10,
        e2nP18, e2gP18, e2nH27, e2gH27) = _read_mesh(comsol_file)

    gmsh.initialize()
    gmsh.model.add("mesh")

    vol_tag = 1
    dim_vol = 3
    gmsh.model.addDiscreteEntity(dim_vol, vol_tag)
    surf_tag = 2
    dim_surf = 2
    gmsh.model.addDiscreteEntity(dim_surf, surf_tag)

    # -------------------------
    # 1. NODES
    # -------------------------

    nodeTags = collect(1:nn)
    gmsh.model.mesh.addNodes(3, 1, nodeTags, n2c)

    # -------------------------
    # 2. ELEMENTS (DOMAINS Ω)
    # -------------------------
    # Comsol:T6 (tri2) ->  Gmsh:2
    elemType = 2  # second-order tetra
    # elemTags = collect(1:neT6)
    elemTags = e2gT6
    #permute
    perm_T6 = [1, 2, 3, 4, 6, 5]
    ne = length(e2nT6) ÷ 6
    for i in 1:ne
        idx = ((i - 1) * 6 + 1):(i * 6)
        elem = e2nT6[idx]
        e2nT6[idx] = elem[perm_T6]
    end
    e2nT3 = Vector{Int64}(undef, ne * 3)
    for i in 1:ne
        idx6 = ((i - 1) * 6 + 1):(i * 6)
        idx3 = ((i - 1) * 3 + 1):(i * 3)
        elem = e2nT6[idx6]

        e2nT3[idx3] = elem[1:3]   # keep only vertices
    end
    gmsh.model.mesh.addElements(
        dim_surf,
        surf_tag,
        [elemType],
        [elemTags],
        [e2nT3]
    )

    # Comsol:Quad9 (quad2)  ->  Gmsh:3
    elemType = 3
    nn_Q9 = 9
    nn_Q4 = 4
    elemTags = e2gQ9
    #permute
    perm_Q9 = [1, 2, 4, 3, 5, 8, 9, 6, 7]
    ne = length(e2nQ9) ÷ nn_Q9
    for i in 1:ne
        idx = ((i - 1) * nn_Q9 + 1):(i * nn_Q9)
        elem = e2nQ9[idx]
        e2nQ9[idx] = elem[perm_Q9]
    end
    e2nQ4 = Vector{Int64}(undef, ne * nn_Q4)
    for i in 1:ne
        idx9 = ((i - 1) * 9 + 1):(i * 9)
        idx4 = ((i - 1) * 4 + 1):(i * 4)
        elem = e2nQ9[idx9]

        e2nQ4[idx4] = elem[1:4]
    end
    gmsh.model.mesh.addElements(
        dim_surf,
        surf_tag,
        [elemType],
        [elemTags],
        [e2nQ4]
    )

    # Comsol:P18 (prism2) -> Gmsh:6
    elemType = 6
    nn_P18 = 18
    nn_P6 = 6
    elemTags = e2gP18
    #permute
    perm_P18 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 12, 15, 16, 17, 18, 11, 13, 14]
    ne = length(e2nP18) ÷ nn_P18
    for i in 1:ne
        idx = ((i - 1) * nn_P18 + 1):(i * nn_P18)
        elem = e2nP18[idx]
        e2nP18[idx] = elem[perm_P18]
    end

    e2nP6 = Vector{Int64}(undef, ne * nn_P6)
    for i in 1:ne
        idx18 = ((i - 1) * nn_P18 + 1):(i * nn_P18)
        idx6 = ((i - 1) * nn_P6 + 1):(i * nn_P6)
        elem = e2nP18[idx18]

        e2nP6[idx6] = elem[1:nn_P6]   # keep only vertices
    end
    gmsh.model.mesh.addElements(
        dim_vol,
        vol_tag,
        [elemType],
        [elemTags],
        [e2nP6]
    )

    # Comsol:T10 (tet2) -> Gmsh:4 (linear tet, corners only)
    if length(e2nT10) > 0
        elemType = 4
        elemTags = e2gT10
        ne = length(e2nT10) ÷ T10n
        e2nT4 = Vector{Int64}(undef, ne * 4)
        for i in 1:ne
            idx10 = ((i - 1) * T10n + 1):(i * T10n)
            idx4 = ((i - 1) * 4 + 1):(i * 4)
            e2nT4[idx4] = e2nT10[idx10][1:4]
        end
        gmsh.model.mesh.addElements(dim_vol, vol_tag, [elemType], [elemTags], [e2nT4])
    end

    # Comsol:H27 (hex2) -> Gmsh:5 (linear hex, corners only)
    if length(e2nH27) > 0
        elemType = 5
        elemTags = e2gH27
        ne = length(e2nH27) ÷ H27n
        e2nH8 = Vector{Int64}(undef, ne * 8)
        for i in 1:ne
            idx27 = ((i - 1) * H27n + 1):(i * H27n)
            idx8 = ((i - 1) * 8 + 1):(i * 8)
            e2nH8[idx8] = e2nH27[idx27][1:8]
        end
        gmsh.model.mesh.addElements(dim_vol, vol_tag, [elemType], [elemTags], [e2nH8])
    end

    gmsh.write(gmsh_file)
    gmsh.finalize()
end

end # module
