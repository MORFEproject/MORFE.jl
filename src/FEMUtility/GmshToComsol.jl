module GmshToComsol

using Gmsh
using Printf

export gmsh_to_comsol

# Inverse permutations: undo the Comsol→Gmsh reorderings from comsol_to_gmsh.
# Derived by computing inv_perm[i] = j s.t. perm[j] = i.
const INV_PERM_T6  = [1, 2, 3, 4, 6, 5]
const INV_PERM_Q9  = [1, 2, 4, 3, 5, 8, 9, 6, 7]
const INV_PERM_P18 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 16, 11, 17, 18, 12, 13, 14, 15]
# T10: inverse of perm [1,2,3,4,5,6,9,8,10,7]
const INV_PERM_T10 = [1, 2, 3, 4, 5, 6, 10, 8, 7, 9]
# H27: face-center swap is its own inverse
const INV_PERM_H27 = [1, 2, 3, 4, 5, 6, 7, 8,
                      9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      25, 26, 23, 24, 21, 22,
                      27]

"""
    gmsh_to_comsol(gmsh_file::String, comsol_file::String)

Reads `gmsh_file` (.msh) and writes a COMSOL (.mphtxt) mesh file.
This is the inverse of `comsol_to_gmsh`: it undoes the node-reordering
permutations and converts 1-based Gmsh indexing back to 0-based COMSOL
indexing.

Assumptions:
- The .msh file was produced by `comsol_to_gmsh` (node tags 1:nn).
- Element tags in the Gmsh file carry the COMSOL geometric-entity indices
  (as written by `comsol_to_gmsh`).
"""
function gmsh_to_comsol(gmsh_file::String, comsol_file::String)
    gmsh.initialize()
    gmsh.open(gmsh_file)

    # -------------------------
    # 1. NODES
    # -------------------------
    nodeTags, coords, _ = gmsh.model.mesh.getNodes(-1, -1)
    nn = length(nodeTags)

    # Sort nodes by tag for a canonical ordering.
    perm = sortperm(nodeTags)
    nodeTags_sorted = nodeTags[perm]

    coords_sorted = Vector{Float64}(undef, nn * 3)
    for (new_i, old_i) in enumerate(perm)
        coords_sorted[3new_i-2:3new_i] = coords[3old_i-2:3old_i]
    end

    # Gmsh node tag → 0-based COMSOL index (COMSOL is 0-indexed).
    tag_to_comsol = Dict{Int64,Int64}(
        Int64(tag) => (i - 1) for (i, tag) in enumerate(nodeTags_sorted)
    )

    # -------------------------
    # 2. ELEMENTS
    # -------------------------
    # Gmsh type 9  = T6  (6-node quadratic triangle)
    # Gmsh type 10 = Q9  (9-node quadratic quadrangle)
    # Gmsh type 13 = P18 (18-node quadratic prism)
    # Gmsh type 11 = T10 (10-node quadratic tetrahedron)
    # Gmsh type 12 = H27 (27-node quadratic hexahedron)
    elemTags_T6,  nodeFlat_T6  = gmsh.model.mesh.getElementsByType(9)
    elemTags_Q9,  nodeFlat_Q9  = gmsh.model.mesh.getElementsByType(10)
    elemTags_P18, nodeFlat_P18 = gmsh.model.mesh.getElementsByType(13)
    elemTags_T10, nodeFlat_T10 = gmsh.model.mesh.getElementsByType(11)
    elemTags_H27, nodeFlat_H27 = gmsh.model.mesh.getElementsByType(12)

    gmsh.finalize()

    # -------------------------
    # 3. INVERSE PERMUTATIONS
    # -------------------------
    # Apply inv_perm to recover COMSOL node ordering, then convert to 0-based.
    function reorder(nodeFlat, npe, inv_perm)
        ne = length(nodeFlat) ÷ npe
        e2n = Vector{Int64}(undef, ne * npe)
        for i in 1:ne
            base = (i - 1) * npe
            for (j, p) in enumerate(inv_perm)
                e2n[base+j] = tag_to_comsol[Int64(nodeFlat[base+p])]
            end
        end
        return e2n, ne
    end

    e2nT6_c,  neT6  = reorder(nodeFlat_T6,  6,  INV_PERM_T6)
    e2nQ9_c,  neQ9  = reorder(nodeFlat_Q9,  9,  INV_PERM_Q9)
    e2nP18_c, neP18 = reorder(nodeFlat_P18, 18, INV_PERM_P18)
    e2nT10_c, neT10 = reorder(nodeFlat_T10, 10, INV_PERM_T10)
    e2nH27_c, neH27 = reorder(nodeFlat_H27, 27, INV_PERM_H27)

    # Geometric entity indices.
    # T6/Q9: reader applied +1, so subtract 1 to recover 0-based COMSOL index.
    # P18/T10/H27: reader stored as-is (no +1), so use element tag directly.
    e2gT6_c  = [Int64(t) - 1 for t in elemTags_T6]
    e2gQ9_c  = [Int64(t) - 1 for t in elemTags_Q9]
    e2gP18_c = [Int64(t)     for t in elemTags_P18]
    e2gT10_c = [Int64(t)     for t in elemTags_T10]
    e2gH27_c = [Int64(t)     for t in elemTags_H27]

    # -------------------------
    # 4. WRITE COMSOL FILE
    # -------------------------
    ntypes = (neT6  > 0 ? 1 : 0) + (neQ9  > 0 ? 1 : 0) +
             (neT10 > 0 ? 1 : 0) + (neP18 > 0 ? 1 : 0) + (neH27 > 0 ? 1 : 0)

    open(comsol_file, "w") do io
        _write_header(io, nn)
        for i in 1:nn
            @printf(io, "%.15g %.15g %.15g\n",
                coords_sorted[3i-2], coords_sorted[3i-1], coords_sorted[3i])
        end
        println(io)
        println(io, "$ntypes # number of element types")

        type_idx = 0
        if neT6 > 0
            _write_elem_section(io, type_idx, "tri2", 6, neT6, e2nT6_c, e2gT6_c)
            type_idx += 1
        end
        if neQ9 > 0
            _write_elem_section(io, type_idx, "quad2", 9, neQ9, e2nQ9_c, e2gQ9_c)
            type_idx += 1
        end
        if neT10 > 0
            _write_elem_section(io, type_idx, "tet2", 10, neT10, e2nT10_c, e2gT10_c)
            type_idx += 1
        end
        if neP18 > 0
            _write_elem_section(io, type_idx, "prism2", 18, neP18, e2nP18_c, e2gP18_c)
            type_idx += 1
        end
        if neH27 > 0
            _write_elem_section(io, type_idx, "hex2", 27, neH27, e2nH27_c, e2gH27_c)
            type_idx += 1
        end
    end
end

function _write_header(io::IO, nn::Int)
    println(io, "# Created by MORFE_jl.")
    println(io)
    println(io, "# Major & minor version")
    println(io, "0 1")
    println(io)
    println(io, "1 # number of tags")
    println(io, "# Tags")
    println(io, "5 mesh1")
    println(io)
    println(io, "1 # number of types")
    println(io, "# Types")
    println(io, "3 obj")
    println(io)
    println(io)
    println(io, "# --------- Object 0 ----------")
    println(io)
    println(io)
    println(io, "0 1 # version")
    println(io, "4 Mesh # class")
    println(io, "4 # version")
    println(io, "3 # sdim")
    println(io, "$nn # number of mesh vertices")
    println(io, "0 # lowest mesh vertex index")
    println(io)
    println(io, "# Mesh vertex coordinates")
end

# Write one element-type block in the format expected by _read_mesh in ComsolToGmsh.jl.
# After the type-name line the reader skips 3 lines, parses the 4th as element count,
# then skips 1 more line before reading element connectivity.
function _write_elem_section(
    io::IO, type_idx::Int, type_name::String,
    npe::Int, ne::Int, e2n::Vector{Int64}, e2g::Vector{Int64}
)
    println(io)
    println(io, "# --------- Type $type_idx ----------")
    println(io)
    println(io, "$type_name # type name")
    println(io, "# number of nodes per element")   # skip 1
    println(io, "$npe")                             # skip 2
    println(io, "# number of elements")             # skip 3
    println(io, "$ne")                              # parsed as element count
    println(io, "# Elements")                       # skip 5
    for i in 1:ne
        idx = ((i - 1) * npe + 1):(i * npe)
        println(io, join(e2n[idx], " "))
    end
    println(io, "# number of parameter values per element")
    println(io, "0")
    println(io, "# Geometric entity indices")
    for g in e2g
        println(io, "$g")
    end
end

end # module
