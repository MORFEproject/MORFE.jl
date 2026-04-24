module AbaqusToGmsh

using Gmsh

export abaqus_to_gmsh, abaqus_to_gmsh_linear

# ---------------------------------------------------------------------------
# Node-ordering permutations: Abaqus → Gmsh
#
# Convention: perm[i] = j means Gmsh node i gets Abaqus node j.
# Applied as:  gmsh_nodes = abaqus_nodes[perm]
#
# Derived by matching reference-element parametric coordinates queried from
# gmsh.model.mesh.getElementProperties against the Abaqus documentation.
# ---------------------------------------------------------------------------

# C3D10 → Gmsh type 11 (Tet10):
#   Abaqus[9]=mid(v2,v4), Abaqus[10]=mid(v3,v4)
#   Gmsh[9] =mid(v3,v4),  Gmsh[10] =mid(v2,v4)  →  swap 9↔10
const PERM_C3D10 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 9]

# C3D20 → Gmsh type 17 (Hex20):
#   Edge-midpoint traversal order differs completely.
#   Derived by coordinate matching.
const PERM_C3D20 = [1, 2, 3, 4, 5, 6, 7, 8,
                    9, 12, 17, 10, 18, 11, 19, 20,
                    13, 16, 14, 15]

# C3D15 → Gmsh type 18 (Wedge15):
#   Bottom triangle mids, vertical mids, and top triangle mids are in
#   different order.  Derived by coordinate matching.
const PERM_C3D15 = [1, 2, 3, 4, 5, 6, 7, 9, 13, 8, 14, 15, 10, 12, 11]

# B32 / T3D3 → Gmsh type 8 (3-node line):
#   Abaqus: [end1, midpoint, end2]
#   Gmsh:   [end1, end2, midpoint]
const PERM_B32 = [1, 3, 2]

# ---------------------------------------------------------------------------
# Element-type registry
# (gmsh_type, n_corners, npe, perm_or_nothing)
# ---------------------------------------------------------------------------
const ELEM_INFO = Dict{String, Tuple{Int,Int,Int,Union{Nothing,Vector{Int}}}}(
    # 3-D solids
    "C3D4"   => (4,  4,  4,  nothing),
    "C3D10"  => (11, 4,  10, PERM_C3D10),
    "C3D10M" => (11, 4,  10, PERM_C3D10),
    "C3D8"   => (5,  8,  8,  nothing),
    "C3D8R"  => (5,  8,  8,  nothing),
    "C3D8I"  => (5,  8,  8,  nothing),
    "C3D20"  => (17, 8,  20, PERM_C3D20),
    "C3D20R" => (17, 8,  20, PERM_C3D20),
    "C3D6"   => (6,  6,  6,  nothing),
    "C3D15"  => (18, 6,  15, PERM_C3D15),
    # Shells
    "S3"     => (2,  3,  3,  nothing),
    "S3R"    => (2,  3,  3,  nothing),
    "S4"     => (3,  4,  4,  nothing),
    "S4R"    => (3,  4,  4,  nothing),
    "S6"     => (9,  3,  6,  nothing),
    "S8"     => (16, 4,  8,  nothing),
    "S8R"    => (16, 4,  8,  nothing),
    # Plane stress / plane strain
    "CPS3"   => (2,  3,  3,  nothing),   "CPE3"   => (2,  3,  3,  nothing),
    "CPS4"   => (3,  4,  4,  nothing),   "CPE4"   => (3,  4,  4,  nothing),
    "CPS4R"  => (3,  4,  4,  nothing),   "CPE4R"  => (3,  4,  4,  nothing),
    "CPS6"   => (9,  3,  6,  nothing),   "CPE6"   => (9,  3,  6,  nothing),
    "CPS8"   => (16, 4,  8,  nothing),   "CPE8"   => (16, 4,  8,  nothing),
    # Axisymmetric continuum
    "CAX3"   => (2,  3,  3,  nothing),
    "CAX4"   => (3,  4,  4,  nothing),   "CAX4R"  => (3,  4,  4,  nothing),
    "CAX6"   => (9,  3,  6,  nothing),
    "CAX8"   => (16, 4,  8,  nothing),   "CAX8R"  => (16, 4,  8,  nothing),
    # Beams and trusses
    "B31"    => (1,  2,  2,  nothing),
    "B32"    => (8,  2,  3,  PERM_B32),
    "T3D2"   => (1,  2,  2,  nothing),
    "T3D3"   => (8,  2,  3,  PERM_B32),
)

# Gmsh element type → geometric dimension
const GMSH_DIM = Dict{Int,Int}(
    1 => 1, 8 => 1,
    2 => 2, 3 => 2, 9 => 2, 16 => 2,
    4 => 3, 5 => 3, 6 => 3, 11 => 3, 17 => 3, 18 => 3,
)

# Quadratic gmsh_type → (linear_gmsh_type, n_corner_nodes)
const TO_LINEAR = Dict{Int,Tuple{Int,Int}}(
    8  => (1, 2),
    9  => (2, 3),
    16 => (3, 4),
    11 => (4, 4),
    17 => (5, 8),
    18 => (6, 6),
)

# ---------------------------------------------------------------------------
# Internal data structure for one *Element section
# ---------------------------------------------------------------------------
struct _ElemSection
    abaqus_type :: String
    elset       :: String
    gmsh_type   :: Int
    n_corners   :: Int
    npe         :: Int                       # nodes per element (Abaqus count)
    perm        :: Union{Nothing,Vector{Int}}
    elem_ids    :: Vector{Int64}
    e2n         :: Vector{Int64}             # flat, length = ne*npe, Abaqus IDs
end

# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

function _parse_keyword(line::String)
    parts = split(line, ",")
    kw = uppercase(strip(parts[1]))
    opts = Dict{String,String}()
    for p in parts[2:end]
        kv = split(strip(p), "="; limit=2)
        if length(kv) == 2
            opts[uppercase(strip(kv[1]))] = strip(kv[2])
        end
    end
    return kw, opts
end

# Read one logical Abaqus data record, following trailing-comma continuations.
function _read_tokens(io::IO, first_line::String)
    tokens = String[]
    line = first_line
    while true
        stripped = strip(line)
        has_cont = endswith(stripped, ",")
        parts = split(stripped, ",")
        limit = has_cont ? length(parts) - 1 : length(parts)
        for i in 1:limit
            t = strip(parts[i])
            !isempty(t) && push!(tokens, t)
        end
        has_cont || break
        eof(io) && break
        line = readline(io)
    end
    return tokens
end

# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------
function _read_abaqus(inp_file::String)
    node_ids    = Int64[]
    node_coords = Float64[]   # flat [x,y,z, x,y,z, ...]
    sections    = _ElemSection[]

    # Mutable parser state (Ref-boxed so the flush! closure can rebind them)
    cur_type  = Ref{String}("")
    cur_elset = Ref{String}("")
    cur_gt    = Ref{Int}(0)
    cur_nc    = Ref{Int}(0)
    cur_npe   = Ref{Int}(0)
    cur_perm  = Ref{Union{Nothing,Vector{Int}}}(nothing)
    cur_ids   = Int64[]
    cur_e2n   = Int64[]

    function flush!()
        if !isempty(cur_type[]) && !isempty(cur_ids)
            push!(sections, _ElemSection(
                cur_type[], cur_elset[], cur_gt[], cur_nc[], cur_npe[], cur_perm[],
                copy(cur_ids), copy(cur_e2n)
            ))
        end
        cur_type[]  = ""
        cur_elset[] = ""
        empty!(cur_ids)
        empty!(cur_e2n)
    end

    mode = Ref{Symbol}(:none)   # :nodes | :elements | :none

    open(inp_file, "r") do io
        while !eof(io)
            line = strip(readline(io))
            isempty(line)            && continue
            startswith(line, "**")   && continue   # Abaqus comment

            if startswith(line, "*")
                flush!()
                mode[] = :none
                kw, opts = _parse_keyword(line)

                if kw == "*NODE"
                    mode[] = :nodes

                elseif kw == "*ELEMENT"
                    atype = uppercase(get(opts, "TYPE", ""))
                    if haskey(ELEM_INFO, atype)
                        (cur_gt[], cur_nc[], cur_npe[], cur_perm[]) = ELEM_INFO[atype]
                        cur_type[]  = atype
                        cur_elset[] = get(opts, "ELSET", "")
                        mode[]      = :elements
                    end
                end
                continue
            end

            if mode[] == :nodes
                tokens = _read_tokens(io, line)
                length(tokens) < 3 && continue
                push!(node_ids, parse(Int64, tokens[1]))
                push!(node_coords, parse(Float64, tokens[2]))
                push!(node_coords, parse(Float64, tokens[3]))
                # 2-D meshes supply only x,y; pad z=0
                push!(node_coords, length(tokens) >= 4 ? parse(Float64, tokens[4]) : 0.0)

            elseif mode[] == :elements
                tokens = _read_tokens(io, line)
                npe = cur_npe[]
                length(tokens) < 1 + npe && continue
                push!(cur_ids, parse(Int64, tokens[1]))
                for k in 2:(1 + npe)
                    push!(cur_e2n, parse(Int64, tokens[k]))
                end
            end
        end
        flush!()
    end

    return node_ids, node_coords, sections
end

# ---------------------------------------------------------------------------
# Shared Gmsh model builder
# ---------------------------------------------------------------------------
function _build_and_write(node_ids, node_coords, sections,
                          linear::Bool, gmsh_file::String)
    # Compact, sorted Gmsh node tags (Abaqus IDs may be non-contiguous)
    perm = sortperm(node_ids)
    sorted_ids = node_ids[perm]
    nn = length(sorted_ids)

    flat_coords = Vector{Float64}(undef, nn * 3)
    for (new_i, old_i) in enumerate(perm)
        flat_coords[3new_i-2:3new_i] = node_coords[3old_i-2:3old_i]
    end

    id_to_tag = Dict{Int64,Int64}(sorted_ids[i] => Int64(i) for i in 1:nn)

    # Optionally downgrade quadratic → linear
    eff_sections = linear ? map(sections) do s
        gt_lin, nc = get(TO_LINEAR, s.gmsh_type, (s.gmsh_type, s.n_corners))
        _ElemSection(s.abaqus_type, s.elset, gt_lin, nc, s.npe, s.perm,
                     s.elem_ids, s.e2n)
    end : sections

    dims_present = Set{Int}(GMSH_DIM[s.gmsh_type] for s in eff_sections)
    max_dim = isempty(dims_present) ? 3 : maximum(dims_present)

    gmsh.initialize()
    gmsh.model.add("mesh")
    for d in sort(collect(dims_present))
        gmsh.model.addDiscreteEntity(d, d)
    end

    node_tags = collect(Int64(1):Int64(nn))
    gmsh.model.mesh.addNodes(max_dim, max_dim, node_tags, flat_coords)

    # Merge sections of the same effective gmsh_type, then add elements
    type_map = Dict{Int, Tuple{Vector{Int64}, Vector{Int64}}}()

    for s in eff_sections
        gt  = s.gmsh_type
        npe = s.npe        # original (quadratic) npe — for striding e2n
        nc  = s.n_corners  # nodes to emit (nc == npe when not downgrading)
        ne  = length(s.elem_ids)

        if !haskey(type_map, gt)
            type_map[gt] = (Int64[], Int64[])
        end
        ids_acc, e2n_acc = type_map[gt]

        append!(ids_acc, Int64.(s.elem_ids))

        limit = linear ? nc : npe
        for i in 1:ne
            raw = s.e2n[(i-1)*npe+1 : i*npe]
            ordered = isnothing(s.perm) ? raw : raw[s.perm]
            for k in 1:limit
                push!(e2n_acc, id_to_tag[Int64(ordered[k])])
            end
        end
    end

    for (gt, (ids, e2n)) in type_map
        d = GMSH_DIM[gt]
        gmsh.model.mesh.addElements(d, d, [gt], [ids], [e2n])
    end

    gmsh.write(gmsh_file)
    gmsh.finalize()
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    abaqus_to_gmsh(inp_file::String, gmsh_file::String)

Read an Abaqus `.inp` mesh and write a Gmsh `.msh` file.

Supported element types: C3D4/10/10M, C3D8/8R/8I, C3D20/20R, C3D6, C3D15,
S3/3R, S4/4R, S6, S8/8R, CPS3/4/4R/6/8, CPE3/4/4R/6/8, CAX3/4/4R/6/8/8R,
B31, B32, T3D2, T3D3.

Node IDs may be non-contiguous; they are remapped to a compact 1-based Gmsh
tag space. Unknown element types are silently skipped.
"""
function abaqus_to_gmsh(inp_file::String, gmsh_file::String)
    node_ids, node_coords, sections = _read_abaqus(inp_file)
    _build_and_write(node_ids, node_coords, sections, false, gmsh_file)
end

"""
    abaqus_to_gmsh_linear(inp_file::String, gmsh_file::String)

Like `abaqus_to_gmsh` but downgrades quadratic elements to their linear
(corner-nodes-only) counterparts before writing.
"""
function abaqus_to_gmsh_linear(inp_file::String, gmsh_file::String)
    node_ids, node_coords, sections = _read_abaqus(inp_file)
    _build_and_write(node_ids, node_coords, sections, true, gmsh_file)
end

end # module
