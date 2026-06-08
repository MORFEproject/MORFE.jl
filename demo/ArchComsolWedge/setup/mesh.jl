"""
FEM mesh setup for the arch_2_force COMSOL mesh.

Provides:
  _read_mesh(path)    — minimal .mphtxt reader (no Gmsh dependency)
  load_arch_mesh(path) → (grid, constrained_node_set)

Cell type: QuadraticWedge (18-node P18 prism, Ferrite RefPrism).
Dirichlet nodes: the two arch-foot Q9 faces (COMSOL entity IDs 0 and 10 → 1 and 11 in Julia).
"""

using Ferrite

# -----------------------------------------------------------------------
# .mphtxt reader
# -----------------------------------------------------------------------

const _TR6n = 6; const _T10n = 10; const _P18n = 18; const _QU9n = 9; const _H27n = 27

function _read_mesh(mesh_file::String)
    neT6 = 0;  e2nT6 = Int64[];  e2gT6 = Int64[]
    neQ9 = 0;  e2nQ9 = Int64[];  e2gQ9 = Int64[]
    e2nT10 = Int64[]; e2gT10 = Int64[]
    neP18 = 0; e2nP18 = Int64[]; e2gP18 = Int64[]
    e2nH27 = Int64[]; e2gH27 = Int64[]
    nn = 0;    n2c = Float64[];   dim = 3
    open(mesh_file, "r") do fhand
        while !eof(fhand)
            line = readline(fhand)
            if contains(line, "# number of mesh vertices")
                nn = Meta.parse(split(line)[1]); n2c = Vector{Float64}(undef, nn * dim)
            elseif contains(line, "# Mesh vertex coordinates")
                for i in 1:nn; parts = split(readline(fhand))
                    for j in 1:dim; n2c[j + (i-1)*dim] = Meta.parse(parts[j]); end; end
            elseif contains(line, "tri2 # type name")
                readline(fhand); readline(fhand); readline(fhand)
                neT6 = Meta.parse(split(readline(fhand))[1]); readline(fhand)
                e2nT6 = Vector{Int64}(undef, neT6*_TR6n); e2gT6 = Vector{Int64}(undef, neT6)
                for i in 1:neT6; parts = split(readline(fhand))
                    for j in 1:_TR6n; e2nT6[j+(i-1)*_TR6n] = Meta.parse(parts[j])+1; end; end
                while true; l = readline(fhand)
                    if contains(l, "# Geometric entity indices")
                        for i in 1:neT6; e2gT6[i] = Meta.parse(readline(fhand))+1; end; break; end; end
            elseif contains(line, "quad2 # type name")
                readline(fhand); readline(fhand); readline(fhand)
                neQ9 = Meta.parse(split(readline(fhand))[1]); readline(fhand)
                e2nQ9 = Vector{Int64}(undef, neQ9*_QU9n); e2gQ9 = Vector{Int64}(undef, neQ9)
                for i in 1:neQ9; parts = split(readline(fhand))
                    for j in 1:_QU9n; e2nQ9[j+(i-1)*_QU9n] = Meta.parse(parts[j])+1; end; end
                while true; l = readline(fhand)
                    if contains(l, "# Geometric entity indices")
                        for i in 1:neQ9; e2gQ9[i] = Meta.parse(readline(fhand))+1; end; break; end; end
            elseif contains(line, "prism2 # type name")
                readline(fhand); readline(fhand); readline(fhand)
                neP18 = Meta.parse(split(readline(fhand))[1]); readline(fhand)
                e2nP18 = Vector{Int64}(undef, neP18*_P18n); e2gP18 = Vector{Int64}(undef, neP18)
                for i in 1:neP18; parts = split(readline(fhand))
                    for j in 1:_P18n; e2nP18[j+(i-1)*_P18n] = Meta.parse(parts[j])+1; end; end
                while true; l = readline(fhand)
                    if contains(l, "# Geometric entity indices")
                        for i in 1:neP18; e2gP18[i] = Meta.parse(readline(fhand)); end; break; end; end
            end
        end
    end
    return nn, n2c, e2nT6, e2gT6, e2nQ9, e2gQ9, e2nT10, e2gT10, e2nP18, e2gP18, e2nH27, e2gH27
end

# -----------------------------------------------------------------------
# QuadraticWedge cell type
# -----------------------------------------------------------------------

struct QuadraticWedge <: Ferrite.AbstractCell{Ferrite.RefPrism}
    nodes::NTuple{18,Int}
end
Ferrite.default_interpolation(::Type{QuadraticWedge})  = Ferrite.Lagrange{Ferrite.RefPrism,2}()
Ferrite.geometric_interpolation(::Type{QuadraticWedge}) = Ferrite.Lagrange{Ferrite.RefPrism,2}()

# COMSOL prism2 → Ferrite RefPrism node permutation (from ComsolToGmsh perm_P18)
const PERM_P18 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 12, 15, 16, 17, 18, 11, 13, 14]

# COMSOL geometric entity IDs (1-indexed after +1 offset) for the arch feet.
# Raw file IDs 0 (left foot) and 10 (right foot) → Julia IDs 1 and 11.
# Source: arch_2_force.jl legacy config, boundaries_list_ = [[1, 11]].
const ARCH_DIRICHLET_BND_IDS = Set{Int}([1, 11])

# -----------------------------------------------------------------------
# load_arch_mesh
# -----------------------------------------------------------------------

"""
    load_arch_mesh(path) -> (grid::Ferrite.Grid, constrained::Set{Int})

Read a COMSOL .mphtxt file and return the Ferrite P18-prism grid together
with the set of node indices lying on the Dirichlet (anchor) boundaries.
"""
function load_arch_mesh(mphtxt_path::AbstractString)
    (nn, n2c,
     e2nT6, e2gT6, e2nQ9, e2gQ9,
     _, _, e2nP18, _, _, _) = _read_mesh(mphtxt_path)

    nodes = [Ferrite.Node(Ferrite.Vec{3,Float64}((n2c[3i-2], n2c[3i-1], n2c[3i]))) for i in 1:nn]

    neP18 = length(e2nP18) ÷ 18
    cells = [QuadraticWedge(ntuple(j -> e2nP18[(i-1)*18 + PERM_P18[j]], Val(18))) for i in 1:neP18]

    grid = Ferrite.Grid(cells, nodes)

    constrained = Set{Int}()
    for i in 1:(length(e2nT6) ÷ 6)
        e2gT6[i] ∈ ARCH_DIRICHLET_BND_IDS || continue
        for j in 1:6; push!(constrained, e2nT6[(i-1)*6 + j]); end
    end
    for i in 1:(length(e2nQ9) ÷ 9)
        e2gQ9[i] ∈ ARCH_DIRICHLET_BND_IDS || continue
        for j in 1:9; push!(constrained, e2nQ9[(i-1)*9 + j]); end
    end
    return grid, constrained
end
