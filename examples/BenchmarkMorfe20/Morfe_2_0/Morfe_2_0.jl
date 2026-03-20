module Morfe_2_0

using SparseArrays: SparseMatrixCSC
using ExtendableSparse: SparseMatrixLNK
using FEMQuad

include("defs.jl")
include("shape_functions.jl")
include("materials.jl")
include("mesh.jl")
include("field.jl")
include("assembler.jl")
include("assembler_dummy.jl")
include("elemental.jl")

export Field, Grid, material, MORFE_newmaterial, read_mesh, Infostruct
export assembler_dummy_MK, assembler_MK!, assembly_G!, assembly_H!

end # module