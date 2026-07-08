using FerriteGmsh
using Arpack
using LinearMaps

_SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)

function load_arch_mesh(mesh_path::AbstractString)
    return _SVK.load_comsol_grid(mesh_path, Set([1, 11]))
end
