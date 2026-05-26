module FEMUtility

export abaqus_to_gmsh, abaqus_to_gmsh_linear,
       comsol_to_gmsh, comsol_to_gmsh_linear,
       gmsh_to_comsol

function abaqus_to_gmsh(args...; kwargs...)
    error("abaqus_to_gmsh requires Gmsh.jl.\nLoad it with `using Gmsh` to activate the MORFE extension.")
end

function abaqus_to_gmsh_linear(args...; kwargs...)
    error("abaqus_to_gmsh_linear requires Gmsh.jl.\nLoad it with `using Gmsh` to activate the MORFE extension.")
end

function comsol_to_gmsh(args...; kwargs...)
    error("comsol_to_gmsh requires Gmsh.jl.\nLoad it with `using Gmsh` to activate the MORFE extension.")
end

function comsol_to_gmsh_linear(args...; kwargs...)
    error("comsol_to_gmsh_linear requires Gmsh.jl.\nLoad it with `using Gmsh` to activate the MORFE extension.")
end

function gmsh_to_comsol(args...; kwargs...)
    error("gmsh_to_comsol requires Gmsh.jl.\nLoad it with `using Gmsh` to activate the MORFE extension.")
end

end # module FEMUtility
