"""
MORFE.jl demo — 08_mems_micromirror
Single-crystal silicon MEMS mirror-gimbal, Gmsh mesh, St. Venant-Kirchhoff.

Prerequisite: `mirror_gimbal.msh`, produced by `mesh.jl` from a STEP export of
`Mirror-Gimbal-v1.2.SLDPRT` (see README.md).

Run in the folder:  julia --project main.jl
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
    Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
    Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays", "Gmsh"])
end
Pkg.instantiate()

using MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps
SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)

mesh = joinpath(@__DIR__, "mirror_gimbal.msh")
isfile(mesh) || error("Missing $mesh — run `julia --project mesh.jl` first (see README.md).")

mirror = SVK.mechanical_model(
    mesh;
    # Single-crystal silicon, isotropic-equivalent, µm-consistent units (as example 03).
    material = SVK.SVKMaterial(E = 170e3, ν = 0.28, ρ = 2.33e-3),
    damping = SVK.RayleighDamping(
        α = 0.0,
        β = 0.0,
    ),
    dirichlet = "clamp",   # named facetset written by mesh.jl (anchor pads)
    fe_order = 2, quad_order = 3,
)

rom = SVK.parametrise(mirror; master = [1], order = 5)

# Near-resonant harmonic forcing — uncomment to add two external states:
# rom = SVK.parametrise(mirror; master = [1], order = 5,
#     forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.03))

SVK.print_equations(rom)
SVK.save_rom(rom, joinpath(@__DIR__, "results"))
println("\nResults written to $(joinpath(@__DIR__, "results"))")
