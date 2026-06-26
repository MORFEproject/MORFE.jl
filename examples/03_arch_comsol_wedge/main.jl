"""
MORFE.jl demo — 03_arch_comsol_wedge
Isotropic polysilicon arch, COMSOL P18 wedge mesh, St. Venant-Kirchhoff.

Run in the folder:  julia --project main.jl
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
	Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays"])
end
Pkg.instantiate()

using MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps
SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)

arch = SVK.mechanical_model(
	joinpath(@__DIR__, "arch_2_force.mphtxt");
	material = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3),
	damping = SVK.RayleighDamping(
		α = 0.0,
		β = 0.0,
	),
	dirichlet = Set([1, 11]),   # COMSOL entity IDs for the two arch feet (raw IDs 0 and 10)
	fe_order = 2, quad_order = 4,
)

rom = SVK.parametrise(arch; master = [1], order = 5)

# Near-resonant harmonic forcing — uncomment to add two external states:
# rom = SVK.parametrise(arch; master = [1], order = 5,
#     forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.03))

SVK.print_equations(rom)
SVK.save_rom(rom, joinpath(@__DIR__, "results"))
println("\nResults written to $(joinpath(@__DIR__, "results"))")
