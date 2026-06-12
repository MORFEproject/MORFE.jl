"""
    SVKMaterial(; E, ν, ρ)

St. Venant-Kirchhoff material: Young's modulus `E`, Poisson ratio `ν`,
density `ρ`. Lamé constants `λ`, `μ` are derived. The SVK model generates
quadratic and cubic nonlinearities in the displacement — nothing higher.
"""
struct SVKMaterial{T}
    E::T
    ν::T
    ρ::T
    λ::T
    μ::T
end
function SVKMaterial(; E, ν, ρ)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    return SVKMaterial(promote(E, ν, ρ, λ, μ)...)
end

"""
    RayleighDamping(; α, β)

Rayleigh damping `C = α M + β K`.
"""
struct RayleighDamping{T}
    α::T
    β::T
end
RayleighDamping(; α, β) = RayleighDamping(promote(α, β)...)

"""
    HarmonicForcing(; mode, amplitude, Ω = nothing)

Harmonic load `f(t) = amplitude · M·ϕ_mode · cos(Ω t)`: shaped like mode
`mode`, oscillating at that same mode's natural frequency unless `Ω` is given.
`mode` must be one of the `master` mode pairs passed to `parametrise`.
"""
struct HarmonicForcing{T}
    mode::Int
    amplitude::T
    Ω::Union{Nothing, T}
end
HarmonicForcing(; mode, amplitude, Ω = nothing) =
    HarmonicForcing(mode, amplitude, Ω === nothing ? nothing : convert(typeof(amplitude), Ω))

"""
Assembled second-order mechanical model `M ü + C u̇ + K u = f_nl(u) (+ forcing)`,
restricted to free DOFs, with a lazy factory for the FEM nonlinear terms:
`term_factory(degree, max_cols)` → `FEMMultilinearMap`.
"""
struct AssembledMechanicalModel{TK, TM, TC, F, MAT, DMP}
    K::TK
    M::TM
    C::TC
    term_factory::F
    nonlinear_degrees::Tuple{Vararg{Int}}
    material::MAT
    damping::DMP
    info::NamedTuple
end

function Base.show(io::IO, ::MIME"text/plain", m::AssembledMechanicalModel)
    println(io, "AssembledMechanicalModel ($(m.info.backend))")
    println(io, "  free DOFs : $(m.info.n_dofs) (of $(m.info.n_dofs_total))")
    println(io,
        "  material  : SVK  E=$(m.material.E)  ν=$(m.material.ν)  ρ=$(m.material.ρ)")
    print(io, "  damping   : Rayleigh  α=$(m.damping.α)  β=$(m.damping.β)")
end

"""
Invariant-manifold ROM returned by `parametrise`. Fields: `W` (parametrisation),
`R` (reduced dynamics), `eigenvalues`, `master` (mode pairs), `order`,
`forcing` (`nothing` or `HarmonicForcing`), `info` (NamedTuple).
"""
struct InvariantManifoldROM{TW, TR, T, FRC}
    W::TW
    R::TR
    eigenvalues::Vector{Complex{T}}
    master::Vector{Int}
    order::Int
    forcing::FRC
    info::NamedTuple
end

function Base.show(io::IO, ::MIME"text/plain", rom::InvariantManifoldROM)
    ROM = 2 * length(rom.master)
    println(io, "InvariantManifoldROM")
    println(io, "  master pairs : $(rom.master)  (ROM = $ROM, N_EXT = $(rom.info.N_EXT))")
    for p in rom.master
        λ = rom.eigenvalues[2p - 1]
        println(io, "    pair $p: λ = $λ   (f = $(abs(λ) / 2π))")
    end
    if rom.forcing === nothing
        println(io, "  forcing      : none (autonomous)")
    else
        println(io,
            "  forcing      : mode $(rom.forcing.mode), amplitude $(rom.forcing.amplitude), Ω = $(rom.info.Ω)")
    end
    println(io, "  order        : $(rom.order)   ($(rom.info.n_monomials) monomials)")
    print(io, "  solve time   : $(round(rom.info.solve_time_s; digits = 2)) s")
end
