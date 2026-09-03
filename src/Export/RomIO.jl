"""
Module `RomIO` — backend-agnostic persistence for computed ROMs.

`save_rom` writes the standard result layout every MORFE workflow shares:

	dir/
	  data/
	    W.jls                 — serialised `Parametrisation`
	    R.jls                 — serialised `ReducedDynamics`
	    R_coefficients.csv    — complex reduced dynamics, one monomial per row
	  figures/                — created empty, for downstream plots
	  summary.txt             — caller metadata + julia version, git commit, timestamp

The CSV schema is `exp_1,…,exp_NVAR,R1_re,R1_im,…` with rows whose
coefficients are all below `drop_below` omitted. `read_rom_coefficients`
parses that schema back; it is the entry point every validation script uses.

`normal_form_branch` is the other direction: it turns a solved `ReducedDynamics` into the
periodic-orbit data a plot needs, without depending on any plotting package.
"""
module RomIO

using LinearAlgebra: eigvals
using Serialization
using ..ParametrisationMethod: Parametrisation, ReducedDynamics, coefficients,
                               multiindex_set
using ..ExternalSystems: external_basis

export save_rom, read_rom_coefficients, write_rom_coefficients_csv, normal_form_branch

"""
	write_rom_coefficients_csv(path, exponents, coefficients; drop_below = 1e-14)

Write the standard `R_coefficients.csv`. `exponents` is the vector of
multiindex exponents (length L); `coefficients` is the `(NR, L)` complex
matrix. Rows with all `|c| ≤ drop_below` are omitted.
"""
function write_rom_coefficients_csv(path::AbstractString, exponents,
        coefficients::AbstractMatrix; drop_below::Real = 1e-14)
    NR = size(coefficients, 1)
    nexp = length(first(exponents))
    open(path, "w") do io
        header = join(["exp_$i" for i in 1:nexp], ",") * "," *
                 join(["R$(i)_re,R$(i)_im" for i in 1:NR], ",")
        println(io, header)
        for (m, ex) in enumerate(exponents)
            c = @view coefficients[:, m]
            any(x -> abs(x) > drop_below, c) || continue
            row = join(string.(Int.(ex)), ",") * "," *
                  join(["$(real(c[i])),$(imag(c[i]))" for i in 1:NR], ",")
            println(io, row)
        end
    end
    return path
end

"""
	save_rom(dir, W, R; external_system = nothing,
	         metadata = Pair{String, <:Any}[], drop_below = 1e-14)

Write the standard ROM result layout (see module docstring) under `dir`.
`metadata` pairs are written first into `summary.txt`, followed by
`julia_version`, the current git commit (when available) and a timestamp.

Pass `external_system` whenever the model had one.  `W` and `R` record no external
metadata of their own, so without it an archive of a **re-based** system cannot be mapped
back to the physical external coordinates: its external columns and rows are expressed in
`r′`, and only the basis `Q` recovers `r = Q r′`.  When the system carries a basis it is
serialised to `data/external_basis.jls` and reported in `summary.txt`; a system that was
never re-based writes nothing, since `r′` and `r` coincide.
"""
function save_rom(dir::AbstractString, W::Parametrisation, R::ReducedDynamics;
        external_system = nothing,
        metadata = Pair{String, Any}[], drop_below::Real = 1e-14)
    data = joinpath(dir, "data")
    mkpath(data)
    mkpath(joinpath(dir, "figures"))
    serialize(joinpath(data, "W.jls"), W)
    serialize(joinpath(data, "R.jls"), R)
    basis = external_basis(external_system)
    basis === nothing || serialize(joinpath(data, "external_basis.jls"), basis)
    write_rom_coefficients_csv(joinpath(data, "R_coefficients.csv"),
        R.poly.multiindex_set.exponents, R.poly.coefficients;
        drop_below = drop_below)
    open(joinpath(dir, "summary.txt"), "w") do io
        for (k, v) in metadata
            println(io, k, ": ", v)
        end
        if basis !== nothing
            println(
                io, "external_coordinates: re-based; reduced external coordinates are ",
                "r′ with r = Q r′, Q in data/external_basis.jls")
        end
        println(io, "julia_version: ", VERSION)
        commit = try
            readchomp(`git rev-parse --short HEAD`)
        catch
            "unknown"
        end
        println(io, "morfe_commit: ", commit)
        println(io, "timestamp: ", time())
    end
    return dir
end

"""
	read_rom_coefficients(csv) -> (exponents::Matrix{Int}, coefficients::Matrix{ComplexF64})

Parse an `R_coefficients.csv` written by [`write_rom_coefficients_csv`](@ref).
Returns `exponents` of size `(L, NVAR)` (one monomial per row) and
`coefficients` of size `(L, NR)`.
"""
function read_rom_coefficients(csv::AbstractString)
    lines = readlines(csv)
    isempty(lines) && error("read_rom_coefficients: empty file $csv")
    header = split(lines[1], ',')
    nexp = count(h -> startswith(h, "exp_"), header)
    nre = count(h -> endswith(h, "_re"), header)
    @assert nexp > 0 && nre > 0 "read_rom_coefficients: unrecognised header in $csv"
    L = length(lines) - 1
    exponents = Matrix{Int}(undef, L, nexp)
    coefficients = Matrix{ComplexF64}(undef, L, nre)
    for (r, line) in enumerate(@view lines[2:end])
        parts = split(line, ',')
        for j in 1:nexp
            exponents[r, j] = parse(Int, parts[j])
        end
        for i in 1:nre
            re = parse(Float64, parts[nexp + 2i - 1])
            im_ = parse(Float64, parts[nexp + 2i])
            coefficients[r, i] = complex(re, im_)
        end
    end
    return exponents, coefficients
end

"""
	normal_form_branch(R; parameter = nothing,
					   amplitudes = range(0, 3; length = 600),
					   parameter_range = (-Inf, Inf),
					   external_point = nothing,
					   sheet = :all, jump = 1e-3)
		-> (; amplitude, frequency, parameter, stable)

Periodic-orbit data for a reduced dynamics in **complex normal form spanned by a single
conjugate pair**: the limit-cycle branch, or the backbone, as plain vectors ready to plot.

## What it computes

With one conjugate pair the polar substitution `z₁ = ρ·exp(iθ)` removes the phase. Every
monomial surviving a complex-normal-form reduction has `a - b = 1` in its first two
exponents, so `z₁^a z̄₁^b η^c` contributes `ρ^(a+b) exp(iθ) η^c` and the first row factors:

	R₁(ρ, ρ, η) = ρ · g(ρ, η),      g(ρ, η) = Σ c_{abc} ρ^(a+b-1) η^c

which splits the dynamics into an amplitude equation and a phase equation,

	ρ̇ = ρ · Re g(ρ, η)             Ω = θ̇ = Im g(ρ, η)

Everything returned is read off `g`. A limit cycle is a root of `Re g`, its frequency is
`Im g` there, and it is stable where `∂ρ(ρ · Re g) < 0`.

## Two modes

- **`parameter = nothing`** — the **backbone**. `frequency` is `Im g(ρ)` over `amplitudes`,
  evaluated at `external_point`; `parameter` comes back empty. No root-find is involved, and
  it needs no external coordinate, so an autonomous ROM (a conservative oscillator, say) is
  covered.
- **`parameter = k`** — the **bifurcation diagram** in the `k`-th external coordinate. For
  each amplitude the corresponding parameter values are solved for, so a fold in `ρ` against
  the parameter is traced without any special handling.

## Why it solves for the parameter, not the amplitude

At fixed `ρ`, `Re g(ρ, η)` is a *real polynomial* in `η_k`, so its roots come from one
companion-matrix eigenvalue solve. There is no iteration, no initial guess, no continuation
and no step-size control, and a fold is simply a monotone `η(ρ)`.

The other orientation is worse than merely inconvenient. Continuing in the parameter makes
`ρ` the unknown, which is the direction whose series diverges over most of a typical sweep;
in `examples/05_karman_vortex_street` that arrangement once drove an order-7 branch to
`ρ = 234` before it was abandoned for this one.

## This function assumes a single conjugate pair in complex normal form

`ROM ≠ 2` is rejected, and so is any first-row monomial with `a - b ≠ 1`. The second check
is the important one: such a monomial keeps a residual `exp(i(a-b-1)θ)`, so `R₁/z₁` is not a
function of `ρ` alone and the polar reduction above is invalid. Nothing about the resulting
numbers would look wrong, which is why this is an error rather than a warning. A `:graph`
style reduction will fail here by design.

## One root per amplitude, or all of them

`Re g(ρ, ·)` has as many roots as its degree in `η_k`, and a high-order reduction can put
more than one of them inside a physically sensible window. They are different sheets: only
the one growing out of the bifurcation is the continuation of the branch, and the others
are the truncated series crossing zero again somewhere it has stopped converging.

- **`sheet = :all`** (the default) returns every real root in range, so an amplitude with
  several roots appears several times. Nothing is decided for you.
- **`sheet = :primary`** follows the sheet the sweep starts on: at each amplitude it keeps
  the root nearest the previous one, seeded from `η = 0`, and **stops** when the nearest
  root is farther than `jump`. That gap means the tracked sheet has ended and the next root
  belongs to a different one, so continuing would splice two branches into one curve.
  `jump` is in the parameter's own units; the default suits a coordinate of order `1e-2`.

Marching in `ρ` is what makes this reliable: the sheets are single-valued in amplitude even
where they fold in the parameter, so no fold-handling is needed.

## Arguments

- `amplitudes` — the `ρ` grid. Zero is fine and gives the linear behaviour.
- `parameter_range` — `(lo, hi)` bounds; roots outside are dropped, as are complex roots.
- `external_point` — values of the *other* external coordinates, defaulting to zeros. In
  `parameter = k` mode entry `k` is ignored, being what is solved for.
- `sheet`, `jump` — branch selection, above.

## Returns

Four vectors of equal length. In `parameter = nothing` mode `parameter` is empty and the
other three follow `amplitudes`; otherwise each entry is one `(ρ, η)` point of the branch.

## Plotting

MORFE draws nothing. Load a Makie backend and plot the vectors:

```julia
using CairoMakie

b = normal_form_branch(R; parameter = 1)
lines(b.parameter, b.amplitude; axis = (xlabel = "η", ylabel = "ρ"))
```

The cohomological solve is graded, so one curve per truncation order comes from truncating
`R` with [`restrict_ReducedDynamics_to_degree`](@ref) and calling this again; no re-solve is
needed.

See also [`ReducedDynamics`](@ref), [`restrict_ReducedDynamics_to_degree`](@ref).
"""
function normal_form_branch(R::ReducedDynamics{ROM, NVAR};
        parameter::Union{Nothing, Int} = nothing,
        amplitudes = range(0, 3; length = 600),
        parameter_range::Tuple{Real, Real} = (-Inf, Inf),
        external_point = nothing,
        sheet::Symbol = :all,
        jump::Real = 1e-3) where {ROM, NVAR}
    sheet in (:all, :primary) || throw(ArgumentError(
        "sheet must be :all or :primary, got :$sheet"))
    ROM == 2 || throw(ArgumentError(
        "normal_form_branch needs a single conjugate pair (ROM = 2); this ReducedDynamics " *
        "has ROM = $ROM. Reduce onto one pair, or slave the extra coordinates first."))
    N_EXT = NVAR - ROM
    η₀ = external_point === nothing ? zeros(Float64, N_EXT) : collect(Float64, external_point)
    length(η₀) == N_EXT || throw(ArgumentError(
        "external_point has $(length(η₀)) entries but the model has $N_EXT external " *
        "coordinates"))
    if parameter !== nothing
        1 <= parameter <= N_EXT || throw(ArgumentError(
            "parameter = $parameter is not in 1:$N_EXT"))
    end

    exps = multiindex_set(R).exponents
    row = @view coefficients(R)[1, :]

    # `g`'s exponents: ρ^(a+b-1) times the external monomial. A monomial that fails a-b == 1
    # keeps an exp(i(a-b-1)θ) factor, so R₁/z₁ would not be a function of ρ alone.
    keep = Int[]
    for (m, e) in enumerate(exps)
        abs(row[m]) == 0 && continue
        e[1] - e[2] == 1 || throw(ArgumentError(
            "monomial $(Tuple(e)) has a - b = $(e[1] - e[2]) ≠ 1 in row 1, so this reduced " *
            "dynamics is not in complex normal form on a single conjugate pair; the polar " *
            "substitution z₁ = ρ·exp(iθ) does not remove the phase. Re-run `parametrise` " *
            "with `style = :complex_normal_form`."))
        push!(keep, m)
    end
    ρ_pow = [exps[m][1] + exps[m][2] - 1 for m in keep]
    ext_pow = [Tuple(exps[m][(ROM + 1):NVAR]) for m in keep]
    coef = [row[m] for m in keep]

    _g(ρ, η) = sum(
        (coef[i] * ρ^ρ_pow[i] * prod(η[j]^ext_pow[i][j] for j in 1:N_EXT; init = 1.0)
        for i in eachindex(keep)); init = zero(ComplexF64))
    # d/dρ (ρ · Re g) = Σ Re(c) (ρ_pow + 1) ρ^ρ_pow · η-part
    _dρ(ρ, η) = sum(
        (real(coef[i]) * (ρ_pow[i] + 1) * ρ^ρ_pow[i] *
         prod(η[j]^ext_pow[i][j] for j in 1:N_EXT; init = 1.0)
        for i in eachindex(keep)); init = 0.0)

    amp = Float64[]
    freq = Float64[]
    par = Float64[]
    stab = Bool[]

    if parameter === nothing
        for ρ in amplitudes
            g = _g(Float64(ρ), η₀)
            push!(amp, ρ)
            push!(freq, imag(g))
            push!(stab, _dρ(Float64(ρ), η₀) < 0)
        end
        return (; amplitude = amp, frequency = freq, parameter = par, stable = stab)
    end

    lo, hi = parameter_range
    η = copy(η₀)
    η_prev = NaN
    for ρ in amplitudes
        # Re g(ρ, ·) as a real polynomial in η[parameter], ascending powers.
        p = zeros(Float64, maximum(e[parameter] for e in ext_pow) + 1)
        for i in eachindex(keep)
            other = prod(j == parameter ? 1.0 : η₀[j]^ext_pow[i][j] for j in 1:N_EXT;
                init = 1.0)
            p[ext_pow[i][parameter] + 1] += real(coef[i]) * Float64(ρ)^ρ_pow[i] * other
        end
        roots = [r for r in _real_roots(p) if lo <= r <= hi]
        if sheet === :primary
            isempty(roots) && break
            r = roots[argmin(abs.(roots .- (isnan(η_prev) ? 0.0 : η_prev)))]
            # A nearest root that is not near means this sheet has ended and the next
            # one belongs to a different branch; stop rather than jump across the gap.
            isnan(η_prev) || abs(r - η_prev) <= jump || break
            roots = [r]
            η_prev = r
        end
        for r in roots
            η[parameter] = r
            push!(amp, ρ)
            push!(par, r)
            push!(freq, imag(_g(Float64(ρ), η)))
            push!(stab, _dρ(Float64(ρ), η) < 0)
        end
    end
    return (; amplitude = amp, frequency = freq, parameter = par, stable = stab)
end

"""
	_real_roots(p) -> Vector{Float64}

Real roots of the polynomial whose coefficients `p` are given in **ascending** powers, by
companion-matrix eigenvalues.

Trailing near-zero coefficients are trimmed first: they would otherwise make the companion
matrix singular and manufacture spurious roots at infinity. A root counts as real when its
imaginary part is negligible against its own magnitude, since an exactly real root of a
floating-point companion matrix generally comes back with a tiny imaginary part.
"""
function _real_roots(p::Vector{Float64})
    n = something(findlast(c -> abs(c) > 1e-14 * maximum(abs, p; init = 1.0), p), 0)
    n <= 1 && return Float64[]
    c = p[1:n] ./ p[n]
    C = zeros(Float64, n - 1, n - 1)
    for i in 2:(n - 1)
        C[i, i - 1] = 1.0
    end
    for i in 1:(n - 1)
        C[1, i] = -c[n - i]
    end
    return [real(λ) for λ in eigvals(C) if abs(imag(λ)) <= 1e-8 * max(abs(λ), 1.0)]
end

end # module RomIO
