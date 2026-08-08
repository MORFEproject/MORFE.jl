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
"""
module RomIO

using Serialization
using ..ParametrisationMethod: Parametrisation, ReducedDynamics
using ..ExternalSystems: external_basis

export save_rom, read_rom_coefficients, write_rom_coefficients_csv

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

end # module RomIO
