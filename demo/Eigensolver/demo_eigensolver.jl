import Pkg
using Logging

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PROJECT_TOML = normpath(joinpath(PROJECT_ROOT, "Project.toml"))

if isfile(PROJECT_TOML) && normpath(Base.active_project()) != PROJECT_TOML
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            Logging.with_logger(Logging.NullLogger()) do
                Pkg.activate(PROJECT_ROOT; io=devnull)
            end
        end
    end
end

# -------------------------------------------------------------------
# import modules, define helper functions

if !isdefined(@__MODULE__, :Eigensolvers)
    include(joinpath(@__DIR__, "../../src/Eigensolvers.jl"))
end

import .Eigensolvers

using SparseArrays
using LinearAlgebra
using Random
using Printf

function load_plots_module_silent()
    # Use a dedicated local env so the main project manifest never interferes.
    plot_env = joinpath(@__DIR__, ".plotenv")

    # If this env was resolved with a different Julia version, rebuild its manifest.
    manifest_path = joinpath(plot_env, "Manifest.toml")
    if isfile(manifest_path)
        manifest_text = read(manifest_path, String)
        m = match(r"julia_version\s*=\s*\"([^\"]+)\"", manifest_text)
        if m !== nothing
            current_mm = "$(VERSION.major).$(VERSION.minor)"
            manifest_ver = m.captures[1]
            if !startswith(manifest_ver, current_mm * ".")
                rm(manifest_path; force=true)
            end
        end
    end

    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            Logging.with_logger(Logging.NullLogger()) do
                Pkg.activate(plot_env; io=devnull)
                if !haskey(Pkg.project().dependencies, "Plots")
                    Pkg.add("Plots"; io=devnull)
                else
                    Pkg.instantiate(; io=devnull)
                end
            end
        end
    end
    return redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            Logging.with_logger(Logging.NullLogger()) do
                Base.require(Base.PkgId(Base.UUID("91a5bcdd-55d7-5caf-9e0b-520d859cae80"), "Plots"))
            end
        end
    end
end

function load_sparse_csv(path::AbstractString)
    lines = readlines(path)
    header = split(strip(lines[1]), ',')
    nrows = parse(Int, strip(header[2]))
    ncols = parse(Int, strip(header[3]))

    nnz = length(lines) - 1
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{Float64}(undef, nnz)

    for (k, line) in enumerate(lines[2:end])
        t = split(strip(line), ',')
        I[k] = parse(Int, strip(t[1]))
        J[k] = parse(Int, strip(t[2]))
        V[k] = parse(Float64, strip(t[3]))
    end

    return sparse(I, J, V, nrows, ncols)
end


# -------------------------------------------------------------------
# define necessary directories, time, plot 

K_path = joinpath(@__DIR__, "K.csv")
M_path = joinpath(@__DIR__, "M.csv")

show_time = true
enable_plot = true
plot_file = joinpath(@__DIR__, "eigenpairs_complex_plane.png")


# -------------------------------------------------------------------
# load matrices, setup and solve eigenproblem

K = load_sparse_csv(K_path)
M = load_sparse_csv(M_path)

A = -K
B = M

n = size(A, 1)

nev = min(40, n - 2)
shift = 1e-3 +1.0im  # nothing or complex-valued shift
whichs = :LM # :LR, :LM, ...
tolerance = 1e-16
ncv = max(nev + 30, 120)
v0 = randn(MersenneTwister(1), n)


t0 = time()
result = Eigensolvers.generalized_eigenpairs(
    A,
    B;
    nev=nev,
    sigma=shift,
    which=whichs,
    tol=tolerance,
    ncv=ncv,
    v0=v0,
    sort_largest_real=true,
)
elapsed = time() - t0

# -------------------------------------------------------------------
# print results and diagnostics

println("Converged: ", result.nconv)
println("which = ", whichs, ", sigma = ", shift)

fmt_complex(z) = @sprintf("%.5f %+.5fi", real(z), imag(z))

println("Eigenvalues:")
for (i, lambda) in enumerate(result.values)
    println(@sprintf("  %3d: %24s", i, fmt_complex(lambda)))
end

if show_time
    println(@sprintf("Solve time: %.6f s", elapsed))
end

if enable_plot
    plots_mod = load_plots_module_silent()

    real_parts = real.(result.values)
    scatter_fn = getfield(plots_mod, :scatter)
    vline_fn = getfield(plots_mod, Symbol("vline!"))
    hline_fn = getfield(plots_mod, Symbol("hline!"))
    savefig_fn = getfield(plots_mod, :savefig)
    shift_label = isnothing(shift) ? "none" : string(shift)

    p = Base.invokelatest(
        scatter_fn,
        real_parts,
        imag.(result.values);
        ms=6,
        alpha=0.9,
        marker_z=real_parts,
        c=:viridis,
        colorbar_title="Re(lambda)",
        xlabel="Re(lambda)",
        ylabel="Im(lambda)",
        title="Generalized Eigenpairs (nev=$(nev), shift=$(shift_label))",
        legend=false,
        grid=true,
    )
    Base.invokelatest(vline_fn, p, [0.0]; color=:black, lw=1)
    Base.invokelatest(hline_fn, p, [0.0]; color=:black, lw=1)

    if isfile(plot_file)
        rm(plot_file; force=true)
    end
    Base.invokelatest(savefig_fn, p, plot_file)
    println("Saved eigenpair plot to: ", plot_file)
end
