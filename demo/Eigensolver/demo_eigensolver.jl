# -------------------------------------------------------------------
# import modules, define helper functions

include(joinpath(@__DIR__, "../../src/SpectralDecomposition/Eigensolvers.jl"))
import .Eigensolvers

using SparseArrays
using LinearAlgebra
using Logging
using Random
using Printf

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
shift = 1e-3 + 1.0im  # nothing or complex-valued shift
whichs = :LM # :LR, :LM, ...
tolerance = 1e-16
ncv = max(nev + 30, 120)
v0 = randn(MersenneTwister(1), n)


t0 = time()
result = Eigensolvers.generalized_eigenpairs(
	A,
	B;
	nev = nev,
	shift = shift,
	which = whichs,
	tol = tolerance,
	ncv = ncv,
	v0 = v0,
	sort_largest_real = true,
)
elapsed = time() - t0

# -------------------------------------------------------------------
# print results and diagnostics

println("Converged: ", result.nconv)
println("which = ", whichs, ", shift = ", shift)

fmt_complex(z) = @sprintf("%.5f %+.5fi", real(z), imag(z))

println("Eigenvalues:")
for (i, lambda) in enumerate(result.values)
	println(@sprintf("  %3d: %24s", i, fmt_complex(lambda)))
end

if show_time
	println(@sprintf("Solve time: %.6f s", elapsed))
end

# -------------------------------------------------------------------
# plotting (only if Plots.jl is available)
if enable_plot
	try
		# Try to load Plots; if it fails, we skip plotting.
		using Plots

		real_parts = real.(result.values)
		p = scatter(real_parts, imag.(result.values);
			ms = 6, alpha = 0.9,
			marker_z = real_parts, c = :viridis,
			colorbar_title = "Re(lambda)",
			xlabel = "Re(lambda)", ylabel = "Im(lambda)",
			title = "Generalized Eigenpairs (nev=$(nev), shift=$(shift))",
			legend = false, grid = true)
		vline!([0.0]; color = :black, lw = 1)
		hline!([0.0]; color = :black, lw = 1)

		# Overwrite any existing file
		savefig(p, plot_file)
		println("Saved eigenpair plot to: ", plot_file)

	catch e
		@warn "Plots.jl is not available or failed to load; skipping plot generation." exception=e
	end
end
