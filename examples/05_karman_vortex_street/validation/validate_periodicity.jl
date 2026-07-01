"""
    validate_periodicity.jl — FOM time-integration validation for the cylinder-flow DPIM demo.

For each solution point on a ROM frequency-response curve (read from CSV_PATH),
this script:
  1. Reconstructs the full FOM perturbation state  s(0) = Re(W(z, η′))
     from the serialised parametrisation W and the reduced coordinates (x1, x2).
  2. Integrates the full-order perturbation NSE for the given period T.
  3. Reports the mass-weighted periodicity error  ‖s(T) − s(0)‖_M.

Edit the constants block below to change input file, time step, or implicit weight.

CSV columns (comma-separated, header row mandatory):
  x1    Re(z₁)  — real part of the first master-mode amplitude
  x2    Im(z₁)  — imaginary part
  eta   1/Re − 1/Re₀
  T     integration period in seconds

Linear operators are cached in DATA_DIR/linear_ops.jls on the first run and
reloaded automatically. Delete that file to force recomputation.
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using MORFE
using Ferrite
using FerriteGmsh
using Gmsh
using Arpack
using LinearMaps
using StaticArrays
using LinearAlgebra
using SparseArrays
using Printf
using Serialization
using DelimitedFiles
using KLU

const EXAMPLE_DIR = realpath(joinpath(@__DIR__, ".."))

include(joinpath(EXAMPLE_DIR, "config.jl"))
include(joinpath(EXAMPLE_DIR, "fem", "mesh.jl"))
include(joinpath(EXAMPLE_DIR, "fem", "fem_setup.jl"))
include(joinpath(EXAMPLE_DIR, "solver", "steady_state.jl"))
include(joinpath(EXAMPLE_DIR, "fem", "linear_operators.jl"))
include(joinpath(EXAMPLE_DIR, "fem", "fluid_maps.jl"))
include(joinpath(EXAMPLE_DIR, "solver", "time_integration.jl"))
include(joinpath(EXAMPLE_DIR, "solver", "shooting.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these paths and parameters as needed
# ─────────────────────────────────────────────────────────────────────────────

const DATA_DIR  = joinpath(EXAMPLE_DIR, "results",
                            @sprintf("Re%.2f_ord%d", Re₀, MAX_ORD), "data")
const CSV_PATH  = joinpath(DATA_DIR, @sprintf("PO_Karman_%d_IC.csv", MAX_ORD))
const Δt_INTEG  = 1e-3   # time step (s)
const θ_INTEG   = 0.5    # implicit weight (0.5 = Crank-Nicolson, 1 = backward Euler)
const N_DAMP    = 10      # damped spin-up orbits
const γ_DAMP    = 5.0     # artificial damping coefficient (should exceed max linear growth rate)
const MAX_PICARD = 200    # maximum undamped Picard orbits
const TOL_PO    = 1e-3    # convergence tolerance on ‖E‖_M / X_max

isfile(CSV_PATH) || error("Input CSV not found: $CSV_PATH")
isdir(DATA_DIR)  || error("Data directory not found: $DATA_DIR — run main.jl first.")

@printf("validate_periodicity.jl\n")
@printf("  CSV:      %s\n", CSV_PATH)
@printf("  data_dir: %s\n", DATA_DIR)
@printf("  Δt = %.2e s,  θ = %.1f\n\n", Δt_INTEG, θ_INTEG)

# ─────────────────────────────────────────────────────────────────────────────
# Load ROM parametrisation W
# ─────────────────────────────────────────────────────────────────────────────

W_path = joinpath(DATA_DIR, "W.jls")
isfile(W_path) || error("W.jls not found in $DATA_DIR — run main.jl first.")

@info "Loading parametrisation W from $W_path ..."
W    = deserialize(W_path)
C = MORFE.ParametrisationMethod.coefficients(W)

n_fom  = size(C, 1)
n_mono = size(C, 3)
@info "  W: FOM=$n_fom, monomials=$n_mono"

# ─────────────────────────────────────────────────────────────────────────────
# Build FOM: mesh + FEM setup + linear operators
# ─────────────────────────────────────────────────────────────────────────────

# Reuse existing mesh file to avoid re-running Gmsh.
meshfile = joinpath(EXAMPLE_DIR, "fem", "cylinder_flow.msh")
isfile(meshfile) || error("Mesh file not found: $meshfile — run main.jl first to generate it.")

@info "Setting up FEM from existing mesh ..."
fom = setup_fem(meshfile)
@info "  n_free_dpim = $(fom.n_free_dpim)"

# Load or recompute linear operators ─────────────────────────────────────────

linear_ops_file = joinpath(DATA_DIR, "linear_ops.jls")

if isfile(linear_ops_file)
    @info "Loading cached operators from $linear_ops_file ..."
    ops    = deserialize(linear_ops_file)
    B₀     = ops.B₀
    B₁     = ops.B₁
    K_visc = ops.K_visc
    h₀_vec = ops.h₀_vec
else
    @info "Computing operators (Newton solve + linear assembly) — this takes a few minutes ..."

    (_, _, s₀_full) = solve_steady_state(fom; Re0 = Re₀)

    B₀, B₁ = assemble_linear_operators(s₀_full, fom; Re0 = Re₀)

    K_visc = assemble_K_visc(fom)
    K_visc .*= -_CYL_D                           # physical sign: g₁ = -D·η′·K_raw·s
    h₀_vec = K_visc * s₀_full[fom.free_dpim]     # base-flow forcing vector (Float64)

    @info "Caching operators to $linear_ops_file ..."
    serialize(linear_ops_file, (; B₀, B₁, K_visc, h₀_vec))
end

# ─────────────────────────────────────────────────────────────────────────────
# Read CSV
# ─────────────────────────────────────────────────────────────────────────────

raw = readdlm(CSV_PATH, ',', '\n'; header = true)
data_mat = raw[1]          # Matrix{Any}
header   = vec(raw[2])     # Vector of column names

# Locate required columns (case-insensitive).
# Accepted aliases: x1/x for Re(z₁), x2/y for Im(z₁).
col_idx = Dict{String, Int}()
for (i, h) in enumerate(header)
    col_idx[lowercase(strip(string(h)))] = i
end

_find_col(primary, fallback) =
    haskey(col_idx, primary)  ? col_idx[primary]  :
    haskey(col_idx, fallback) ? col_idx[fallback] :
    error("Column '$primary' (or '$fallback') not found in CSV header: $(header)")

ix  = _find_col("x1",  "x")
iy  = _find_col("x2",  "y")
ie  = _find_col("eta", "eta")
it  = _find_col("t",   "t")
ir  = _find_col("re",  "re")

# Drop trailing empty rows (readdlm may include them if the file ends with \n)
last_data = findlast(i -> !all(x -> x == "" || x == 0, data_mat[i, :]),
                     1:size(data_mat, 1))
data_mat = data_mat[1:something(last_data, 0), :]

n_pts = size(data_mat, 1)
n_pts > 0 || error("CSV contains no data rows.")
@info "CSV loaded: $n_pts rows"

# ─────────────────────────────────────────────────────────────────────────────
# Integration loop
# ─────────────────────────────────────────────────────────────────────────────

# Compute pressure-lift weights for period estimation + lift tracking
@info "Computing pressure-lift weights ..."
l_free = Base.invokelatest(compute_pressure_lift_weights, fom)[fom.free_dpim]

E_norms   = zeros(n_pts)
X_maxes   = zeros(n_pts)
F_L_maxes = zeros(n_pts)
F_L_mines = zeros(n_pts)
n_orbs    = zeros(Int, n_pts)
conv_flag = falses(n_pts)

sep = "─" ^ 144
@printf("\n%s\n", sep)
@printf("  %4s  %10s  %12s  %12s  %12s  %10s  %14s  %14s  %14s  %12s  %6s  %5s\n",
        "row", "Re", "x", "y", "eta", "T (s)", "‖E‖_M", "X_max", "‖E‖_M/X_max", "ΔF_L", "orbits", "conv")
@printf("%s\n", sep)

for i in 1:n_pts
    x_i   = Float64(data_mat[i, ix])
    y_i   = Float64(data_mat[i, iy])
    eta_i = Float64(data_mat[i, ie])
    T_i   = Float64(data_mat[i, it])
    Re_i  = Float64(data_mat[i, ir])

    # ROM initial condition: s = Re(W(z)),  z = [x+iy, x-iy, eta]
    z      = ComplexF64[x_i + im * y_i,  x_i - im * y_i,  eta_i + 0im]
    s_init = vec(real.(evaluate(W.poly, z)))

    # Find FOM periodic orbit (damped spin-up + period probe + Picard)
    _, E_norm, X_max, F_L_max, F_L_min, n_orb, conv = Base.invokelatest(
        find_periodic_orbit, s_init, eta_i, T_i, fom, B₀, B₁, K_visc, h₀_vec;
        Δt = Δt_INTEG, θ = θ_INTEG,
        n_damp = N_DAMP, γ_damp = γ_DAMP,
        max_picard = MAX_PICARD, tol = TOL_PO,
        lift_weights = l_free, verbose = false,
    )
    E_norms[i]   = E_norm
    X_maxes[i]   = X_max
    F_L_maxes[i] = isfinite(F_L_max) ? F_L_max : NaN
    F_L_mines[i] = isfinite(F_L_min) ? F_L_min : NaN
    n_orbs[i]    = n_orb
    conv_flag[i] = conv

    rel_err = X_max > 0.0 ? E_norm / X_max : NaN
    ΔF_L    = isfinite(F_L_max) && isfinite(F_L_min) ? F_L_max - F_L_min : NaN
    @printf("  %4d  %10.4f  %+12.5e  %+12.5e  %+12.5e  %10.5f  %14.6e  %14.6e  %14.6e  %12.4e  %6d  %5s\n",
            i, Re_i, x_i, y_i, eta_i, T_i, E_norm, X_max, rel_err, ΔF_L, n_orb, conv ? "✓" : "✗")
end

@printf("%s\n\n", sep)

# ─────────────────────────────────────────────────────────────────────────────
# Write output CSV
# ─────────────────────────────────────────────────────────────────────────────

out_basename = splitext(basename(CSV_PATH))[1] * "_fom_errors.csv"
out_path     = joinpath(DATA_DIR, out_basename)

open(out_path, "w") do io
    println(io, join(string.(header), ",") * ",E_M_norm,X_max,rel_err,n_orbits,converged,F_L_max,F_L_min,F_L_amp")
    for i in 1:n_pts
        row_str = join(string.(data_mat[i, :]), ",")
        rel_err = X_maxes[i] > 0.0 ? E_norms[i] / X_maxes[i] : NaN
        F_L_amp = isfinite(F_L_maxes[i]) && isfinite(F_L_mines[i]) ?
                  F_L_maxes[i] - F_L_mines[i] : NaN
        @printf(io, "%s,%.15e,%.15e,%.15e,%d,%s,%.15e,%.15e,%.15e\n",
                row_str, E_norms[i], X_maxes[i], rel_err, n_orbs[i], conv_flag[i],
                F_L_maxes[i], F_L_mines[i], F_L_amp)
    end
end

@info "Results written to $out_path"
