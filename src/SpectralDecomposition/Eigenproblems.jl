"""
Module `Eigenproblems` — solve and manage the generalised eigenproblem for an `NDOrderModel`.

Starting from an `NDOrderModel`, the companion-form first-order matrices `(A, B)` are
assembled and the generalised eigenproblem `A v = λ B v` is solved.  The results are
stored in an `Eigenproblem` struct together with:

---
# Module contents
### Eigensolver
Eigensolver is a type that implements different methods to solve the eigenproblem. 
Every Eigensolver must have the parent type [`AbstractEigensolver`](@ref) needs to implement to methods: [`solve`](@ref) and [`solve_left`](@ref).
Implemented eigensolvers are:
1. [`DefaultEigensolver`](@ref): Uses the `eigen()` from the package LinearAlgebra.
	Calculates full spectrum. This solver is only recommended for small problems. 
2. [`ArpackEigensolver`](@ref): Uses the `eigs()` from the package Arpack.
	Calculates a specific number of modes of smalles magnitude by default. 
	Works only for sparse matrices
3. MorfeEigensolver: Uses the `eigs()` from the package Arpack with a shift of the spectrum.
4. [`StructureModalDampingEigensolver`](@ref): Uses the `eigs()` from the package Arpack, but operates
	specifically  on the second order system with Raleigh damping.

### EigenProblem
Struct with the attributs:
- `model`: [`NDOrderModel`](@ref)
- `solver`: [`AbstractEigensolver`](@ref)
- `eigenvalues`: Vector containing the eigenvalues
- `eigenvectors`: 3D Array containing the eigenvectors as columns
- `master_modes`: boolean vector of the same length as eigenvectors, showing wich eigenvectors are master_modes
- `left_eigenvectors`: Matrix containing the left eigenvectors as columns

### Compute eigenproblem [`solve_eigenproblem`](@ref)
Function that takes a [`NDOrderModel`](@ref) and a [`AbstractEigensolver`](@ref) and calculates the eigenvalues and left and right eigenvectors.
Additionally it takes a function to sort the eigenvectors and a function to normalise.
It generates and returns an EigenProblem and is the main interaction of this module.

### Master mode selection 
The selection of the master modes is done after using `solve_eigenproblem`.
There are the options:
1. [`select_master_modes_by_hand`](@ref)
2. [`select_master_modes_by_sorting`](@ref)
3. [`select_master_modes_by_target_frequency`](@ref)

"""
module Eigenproblems

using ..FullOrderModel
using ..Eigensolvers

using LinearAlgebra

export AbstractEigensolver, DefaultEigensolver, ArpackEigensolver, MorfeEigensolver,
       StructureModalDampingEigensolver
export solve, solve_left, sort_by_magnitude!, normalise_biorthogonal!
export Eigenproblem, solve_eigenproblem, get_eigenpairs, select_master_modes_by_hand,
       select_master_modes_by_sorting, select_master_modes_by_target_frequency
export left_eigenmode_orders_from_slice

"""
	AbstractEigensolver

Abstract supertype for all eigensolvers accepted by `solve_eigenproblem`.

Concrete subtypes must implement `solve` and `solve_left` for `NDOrderModel`.
Two built-in subtypes are provided: `DefaultEigensolver` and `ArpackEigensolver`.

## Interface contract — full order-blocks

Both `solve` and `solve_left` must return eigenvectors as `FOM × ORD × n`
arrays containing ALL companion order-blocks, not just the physical slice:

- right: `(λB − A) ψ = 0`, blocks `ψ = [ψ_1; …; ψ_ORD]` with `ψ_{k+1} = λ ψ_k`;
- left (sesquilinear): `φᴴ (λB − A) = 0`, reported eigenvalue `λ`
  (the pencil eigenvalue of the adjoint problem is `conj(λ)`).

The eigensolver is the single owner of eigenvalue knowledge: it uses `λ` to
*define* the eigenvector blocks, and downstream code (orthogonality and
invariance operators) reads the blocks without ever folding eigenvalues.
Solvers that naturally produce only the physical left slice can reconstruct
the blocks with [`left_eigenmode_orders_from_slice`](@ref).
"""
abstract type AbstractEigensolver end

function solve(model::NDOrderModel, solver::AbstractEigensolver, args...)
    error("solve not implemented for $(typeof(solver))")
end
function solve_left(model::NDOrderModel, solver::AbstractEigensolver, args...)
    error("solve not implemented for $(typeof(solver))")
end

"""
	DefaultEigensolver <: AbstractEigensolver

Dense eigensolver backed by `LinearAlgebra.eigen`. Computes all eigenpairs.
Suitable for small to medium full-order models (dense matrices).
"""
struct DefaultEigensolver <: AbstractEigensolver end

"""
	solve(model::NDOrderModel, solver::DefaultEigensolver)

Solves right eigenproblem using `eigen` from LinearAlgebra. 
Let `A` and `B` be the first order matrices of `model`. Then it returns the eigenpairs
```math
	(A-\\lambda_k B)y_k = 0
```
"""
function solve(model::NDOrderModel, solver::DefaultEigensolver)
    A, B = linear_first_order_matrices(model)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    eig_result = eigen(A, B)

    # Reshape eigenvectors from (ORD*FOM) x (ORD*FOM) to (FOM x ORD x number of eigenvalues)
    num_eigenvals = length(eig_result.values)
    reshaped_eigenvectors = reshape(eig_result.vectors, FOM, ORD, num_eigenvals)
    return eig_result.values, reshaped_eigenvectors
end

"""
	solve_left(model::NDOrderModel, solver::DefaultEigensolver)

Solves left eigenproblem using `eigen` from LinearAlgebra. 
let `A` and `B` be the first order matrices of `model`. Then it returns the eigenpairs
```math
	x_k^H(A-\\lambda_k B) = 0
```
"""
function solve_left(model::NDOrderModel, solver::DefaultEigensolver)
    A, B = linear_first_order_matrices(model)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    eig_result = eigen(A', B')

    # Reshape eigenvectors from (ORD*FOM) x (ORD*FOM) to (FOM x ORD x number of eigenvalues)
    num_eigenvals = length(eig_result.values)
    reshaped_eigenvectors = reshape(eig_result.vectors, FOM, ORD, num_eigenvals)
    return conj(eig_result.values), reshaped_eigenvectors
end

"""
	ArpackEigensolver <: AbstractEigensolver

Sparse eigensolver backed by `Arpack.eigs`. Computes `nev` smallest-magnitude
eigenpairs. If `nev` is not specified, Arpack's default count is used.

# Fields

- `nev::Union{Nothing, Int64}` — number of eigenpairs to compute.  `nothing` means
  "not chosen yet"; the zero-argument constructor warns, because leaving the count
  to Arpack on a large model is rarely what the caller wants.
- `eigenvalues::Union{Nothing, Vector{ComplexF64}}` — the eigenvalues from the most
  recent [`solve`](@ref), cached so the left problem can reuse them.  Left
  **undefined** by the constructors, not set to `nothing`: guard with `isdefined`
  before reading it.
"""
mutable struct ArpackEigensolver <: AbstractEigensolver
    nev::Union{Nothing, Int64}
    eigenvalues::Union{Nothing, Vector{ComplexF64}}

    function ArpackEigensolver()
        @warn "Initialised ArpackEigensolver with no nev. Not recommended! Use DefaultEigensolver instead."
        new(nothing)
    end
    function ArpackEigensolver(nev::Int64)
        @assert nev>0 "nev must be greater than zero!"
        new(nev)
    end
end

"""
	MorfeEigensolver <: AbstractEigensolver

Sparse eigensolver for the *shifted* eigenproblem, backed by `Arpack.eigs` through
[`generalised_eigenpairs`](@ref).  Shifting targets the eigenvalues nearest a chosen
point in the complex plane, which is how master modes around a given frequency are
picked out of a large spectrum.

# Fields

- `nev::Union{Nothing, Int64}` — number of eigenpairs to compute.  `nothing` is
  resolved to the full problem size on the first [`solve`](@ref).
- `shift::Union{Nothing, ComplexF64}` — the shift point.  `nothing` falls back to
  the unshifted `eigs`.
- `eigenvalues::Union{Nothing, Vector{ComplexF64}}` — eigenvalues from the most
  recent [`solve`](@ref), cached so the left problem can reuse them.  Left
  **undefined** by the constructors, not set to `nothing`: guard with `isdefined`
  before reading it.
"""
mutable struct MorfeEigensolver <: AbstractEigensolver
    nev::Union{Nothing, Int64}
    shift::Union{Nothing, ComplexF64}
    eigenvalues::Union{Nothing, Vector{ComplexF64}}

    function MorfeEigensolver()
        new(nothing, nothing)
    end
    function MorfeEigensolver(nev::Int64, shift::ComplexF64)
        @assert nev>0 "nev must be greater than zero!"
        new(nev, shift)
    end
end

"""
	solve(model::NDOrderModel, solver::MorfeEigensolver)

Solves right eigenproblem using `generalised_eigenpairs` from `MORFE.Eigensolvers`. 
Let `A` and `B` be the first order matrices of `model`. Then it returns the eigenpairs
```math
	(A-\\lambda_k B)y_k = 0
```
"""
function solve(model::NDOrderModel, solver::MorfeEigensolver)
    A, B = linear_first_order_matrices(model)
    if solver.nev === nothing
        solver.nev = size(A, 1)
    end
    eig_result = generalised_eigenpairs(A, B; solver.nev, shift = solver.shift)

    # Reshape eigenvectors from (ORD*FOM) x (number of eigenvalues) to (FOM x ORD x number of eigenvalues)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    num_eigenvals = length(eig_result.values)
    reshaped_eigenvectors = reshape(eig_result.vectors, FOM, ORD, num_eigenvals)
    solver.eigenvalues = eig_result.values
    return eig_result.values, reshaped_eigenvectors
end

"""
	solve_left(model::NDOrderModel, solver::MorfeEigensolver)

Solves left eigenproblem using σ-shift to recover the correct eigenvalues.
"""
function solve_left(model::NDOrderModel, solver::MorfeEigensolver)
    A, B = linear_first_order_matrices(model)
    @assert solver.nev == length(solver.eigenvalues)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    left_eigenvectors = Array{ComplexF64}(undef, FOM, ORD, solver.nev)
    eigenvalues = Vector{ComplexF64}(undef, solver.nev)
    for i in 1:(solver.nev)
        eig_results = generalised_eigenpairs(
            A', B'; nev = 1, ncv = 30, shift = conj(solver.eigenvalues[i]))
        left_eigenvectors[:, :, i] = reshape(eig_results.vectors[:, 1], FOM, ORD)
        eigenvalues[i] = conj(eig_results.values[1])
    end
    return eigenvalues, left_eigenvectors
end

"""
	StructureModalDampingEigensolver <: AbstractEigensolver

solves eigenproblem of the mechanical 2nd order problem:
```math
M \\ddot{U} + C \\dot{U} + K U + n(U) = 0
```
with ``C = \\alpha*M + \\beta*K`` under the assumption that M and K are spd matrices.

The steps are:
1) calculate eigenpair \$\\omega_k, \\varphi_k\$ of: \$(K-\\omega_k^2*M)\\varphi_k=0\$
2) calculate: \$\\xi_k=0.5(\\frac{\\alpha}{\\omega_k} + \\beta * \\omega_k)\$
3) calculate eigenvalues: \$\\lambda_k = -\\xi_k*\\omega_k \\sqrt{1-\\xi_k^2}\$

Calculates only the first `nev` eigenvectors.

# Fields

- `eigenvalues::Union{Nothing, Vector}` — eigenvalues from the most recent
  [`solve`](@ref), `nothing` before the first one.  Cached so the left problem can
  reuse them instead of re-running Arpack.
- `nev::Int64` — number of modes to compute, counted in the second-order problem
  (so `nev` frequencies `ωₖ`, not `2·nev` first-order eigenvalues).
- `α::Float64` — mass-proportional damping coefficient in `C = αM + βK`.
- `β::Float64` — stiffness-proportional damping coefficient in `C = αM + βK`.
"""
mutable struct StructureModalDampingEigensolver <: AbstractEigensolver
    eigenvalues::Union{Nothing, Vector}
    nev::Int64
    α::Float64
    β::Float64

    function StructureModalDampingEigensolver(nev::Int64, α::Float64, β::Float64)
        @assert nev>0 "nev must be greater than zero!"
        new(nothing, nev, α, β)
    end
end

"""
	solve(mass::AbstractMatrix, stiffness::AbstractMatrix, solver::StructureModalDampingEigensolver)

Calculates right eigenvectors in the secod order form.
Uses the relations:
```math
	(K-\\omega_k^2*M)\\y_{k,U}=0 
```
```math
	y_{k,V}=\\lambda_k y_{k,U} 
```
and assumes ``C = \\alpha*M + \\beta*K``.
"""
function solve(
        ::AbstractMatrix,
        ::AbstractMatrix,
        ::StructureModalDampingEigensolver
)
    error(
        "StructureModalDampingEigensolver requires Arpack.jl and LinearMaps.jl.\n" *
        "Load them with `using Arpack, LinearMaps` to activate the MORFE extension.",
    )
end

"""
	solve(model::NDOrderModel, solver::StructureModalDampingEigensolver)

Thin wrapper: extracts `M = model.linear_terms[end]` and `K = model.linear_terms[1]`
and delegates to `solve(M, K, solver)`.
"""
function solve(model::NDOrderModel, solver::StructureModalDampingEigensolver)
    return solve(model.linear_terms[end], model.linear_terms[1], solver)
end

"""
	left_eigenmode_orders_from_slice(linear_terms, left_slice, eigenvalues)
	-> Array{ComplexF64, 3}

Reconstruct the full left eigenvector order-blocks from the physical-space
slice, for callers that compute only the latter. For the sesquilinear left
eigenvector `φ = [φ_1; …; φ_ORD]` of reported eigenvalue `λ` (physical slice
`ℓ = φ_ORD`, satisfying `L(λ)ᴴ ℓ = 0` with `L(s) = Σ_k B_k s^k`), the companion
block equations give

```
φ_{ORD-1} = conj(λ) · (B_ORDᴴ ℓ) + B_{ORD-1}ᴴ ℓ
φ_j       = conj(λ) · φ_{j+1}   + B_jᴴ ℓ          j = ORD-2, …, 1
```

The eigenvalue is used only to *define* the eigenvector from its slice — the
per-monomial cohomological solve reads the blocks and never touches it. Prefer
eigensolvers that return the full blocks directly (`solve_left` does); use this
only when a slice is all you have.

The slice may carry an arbitrary per-mode scale: the blocks scale with it, and
the orthogonality equations are invariant under per-mode row scaling.
"""
function left_eigenmode_orders_from_slice(
        linear_terms::NTuple{ORDP1, <:AbstractMatrix},
        left_slice::AbstractMatrix,      # FOM × n
        eigenvalues::AbstractVector     # length n — reported (right) eigenvalues
) where {ORDP1}
    ORD = ORDP1 - 1
    FOM = size(left_slice, 1)
    n = size(left_slice, 2)
    @assert length(eigenvalues) == n "eigenvalues must match the number of slice columns"
    blocks = Array{ComplexF64}(undef, FOM, ORD, n)
    for k in 1:n
        ℓ = view(left_slice, :, k)
        ν = conj(eigenvalues[k])
        blocks[:, ORD, k] .= ℓ
        if ORD > 1
            @views blocks[:, ORD - 1, k] .= ν .* (linear_terms[ORDP1]' * ℓ) .+
                                            linear_terms[ORD]' * ℓ
            for j in (ORD - 2):-1:1
                @views blocks[:, j, k] .= ν .* blocks[:, j + 1, k] .+
                                          linear_terms[j + 1]' * ℓ
            end
        end
    end
    return blocks
end

"""
	_structural_left_eigenmode_orders(λ, Y, mass, damping) -> Array{ComplexF64, 3}

Analytic companion left-eigenvector order-blocks for a proportionally damped
second-order structure `M ẍ + C ẋ + K x = 0` with real symmetric `M, K` and
`C = αM + βK`. For the sesquilinear left eigenvector `φ` (solving
`φᴴ (λB − A) = 0` on the companion pencil) with real position mode
`ϕ = Y[:, 1, k]`:

```
φ_2 = ϕ                    (physical slice)
φ_1 = (conj(λ) M + C) ϕ
```

These are exactly the blocks `solve_left` would return — algebraically equal to
`-(1/conj(λ)) Kᵀϕ` via the quadratic eigenrelation, but built from the
moderate-norm `M`, `C` instead of `K` (which amplifies eigensolver noise by
`(ω_max/ω₁)²`). No adjoint eigensolve is needed because the proportional
damping makes the blocks analytic in `ϕ`.
"""
function _structural_left_eigenmode_orders(
        λ::AbstractVector,
        Y::AbstractArray{<:Complex, 3},
        mass::AbstractMatrix,
        damping::AbstractMatrix
)
    FOM = size(Y, 1)
    n_eigs = size(Y, 3)
    @assert size(Y, 2) == 2 "structural left blocks require a second-order model (ORD = 2)"
    left = Array{ComplexF64}(undef, FOM, 2, n_eigs)
    for k in 1:n_eigs
        ϕ = view(Y, :, 1, k)
        left[:, 2, k] .= ϕ
        left[:, 1, k] .= conj(λ[k]) .* (mass * ϕ) .+ damping * ϕ
    end
    return left
end

"""
	solve_eigenproblem(model::NDOrderModel, solver::StructureModalDampingEigensolver; sorter!)

Specialised path for `StructureModalDampingEigensolver`: mass-normalisation is
built into `solve`, and the left eigenvector order-blocks are analytic in the
position mode (see [`_structural_left_eigenmode_orders`](@ref)).  No adjoint
solve or biorthogonal normalisation is needed.

The optional `sorter!` kwarg has the same semantics as in the general
`solve_eigenproblem`: pass `(args...) -> nothing` to preserve the solver's
natural ordering.
"""
function solve_eigenproblem(
        model::NDOrderModel,
        solver::StructureModalDampingEigensolver;
        sorter!::Function = sort_by_magnitude!)
    λ, Y = solve(model, solver)
    sorter!(λ, Y)
    left = _structural_left_eigenmode_orders(λ, Y,
        model.linear_terms[3], model.linear_terms[2])
    return Eigenproblem(solver, λ, Y, left)
end

"""
	solve_eigenproblem(stiffness, mass, solver::StructureModalDampingEigensolver; sorter!)

Convenience overload: pass `K` and `M` directly without constructing an `NDOrderModel`.
The Rayleigh damping matrix `C = αM + βK` is rebuilt from the solver parameters
for the left eigenvector order-blocks.
"""
function solve_eigenproblem(
        stiffness::AbstractMatrix,
        mass::AbstractMatrix,
        solver::StructureModalDampingEigensolver;
        sorter!::Function = sort_by_magnitude!)
    λ, Y = solve(mass, stiffness, solver)
    sorter!(λ, Y)
    damping = solver.α * mass + solver.β * stiffness
    left = _structural_left_eigenmode_orders(λ, Y, mass, damping)
    return Eigenproblem(solver, λ, Y, left)
end

"""
	Eigenproblem{T}

Stores sorted and biorthogonally normalised left/right eigenpairs of the
generalised eigenproblem `A x = λ B x`, together with a master-mode selector.

Left eigenmodes are kept in two forms: the full order-block array and the
physical-space (highest-order) slice of it.  The full blocks feed the orthogonality
row operators directly, so no eigenvalue folding is needed to reconstruct them; the
slice is what physical-space post-processing and export want.

Fields become populated in stages — the solver fills the eigenpairs, then a
`select_master_modes_*` call marks the masters — which is why the selectors are
nullable rather than empty.

# Fields

- `solver::AbstractEigensolver` — the solver that computed the eigenpairs, retained
  so downstream code can query how they were obtained.
- `eigenvalues::Array{Complex{T}}` — sorted eigenvalues `λ`.
- `eigenmodes::Array{Complex{T}}` — right eigenvectors as `FOM × ORD × n_eigs`,
  sorted to match `eigenvalues`.
- `left_eigenmodes::Union{Nothing, Matrix{Complex{T}}}` — physical-space left
  eigenvectors, `FOM × n_eigs`; `nothing` until set.
- `left_eigenmodes_orders::Union{Nothing, Array{Complex{T}, 3}}` — full left
  eigenvector order-blocks, `FOM × ORD × n_eigs`; `nothing` when the solver supplied
  only the physical slice.
- `master_modes::Union{Nothing, Vector{Bool}}` — flags the master modes spanning the
  invariant manifold; `nothing` until a `select_master_modes_*` call sets it.
- `external_modes::Union{Nothing, Vector{Bool}}` — flags modes driven by external
  forcing rather than solved for; `nothing` when the model has no forcing.
"""
mutable struct Eigenproblem{T}
    solver::AbstractEigensolver
    eigenvalues::Array{Complex{T}}
    eigenmodes::Array{Complex{T}}              # FOM × ORD × n_eigs
    left_eigenmodes::Union{Nothing, Matrix{Complex{T}}}  # FOM × n_eigs — physical-space (highest-order) slice
    left_eigenmodes_orders::Union{Nothing, Array{Complex{T}, 3}}  # FOM × ORD × n_eigs — full order-blocks
    master_modes::Union{Nothing, Vector{Bool}}
    external_modes::Union{Nothing, Vector{Bool}}
    # Constructor from full 3-D left eigenvectors: retains the full order-block
    # array (fed to MasterModeOrthogonality) and the highest-order
    # (physical-space) slice [:, ORD, :].
    function Eigenproblem(
            solver::AbstractEigensolver,
            eigenvalues::Array{Complex{T}},
            eigenmodes::Array{Complex{T}},
            left_eigenmodes::Array{Complex{T}}) where {T}
        FOM = size(eigenmodes, 1)
        ORD = size(eigenmodes, 2)
        n_eigs = size(eigenmodes, 3)
        @assert ndims(eigenmodes) == 3 "eigenmodes must be a 3-D array (FOM × ORD × n_eigs)"
        @assert size(eigenvalues, 1) == n_eigs "length(eigenvalues) must equal size(eigenmodes, 3)"
        @assert ndims(left_eigenmodes) == 3 "left_eigenmodes must be a 3-D array (FOM × ORD × n_eigs)"
        @assert size(left_eigenmodes, 1) == FOM "left_eigenmodes dim 1 must equal FOM = $FOM"
        @assert size(left_eigenmodes, 2) == ORD "left_eigenmodes dim 2 must equal ORD = $ORD"
        @assert size(left_eigenmodes, 3) == n_eigs "left_eigenmodes dim 3 must equal n_eigs = $n_eigs"
        left_orders = Array{Complex{T}, 3}(left_eigenmodes)
        left_phys = Matrix{Complex{T}}(left_eigenmodes[:, ORD, :])
        new{T}(solver, eigenvalues, eigenmodes, left_phys, left_orders, nothing, nothing)
    end

    # Constructor accepting pre-extracted 2-D physical-space left eigenmodes
    # (FOM × n_eigs).  Use when only the physical slice is known — e.g.
    # Mechanical_Problem_Solver (2-D eigenmodes in first-order flat format,
    # 2-D left eigenmodes). For the 3-D case the n_eigs axis is dim 3; for
    # the 2-D (flat first-order) case it is dim 2. `left_eigenmodes_orders`
    # is left as `nothing`; ORD > 1 orthogonality solves require the full blocks.
    function Eigenproblem(
            solver::AbstractEigensolver,
            eigenvalues::Array{Complex{T}},
            eigenmodes::Array{Complex{T}},
            left_eigenmodes::Matrix{Complex{T}}) where {T}
        n_eigs = ndims(eigenmodes) == 3 ? size(eigenmodes, 3) : size(eigenmodes, 2)
        @assert size(eigenvalues, 1) == n_eigs "length(eigenvalues) must equal n_eigs = $n_eigs"
        @assert size(left_eigenmodes, 2) == n_eigs "left_eigenmodes must have n_eigs = $n_eigs columns"
        new{T}(solver, eigenvalues, eigenmodes, left_eigenmodes, nothing, nothing, nothing)
    end
end

"""
	solve_eigenproblem(
		model::NDOrderModel;
		solver::AbstractEigensolver = DefaultEigensolver(),
		sorter!::Function = sort_by_magnitude!,
		normaliser!::Function = normalise_biorthogonal!)

Computes left and right eigenpairs of the problem described in `model` by using the defined `solver`.
Additionally `sorter!` sorts the eigenpairs and `normaliser!` is used to normalise the eigenmodes.
"""
function solve_eigenproblem(
        model::NDOrderModel;
        solver::AbstractEigensolver = DefaultEigensolver(),
        sorter!::Function = sort_by_magnitude!,
        normaliser!::Function = normalise_biorthogonal!)

    #calculate right eigenmodes
    (eigenvalues, eigenmodes) = solve(model, solver)

    #sort eigenpairs
    sorter!(eigenvalues, eigenmodes)

    #calculate left eigenmodes
    (left_eigenvalues, left_eigenmodes) = solve_left(model, solver)
    left_eigenvalues, left_eigenmodes = sort_left_eigenmodes(
        eigenvalues, left_eigenvalues, left_eigenmodes)

    # normalise eigenpairs
    normaliser!(model, eigenmodes, left_eigenmodes)

    # Construct Eigenproblem
    return Eigenproblem(solver, eigenvalues, eigenmodes, left_eigenmodes)
end

"""
	sort_by_magnitude!(eigenvalues, eigenmodes)

Sorts eigenpairs to resemble the order:
	|λ[1]| ≤ |λ[2]| ≤ ...  
where λ=eigenvalues.
"""
function sort_by_magnitude!(eigenvalues, eigenmodes)
    idx = sortperm(eigenvalues; by = abs)
    eigenvalues .= eigenvalues[idx]
    eigenmodes .= eigenmodes[:, :, idx]
end

"""
	sort_left_eigenmodes(eigenvalues, left_eigenvalues, left_eigenmodes)

Sorts the left eigenpairs to match right eigenpairs, by using the distance function
	dist(a, b) = abs(real(a - b)) + abs(imag(a - b))
"""
function sort_left_eigenmodes(eigenvalues, left_eigenvalues, left_eigenmodes)
    tol = 1e-8
    n = length(eigenvalues)
    n_left = length(left_eigenvalues)
    @assert abs(n - n_left)<=1 "Number of left and right eigenvalues doesnt match (+-1)!"
    perm = zeros(Int, n)
    used = falses(n_left)
    dist(a, b) = abs(real(a - b)) + abs(imag(a - b))

    for i in 1:n
        diffs = map(μj -> dist(μj, eigenvalues[i]), left_eigenvalues)

        # ignore already used indices
        for j in 1:n_left
            if used[j]
                diffs[j] = Inf
            end
        end

        j = argmin(diffs)

        if diffs[j] > tol
            @warn "No good match for eigenvalue $i (distance=$(diffs[j]))"
            # println("Eigenvalues", eigenvalues)
            # println("Left eigenvalues ", left_eigenvalues)
        end

        perm[i] = j
        used[j] = true
    end
    reordered = ndims(left_eigenmodes) == 3 ? left_eigenmodes[:, :, perm] :
                left_eigenmodes[:, perm]
    return left_eigenvalues[perm], reordered
end

"""
	normalise_biorthogonal!(
		model::NDOrderModel,
		eigenmodes::Matrix{T},
		left_eigenmodes::Matrix{T})

Normalise eigenmodes to fulfill the equation
	x_i^H * B * y_j = δ_{ij}

Both sides are scaled symmetrically: with `s = sqrt(x_i^H B y_i)`, the right
eigenmode is divided by `s` and the left eigenmode by `conj(s)`, so the
sesquilinear pairing becomes exactly 1.
"""
function normalise_biorthogonal!(
        model::NDOrderModel,
        eigenmodes::Array{T},
        left_eigenmodes::Array{T}) where {T}
    @assert size(eigenmodes)==size(left_eigenmodes) "Size of left and right eigenmodes must be the same!"

    (_, B) = linear_first_order_matrices(model)
    n = ndims(eigenmodes)
    n_eigs = size(eigenmodes, n)
    for i in 1:n_eigs
        ψ = n == 3 ? vec(@view eigenmodes[:, :, i]) : @view eigenmodes[:, i]
        φ = n == 3 ? vec(@view left_eigenmodes[:, :, i]) : @view left_eigenmodes[:, i]
        s = sqrt(φ' * B * ψ)
        if n == 3
            @views eigenmodes[:, :, i] ./= s
            @views left_eigenmodes[:, :, i] ./= conj(s)
        else
            @views eigenmodes[:, i] ./= s
            @views left_eigenmodes[:, i] ./= conj(s)
        end
    end
end

"""
	get_eigenpairs(ep::Eigenproblem)

Returns eigenvalues, (right) eigenmodes and left eigenmodes of an Eigenproblem.
"""
function get_eigenpairs(ep::Eigenproblem)
    return (ep.eigenvalues, ep.eigenmodes, ep.left_eigenmodes)
end

"""
	select_master_modes_by_hand(ep::Eigenproblem, mastermodes::Vector{Bool})

Define master_modes of Eigenproblem by passing a vector of booleans.
	master_modes[i] = true  ===> eigen_modes[:,i] is a mastermode
"""
function select_master_modes_by_hand(ep::Eigenproblem, mastermodes::Vector{Bool})
    n = ndims(ep.eigenmodes) == 3 ? size(ep.eigenmodes, 3) : size(ep.eigenmodes, 2)
    @assert length(mastermodes) == n "mastermodes has wrong length!"
    ep.master_modes = mastermodes
end

"""
	select_master_modes_by_sorting(ep::Eigenproblem, nev::Int64)

Defines master_modes as the first nev eigenpairs. Sorting was done in solve_eigenproblem.
"""
function select_master_modes_by_sorting(ep::Eigenproblem, nev::Int64)
    @assert nev>0 "nev must be bigger then zero"
    n = ndims(ep.eigenmodes) == 3 ? size(ep.eigenmodes, 3) : size(ep.eigenmodes, 2)
    ep.master_modes = [i <= nev for i in 1:n]
end

"""
	select_master_modes_by_target_frequency(ep::Eigenproblem, target_frequencies::Vector, tol)

Defines master_modes by specified target values. 
All eigenvalues that are of distance of one target value smaller than `tol` are marked for mastermodes.
Distance used:
	dist(a, b) = abs(real(a - b)) + abs(imag(a - b)).
"""
function select_master_modes_by_target_frequency(
        ep::Eigenproblem,
        target_frequencies::Vector,
        tol::Float64)
    n = length(ep.eigenvalues)
    if length(target_frequencies) > n
        @warn "target_frequencies has more entries than calculated eigenvalues. Everything after index $n is neglected!"
    end

    dist(a, b) = abs(real(a - b)) + abs(imag(a - b))
    master_modes = falses(n)

    for target_frequency in target_frequencies
        dists = [dist(ep.eigenvalues[j], target_frequency) for j in 1:n]
        tmp = 0
        for j in 1:n
            if dists[j] < tol
                master_modes[j] = true
                tmp += 1
            end
        end
        if tmp == 0
            j = argmin(dists)
            @warn("No eigenvalue found for target $target_frequency. Closest distance = $(dists[j]) at eigenvalue $j")
        end
    end
    println("Chosen mastermodes: ", master_modes)
    ep.master_modes = master_modes
end

function group_conjugate_pairs()
    #TODO
end

end # module
