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
Additionally it takes a function to sort the eigenvectors and a function to normalize.
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
using Arpack

export AbstractEigensolver, DefaultEigensolver, ArpackEigensolver, MorfeEigensolver,
	StructureModalDampingEigensolver
export solve, solve_left, sort_by_magnitude!, normalize_biorthogonal!
export Eigenproblem, solve_eigenproblem, get_eigenpairs, select_master_modes_by_hand,
	select_master_modes_by_sorting, select_master_modes_by_target_frequency

"""
	AbstractEigensolver

Abstract supertype for all eigensolvers accepted by `solve_eigenproblem`.

Concrete subtypes must implement `solve` and `solve_left` for `NDOrderModel`.
Two built-in subtypes are provided: `DefaultEigensolver` and `ArpackEigensolver`.
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
"""
mutable struct ArpackEigensolver <: AbstractEigensolver
	nev::Union{Nothing, Int64}
	eigenvalues::Union{Nothing, Vector{ComplexF64}}

	function ArpackEigensolver()
		@warn "Initialized ArpackEigensolver with no nev. Not recommended! Use DefaultEigensolver instead."
		new(nothing)
	end
	function ArpackEigensolver(nev::Int64)
		@assert nev>0 "nev must be greater than zero!"
		new(nev)
	end
end

"""
	solve(model::NDOrderModel, solver::ArpackEigensolver)

Solves right eigenproblem using `eigs` from Arpack. 
Let `A` and `B` be the first order matrices of `model`. Then it returns the eigenpairs
```math
	(A-\\lambda_k B)y_k = 0
```
"""
function solve(model::NDOrderModel, solver::ArpackEigensolver)
	A, B = linear_first_order_matrices(model)
	FOM = size(model.linear_terms[1], 1)
	ORD = length(model.linear_terms) - 1
	if solver.nev === nothing
		solver.nev = FOM * ORD
		(values, vectors) = eigs(A, B, which = :SM)
	else
		(values, vectors) = eigs(A, B, nev = solver.nev, which = :SM)
	end

	# Reshape eigenvectors from (ORD*FOM) x (number of eigenvalues) to (FOM x ORD x number of eigenvalues)
	num_eigenvals = length(values)
	reshaped_eigenvectors = reshape(vectors, FOM, ORD, num_eigenvals)
	solver.eigenvalues = values
	return values, reshaped_eigenvectors
end

"""
	solve_left(model::NDOrderModel, solver::ArpackEigensolver)

Solves left eigenproblem using `eigs` from Arpack with σ-shift to recover the correct eigenvalues.
"""
function solve_left(model::NDOrderModel, solver::ArpackEigensolver)
	A, B = linear_first_order_matrices(model)
	A_c = complex.(A)
	B_c = complex.(B)
	A_adjoint = A_c'
	B_adjoint = B_c'
	@assert solver.nev == length(solver.eigenvalues)
	FOM = size(model.linear_terms[1], 1)
	ORD = length(model.linear_terms) - 1
	left_eigenvectors = Array{ComplexF64}(undef, FOM, ORD, solver.nev)
	eigenvalues = Vector{ComplexF64}(undef, solver.nev)
	for i in 1:(solver.nev)
		values, vectors = eigs(
			A_adjoint, B_adjoint, sigma = conj(solver.eigenvalues[i]), which = :LM, nev = 1, ncv = 30)
		left_eigenvectors[:, :, i] = reshape(vectors[:, 1], FOM, ORD)
		eigenvalues[i] = conj(values[1])
	end
	return eigenvalues, left_eigenvectors
end

"""
struct MorfeEigensolver <: AbstractEigensolver
	nev::Union{Nothing, Int64}
end

Solves shifted eigenproblem using Arpack eigs for sparse matrices. 
shift = nothing -> uses standard eigs
Computes nev eigenpairs.
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
function solve(mass::AbstractMatrix{T}, stiffness::AbstractMatrix{T}, solver::StructureModalDampingEigensolver) where {T}

	# Undamped eigensolution: solve K x = ω² M x for smallest ω².
	# which=:SM finds the smallest-magnitude eigenvalues of the pencil (stiffness, mass).
	ω2, ϕ = eigs(stiffness, mass; nev = solver.nev, which = :SM, check = 1)
	any(x -> abs(imag(x)) > 1e-12 * abs(real(x)), ω2) && error("Eigenvalues not real.")
	ω2 = real.(ω2)
	ϕ = real.(ϕ)

	FOM = size(ϕ, 1)
	CT = Complex{T}
	λ = zeros(CT, solver.nev * 2)
	eigenvectors = Array{CT}(undef, FOM, 2, solver.nev * 2)

	for i in 1:(solver.nev)
		ω2i = ω2[i]
		real_part = -0.5 * (solver.α + solver.β * ω2i)
		discriminant = real_part^2 - ω2i
		if discriminant < 0
			imag_part = sqrt(-discriminant)
			λ[2*i-1] = complex(real_part, imag_part)
			λ[2*i] = complex(real_part, -imag_part)
		else
			delta = sqrt(discriminant)
			λ[2*i-1] = complex(real_part + delta, 0.0)
			λ[2*i] = complex(real_part - delta, 0.0)
		end

		# Mass‑normalise the undamped mode: φᵀ M φ = 1.
		ϕ_i = view(ϕ, :, i)
		norm_sq = dot(ϕ_i, mass, ϕ_i)
		ϕ_i ./= sqrt(norm_sq)

		# 5. Assemble state‑space eigenvector [ φ ; λφ ]
		eigenvectors[:, 1, 2i-1] .= ϕ_i
		eigenvectors[:, 1, 2i]   .= ϕ_i
		eigenvectors[:, 2, 2i-1] .= λ[2i-1] * ϕ_i
		eigenvectors[:, 2, 2i]   .= λ[2i] * ϕ_i
	end
	return λ, eigenvectors
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
	solve_eigenproblem(model::NDOrderModel, solver::StructureModalDampingEigensolver; sorter!)

Specialised path for `StructureModalDampingEigensolver`: mass-normalization is
built into `solve`, and the physical-space left eigenmodes equal the position
block of the right eigenmodes (`X[:, ORD, k] = Y[:, 1, k]`).  No adjoint solve
or biorthogonal normalization is needed.

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
	return Eigenproblem(solver, λ, Y, Matrix{ComplexF64}(Y[:, 1, :]))
end

"""
	solve_eigenproblem(stiffness, mass, solver::StructureModalDampingEigensolver; sorter!)

Convenience overload: pass `K` and `M` directly without constructing an `NDOrderModel`.
"""
function solve_eigenproblem(
	stiffness::AbstractMatrix,
	mass::AbstractMatrix,
	solver::StructureModalDampingEigensolver;
	sorter!::Function = sort_by_magnitude!)
	λ, Y = solve(mass, stiffness, solver)
	sorter!(λ, Y)
	return Eigenproblem(solver, λ, Y, Matrix{ComplexF64}(Y[:, 1, :]))
end

"""
mutable struct Eigenproblem{T}
	solver::AbstractEigensolver
	eigenvalues::Array{Complex{T}}
	eigenmodes::Array{Complex{T}}
	left_eigenmodes::Union{Nothing, Matrix{Complex{T}}}
	master_modes::Union{Nothing, Vector{Bool}}
	external_modes::Union{Nothing, Vector{Bool}}

Stores sorted and biorthogonally normalised left/right eigenpairs of the
generalised eigenproblem `A x = λ B x`, together with a master-mode selector.
Right eigenmodes are stored as a 3D array `FOM × ORD × n_eigs`.
Left eigenmodes store only the physical-space (highest-order) slice `FOM × n_eigs`.

# Fields
- `solver`: the `AbstractEigensolver` used to compute eigenpairs
- `eigenvalues`: sorted eigenvalues `λ`
- `eigenmodes`: right eigenvectors, sorted to match `eigenvalues`
- `left_eigenmodes`: physical-space left eigenvectors (FOM × n_eigs); `nothing` until set
- `master_modes`: `Vector{Bool}` flagging master modes; `nothing` until set by a `select_master_modes_*` call
"""
mutable struct Eigenproblem{T}
	solver::AbstractEigensolver
	eigenvalues::Array{Complex{T}}
	eigenmodes::Array{Complex{T}}              # FOM × ORD × n_eigs
	left_eigenmodes::Union{Nothing, Matrix{Complex{T}}}  # FOM × n_eigs — physical-space (highest-order) slice
	master_modes::Union{Nothing, Vector{Bool}}
	external_modes::Union{Nothing, Vector{Bool}}
	# Constructor from full 3-D left eigenvectors: extracts the highest-order
	# (physical-space) slice [:, ORD, :] that MasterModeOrthogonality requires.
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
		left_phys = Matrix{Complex{T}}(left_eigenmodes[:, ORD, :])
		new{T}(solver, eigenvalues, eigenmodes, left_phys, nothing, nothing)
	end

	# Constructor accepting pre-extracted 2-D physical-space left eigenmodes
	# (FOM × n_eigs).  Use when the relevant slice is already known — e.g.
	# StructureModalDampingEigensolver, where X[:, ORD, k] == Y[:, 1, k].
	function Eigenproblem(
		solver::AbstractEigensolver,
		eigenvalues::Array{Complex{T}},
		eigenmodes::Array{Complex{T}},
		left_eigenmodes::Matrix{Complex{T}}) where {T}
		FOM = size(eigenmodes, 1)
		n_eigs = size(eigenmodes, 3)
		@assert ndims(eigenmodes) == 3 "eigenmodes must be a 3-D array (FOM × ORD × n_eigs)"
		@assert size(eigenvalues, 1) == n_eigs "length(eigenvalues) must equal size(eigenmodes, 3)"
		@assert size(left_eigenmodes) == (FOM, n_eigs) "left_eigenmodes must be FOM × n_eigs = ($FOM, $n_eigs)"
		new{T}(solver, eigenvalues, eigenmodes, left_eigenmodes, nothing, nothing)
	end
end

"""
	solve_eigenproblem(
		model::NDOrderModel;
		solver::AbstractEigensolver = DefaultEigensolver(),
		sorter::Function = sort_by_magnitude,
		normalizer::Function = normalize_biorthogonal)

Computes left and right eigenpairs of the problem described in `model` by using the defined `solver`.
Additionally `sorter` sorts the eigenpairs and `normalizer` is used to normalize the eigenmodes.
"""
function solve_eigenproblem(
	model::NDOrderModel;
	solver::AbstractEigensolver = DefaultEigensolver(),
	sorter!::Function = sort_by_magnitude!,
	normalizer!::Function = normalize_biorthogonal!)

	#calculate right eigenmodes
	(eigenvalues, eigenmodes) = solve(model, solver)

	#sort eigenpairs
	sorter!(eigenvalues, eigenmodes)

	#calculate left eigenmodes
	(left_eigenvalues, left_eigenmodes) = solve_left(model, solver)
	left_eigenvalues, left_eigenmodes = sort_left_eigenmodes(
		eigenvalues, left_eigenvalues, left_eigenmodes)

	# normalize eigenpairs
	normalizer!(model, eigenmodes, left_eigenmodes)

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
			println("Eigenvalues", eigenvalues)
			println("Left eigenvalues ", left_eigenvalues)
		end

		perm[i] = j
		used[j] = true
	end
	return left_eigenvalues[perm], left_eigenmodes[:, :, perm]
end

"""
	normalize_biorthogonal!(
		model::NDOrderModel,
		eigenmodes::Matrix{T},
		left_eigenmodes::Matrix{T})

Normalize eigenmodes to fulfill the equation
	x_i^H * B * y_j = δ_{ij}
"""
function normalize_biorthogonal!(
	model::NDOrderModel,
	eigenmodes::Array{T},
	left_eigenmodes::Array{T}) where {T}
	@assert size(eigenmodes)==size(left_eigenmodes) "Size of left and right eigenmodes must be the same!"

	(_, B) = linear_first_order_matrices(model)
	for i in 1:size(eigenmodes, 3)
		tmp = vec(left_eigenmodes[:, :, i])' * B * vec(eigenmodes[:, :, i])
		@views eigenmodes[:, :, i] ./= tmp
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
	@assert length(mastermodes)==size(ep.eigenmodes, 3) "mastermodes has wrong length!"
	ep.master_modes = mastermodes
end

"""
	select_master_modes_by_sorting(ep::Eigenproblem, nev::Int64)

Defines master_modes as the first nev eigenpairs. Sorting was done in solve_eigenproblem.
"""
function select_master_modes_by_sorting(ep::Eigenproblem, nev::Int64)
	@assert nev>0 "nev must be bigger then zero"
	n = size(ep.eigenmodes, 3)
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
