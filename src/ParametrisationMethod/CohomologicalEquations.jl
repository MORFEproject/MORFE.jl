"""
	CohomologicalEquations

Solve the cohomological equations that arise in the parametrisation method for
computing Spectral Submanifolds (SSMs) of high-dimensional dynamical systems.

---

# Problem statement

Given a full‑order model of order `ORD` in `FOM` degrees of freedom, the
parametrisation method seeks a change of coordinates

```
x(t) = W(z(t)),          z ∈ ℂᴺᵛᵃʳ
```

such that the reduced dynamics `ż = R(z)` is simpler.  Expanding both `W` and
`R` as dense polynomials in the `NVAR = ROM + N_EXT` reduced variables (ROM
master‑mode amplitudes plus N_EXT external forcing amplitudes) and matching
coefficients monomial by monomial yields the **cohomological equations**: one
linear system per multi‑index `α`.

---

# System structure for multi‑index `α`

Let `s = ⟨λ, α⟩` be the superharmonic frequency and let `nR = |ℛ(α)|` be the
number of master modes resonant at `α`.  The cohomological linear system is

```
┌              ┐ ┌         ┐   ┌           ┐
│  L(s)  C(s)  │ │  W[α]   │ = │  RHS_inv  │   FOM rows  (invariance)
│  L̂(s)  Ĉ(s)  │ │  R_res  │   │  RHS_ort  │   nR  rows  (orthogonality)
└              ┘ └         ┘   └           ┘
```

where:
- `L(s)`  (`FOM × FOM`) is the parametrisation operator,
- `C(s)`  (`FOM × nR`) acts on the unknown resonant reduced-dynamics coefficients,
- `L̂(s)` (`nR × FOM`) is the orthogonality row operator,
- `Ĉ(s)` (`nR × nR`) is the orthogonality joint operator,
- `W[α] ∈ ℂᶠᵒᵐ` is the (zeroth-order) parametrisation coefficient,
- `R_res ∈ ℂⁿᴿ` are the reduced-dynamics coefficients for resonant modes.

Non-resonant master modes are trivially zero and are excluded from the system.
External forcing modes are *known* and appear only on the right-hand side.

---

# Module contents

| Symbol | Description |
|:-------|:------------|
| [`CohomologicalContext`](@ref) | Single flat struct bundling all precomputed operators, the external dynamics matrix, and the resonance set |
| [`solve_single_monomial!`](@ref) | Solve the cohomological system for one multi-index |
| [`solve_cohomological_equations!`](@ref) | Solve for all multi-indices in causal (ascending-degree) order |
| [`solve_cohomological_problem`](@ref) | High-level driver: precompute everything and solve |
"""
module CohomologicalEquations

using ..Multiindices: MultiindexSet
using ..ParametrisationMethod: Parametrisation, ReducedDynamics,
	create_parametrisation_method_objects, compute_higher_derivative_coefficients!,
	multiindex_set
using ..LowerOrderCouplings: compute_lower_order_couplings
using ..InvarianceEquation: assemble_cohomological_matrix_and_rhs,
	precompute_column_polynomials
using ..MasterModeOrthogonality: assemble_orthogonality_matrix_and_rhs,
	precompute_orthogonality_operator_coefficients,
	precompute_orthogonality_column_polynomials
using ..FullOrderModel: NDOrderModel
using ..MultilinearTerms: compute_multilinear_terms
using ..Resonance: ResonanceSet, is_resonant
using LinearAlgebra
using StaticArrays: SVector

export CohomologicalContext,
	solve_cohomological_equations!,
	solve_single_monomial!,
	solve_cohomological_problem

# ==============================================================================
# 1.  Context struct
# ==============================================================================

"""
	CohomologicalContext{T, ORD, ORDP1, NVAR, ROM, FOM}

A single flat struct that bundles **all** precomputed data required to solve the
cohomological equations for every monomial in the parametrisation method.

Using one flat struct rather than nested containers eliminates the name collision
between the invariance‑equation operators and the orthogonality‑condition
operators (both previously called `C_coeffs`/`E_coeffs`) and makes the data
provenance explicit at every call site.

# Type parameters

| Parameter | Meaning |
|:----------|:--------|
| `T`       | Scalar type (typically `ComplexF64`) |
| `ORD`     | Polynomial order of the full‑order ODE |
| `ORDP1`   | `ORD + 1` — required as a separate parameter because Julia's type system does not support arithmetic on type parameters |
| `NVAR`    | Total number of reduced variables: `ROM + N_EXT` |
| `ROM`     | Number of master modes (dimension of the reduced model) |
| `FOM`     | Full‑order state dimension |

# Fields

## Full‑order linear operators

- `linear_terms :: NTuple{ORDP1, Matrix{T}}` —
  coefficient matrices of the linear part of the full‑order model,
  `(B₀, B₁, …, B_ORD)`, each of size `FOM × FOM`.

- `generalised_eigenmodes :: Matrix{T}` — size `FOM × NVAR`.
  Right generalised eigenvectors; columns `1:ROM` are the master modes,
  columns `ROM+1:NVAR` are the external forcing modes.

- `reduced_dynamics_linear :: Matrix{T}` — size `NVAR × NVAR`.
  Jordan normal form of the linearisation; its diagonal entries `λᵢ` are used
  to form the superharmonic frequency `s = ⟨λ, α⟩` for multi‑index `α`.

## Invariance‑equation operators (polynomial coefficients in `s`)

These arrays are produced by
[`InvarianceEquation.precompute_column_polynomials`](@ref).

- `invariance_C_coeffs :: Vector{Matrix{T}}` — length `ROM`.
  `invariance_C_coeffs[r][:, j]` is the degree‑`(j-1)` coefficient of the
  column operator `C_r(s)` acting on the reduced‑dynamics unknown for master
  mode `r`; each matrix has size `FOM × ORD`.

- `invariance_E_coeffs :: Vector{Matrix{T}}` — length `N_EXT`.
  `invariance_E_coeffs[e][:, j]` is the degree‑`(j-1)` coefficient of the
  external‑forcing operator `E_e(s)` for external variable `e`; each matrix
  has size `FOM × ORD`.

## Orthogonality‑condition operators (polynomial coefficients in `s`)

These arrays are produced by
[`MasterModeOrthogonality.precompute_orthogonality_operator_coefficients`](@ref)
and
[`MasterModeOrthogonality.precompute_orthogonality_column_polynomials`](@ref).

- `orthogonality_J_coeffs :: NTuple{ROM, Matrix{T}}` — length `ROM`.
  `orthogonality_J_coeffs[r][j, :]` is the degree‑`(j-1)` coefficient of the
  left row operator `L_r(s)` for master mode `r`; each matrix has size
  `ORD × FOM`.

- `orthogonality_C_coeffs :: Vector{Matrix{T}}` — length `ROM`.
  `orthogonality_C_coeffs[r][j, :]` is the degree‑`(j-1)` coefficient of the
  joint column operator `Ĉ_r(s)` acting on the master‑mode unknowns; each
  matrix has size `(ORD-1) × ROM`.

- `orthogonality_E_coeffs :: Vector{Matrix{T}}` — length `ROM`.
  `orthogonality_E_coeffs[r][j, :]` is the degree‑`(j-1)` coefficient of the
  joint column operator `Ê_r(s)` acting on the external variables; each matrix
  has size `(ORD-1) × N_EXT`.

## External dynamics

- `external_dynamics_matrix :: Matrix{T}` — size `N_EXT × L`
  (L = total number of monomials).
  `external_dynamics_matrix[e, idx]` stores the *known* coefficient of monomial
  `idx` in the reduced dynamics of external variable `e`.  For a linear external
  system (harmonic forcing) this matrix is typically sparse: only the entry
  corresponding to the linear monomial `e_{ROM+e}` is non‑zero, with value
  `λ_{ROM+e}`.  The matrix is initialised by
  [`solve_cohomological_problem`](@ref).

## Resonance

- `resonance_set :: ResonanceSet` — look‑up table indicating which master modes
  are resonant with each monomial.
"""
struct CohomologicalContext{T, ORD, ORDP1, NVAR, ROM, FOM}
	# ── Full-order linear operators ────────────────────────────────────────────
	linear_terms::NTuple{ORDP1, Matrix{T}}
	generalised_eigenmodes::Matrix{T}
	reduced_dynamics_linear::Matrix{T}
	# ── Invariance-equation operators ─────────────────────────────────────────
	invariance_C_coeffs::Vector{Matrix{T}}   # length ROM,   each FOM × ORD
	invariance_E_coeffs::Vector{Matrix{T}}   # length N_EXT, each FOM × ORD
	# ── Orthogonality-condition operators ─────────────────────────────────────
	orthogonality_J_coeffs::NTuple{ROM, Matrix{T}}   # length ROM, each ORD × FOM
	orthogonality_C_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × ROM
	orthogonality_E_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × N_EXT
	# ── Known external dynamics ────────────────────────────────────────────────
	external_dynamics_matrix::Matrix{T}          # N_EXT × L
	# ── Resonance set ──────────────────────────────────────────────────────────
	resonance_set::ResonanceSet
end

# ==============================================================================
# 2.  Auxiliary helpers (module-private)
# ==============================================================================

# Return the positions in `mset` of ALL unit-vector monomials eᵣ for r = 1 … NVAR.
# These are all initialised (master modes from eigenvectors, external modes from the
# linear cohomological solve) and are skipped during the main solve loop.
function _linear_monomial_indices(mset::MultiindexSet{NVAR}) where {NVAR}
	indices = Int[]
	for r in 1:NVAR
		e_r = zeros(Int, NVAR)
		e_r[r] = 1
		idx = findfirst(==(e_r), mset.exponents)
		idx !== nothing && push!(indices, idx)
	end
	return indices
end

# Return a compile-time-sized SVector{ROM, Bool} indicating which master modes
# r = 1 … ROM are resonant with the monomial at position `monomial_idx` in the
# multiindex set.  Using Val(ROM) makes the SVector size a compile-time constant.
@inline function _resonance_vector(
	resonance_set::ResonanceSet,
	monomial_idx::Int,
	::Val{ROM},
) where {ROM}
	return SVector{ROM, Bool}(ntuple(r -> is_resonant(resonance_set, monomial_idx, r), Val(ROM)))
end

# ==============================================================================
# 3.  Solve a single monomial
# ==============================================================================

"""
	solve_single_monomial!(W, R, idx, ctx, model) -> nothing

Solve the cohomological equations for the monomial with multiindex‑set position
`idx`, updating the coefficients of `W` and `R` in‑place.

## Algorithm

1. Compute the superharmonic frequency `s = ⟨λ, α⟩` from the diagonal of
   `ctx.reduced_dynamics_linear`.
2. Look up the resonance pattern `resonance ∈ {true,false}^{ROM}` from
   `ctx.resonance_set`.
3. Compute lower‑order coupling vectors `ξ[j]` (length `FOM`) via
   [`LowerOrderCouplings.compute_lower_order_couplings`](@ref).
4. Compute the nonlinear model RHS via
   [`MultilinearTerms.compute_multilinear_terms`](@ref).  This evaluates all
   nonlinear terms of the full‑order model at the current monomial using
   already‑solved lower‑order parametrisation coefficients.
5. Retrieve the known external dynamics at this monomial from
   `ctx.external_dynamics_matrix[:, idx]`.
6. Assemble and solve the stacked `(FOM + nR) × (FOM + nR)` linear system using
   [`InvarianceEquation.assemble_cohomological_matrix_and_rhs`](@ref) and
   [`MasterModeOrthogonality.assemble_orthogonality_matrix_and_rhs`](@ref).
7. Store `W[α]` (zeroth time‑derivative) and the resonant reduced‑dynamics
   coefficients `R_res`.
8. Compute higher time‑derivative coefficients `W^(j)[α]` for `j = 1 … ORD-1`
   via [`ParametrisationMethod.compute_higher_derivative_coefficients!`](@ref).

## Arguments

- `W :: Parametrisation{ORD, NVAR, T}` — parametrisation object (updated in‑place).
- `R :: ReducedDynamics{ROM, NVAR, T}` — reduced dynamics object (updated in‑place).
- `idx :: Int` — position of the target monomial in the shared multiindex set.
- `ctx :: CohomologicalContext{T, ORD, ORDP1, NVAR, ROM, FOM}` — all precomputed data.
- `model :: NDOrderModel` — full‑order model; provides nonlinear term evaluations.
"""
function solve_single_monomial!(
	W::Parametrisation{ORD, NVAR, T},
	R::ReducedDynamics{ROM, NVAR, T},
	idx::Int,
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, ROM, FOM},
	model::NDOrderModel,
) where {ORD, NVAR, T, ROM, FOM, ORDP1}

	multi = multiindex_set(W)[idx]

	# ── 1. Superharmonic frequency s = ⟨λ, α⟩ ────────────────────────────────
	s = sum(multi[i] * ctx.reduced_dynamics_linear[i, i] for i in 1:NVAR)

	# ── 2. Resonance bitmask ──────────────────────────────────────────────────
	resonance = _resonance_vector(ctx.resonance_set, idx, Val(ROM))

	# ── 3. Lower-order coupling vectors ξ[j] (length FOM each) ───────────────
	lower_order_couplings = compute_lower_order_couplings(multi, W, R)

	# ── 4. Nonlinear model terms at this monomial ─────────────────────────────
	# Evaluates all nonlinear terms of the full-order model using lower-order W
	# coefficients (already solved).  Added directly to the invariance RHS.
	nonlinear_rhs = T.(compute_multilinear_terms(model, multi, W))

	# ── 5. Known external dynamics at this monomial ───────────────────────────
	external_dynamics = view(ctx.external_dynamics_matrix, :, idx)

	# ── 6. Assemble and solve the stacked cohomological system ────────────────
	#   invariance block:    M_inv  (FOM × FOM+nR),  rhs_inv  (FOM)
	#   orthogonality block: M_orth (nR  × FOM+nR),  rhs_orth (nR)
	#   stacked:             (FOM+nR) × (FOM+nR)  →  square system
	M_inv, rhs_inv = assemble_cohomological_matrix_and_rhs(
		s,
		ctx.linear_terms,
		ctx.invariance_C_coeffs,
		ctx.invariance_E_coeffs,
		resonance,
		lower_order_couplings,
		external_dynamics,
	)
	rhs_inv .+= nonlinear_rhs

	M_orth, rhs_orth = assemble_orthogonality_matrix_and_rhs(
		s,
		ctx.orthogonality_J_coeffs,
		ctx.orthogonality_C_coeffs,
		ctx.orthogonality_E_coeffs,
		resonance,
		lower_order_couplings,
		external_dynamics,
	)

	sol = [M_inv; M_orth] \ [rhs_inv; rhs_orth]

	# ── 6a. Parametrisation coefficients (zeroth time-derivative order) ───────
	W.poly.coefficients[:, 1, idx] .= view(sol, 1:FOM)

	# ── 6b. Reduced dynamics — non-zero only for resonant master modes ─────────
	rr = 1
	for r in 1:ROM
		if resonance[r]
			R.poly.coefficients[r, idx] = sol[FOM + rr]
			rr += 1
		else
			R.poly.coefficients[r, idx] = zero(T)
		end
	end

	# ── 7. Higher time-derivative coefficients W^(j)[α], j = 1 … ORD-1 ───────
	compute_higher_derivative_coefficients!(
		W.poly.coefficients,
		R.poly.coefficients,
		external_dynamics,
		s,
		idx,
		ctx.generalised_eigenmodes,
		lower_order_couplings,
	)

	return nothing
end

# ==============================================================================
# 4.  Solve all monomials
# ==============================================================================

"""
	solve_cohomological_equations!(W, R, ctx, model) -> nothing

Solve the cohomological equations for **all** monomials in the multiindex set
of `W` and `R`, processing them in *causal order* (ascending total degree so
that lower‑order coefficients are available when higher‑order ones are solved).

All unit‑vector monomials `eᵣ` for `r = 1 … NVAR` (both master modes and
external forcing modes) are assumed to have been initialised to the spectral
data beforehand and are skipped.

## Arguments

- `W :: Parametrisation{ORD, NVAR, T}` — parametrisation (updated in‑place).
- `R :: ReducedDynamics{ROM, NVAR, T}` — reduced dynamics (updated in‑place).
- `ctx :: CohomologicalContext` — all precomputed data and the resonance set.
- `model :: NDOrderModel` — full‑order model; passed through to
  [`solve_single_monomial!`](@ref) for nonlinear term evaluation.
"""
function solve_cohomological_equations!(
	W::Parametrisation{ORD, NVAR, T},
	R::ReducedDynamics{ROM, NVAR, T},
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, ROM, FOM},
	model::NDOrderModel,
) where {ORD, NVAR, T, ROM, FOM, ORDP1}

	mset    = multiindex_set(W)
	nterms  = length(mset)

	# Causal ordering: ascending total degree so lower-order data is ready first
	indices_by_order = sort(1:nterms, by = i -> sum(mset[i]))

	# All unit-vector monomials (master modes and external forcing) are already
	# initialised before this loop and are skipped.
	skip_set = Set(_linear_monomial_indices(mset))

	for idx in indices_by_order
		idx in skip_set && continue
		solve_single_monomial!(W, R, idx, ctx, model)
	end

	return nothing
end

# ==============================================================================
# 5.  High-level driver
# ==============================================================================

"""
	solve_cohomological_problem(
		model, mset, master_eigenvalues,
		master_modes, left_eigenmodes, resonance_set;
		initial_W = nothing, initial_R = nothing
	) -> (W, R)

High‑level driver that assembles a [`CohomologicalContext`](@ref) from raw
spectral data and solves the full set of cohomological equations.

External eigenvalues are read directly from `model.external_system.eigenvalues`
(or treated as absent when `model.external_system === nothing`).  The
linear‑operator tuple is read from `model.linear_terms`.

## Steps

1. Extract external eigenvalues from `model.external_system` and build the
   Jordan matrix `Λ`.
2. Create (or reuse) the [`Parametrisation`](@ref) `W` and
   [`ReducedDynamics`](@ref) `R` objects and initialise the master‑mode
   linear monomials from `master_modes`.
3. Solve the linear cohomological equations for each external forcing direction:
   `L(s_ext) · W_ext = compute_multilinear_terms(model, e_ext, W)`.
   This yields the particular solution at the forcing frequency and populates
   the external columns of `generalised_right_eigenmodes`.
4. Build the `N_EXT × L` external‑dynamics matrix (non‑zero only at the linear
   forcing monomials `e_{ROM+e}`).
5. Precompute the invariance‑equation operator columns
   (`invariance_C_coeffs`, `invariance_E_coeffs`) via
   [`InvarianceEquation.precompute_column_polynomials`](@ref).
6. Precompute the orthogonality‑condition operators
   (`orthogonality_J_coeffs`, `orthogonality_C_coeffs`, `orthogonality_E_coeffs`)
   via
   [`MasterModeOrthogonality.precompute_orthogonality_operator_coefficients`](@ref)
   and
   [`MasterModeOrthogonality.precompute_orthogonality_column_polynomials`](@ref).
7. Assemble a [`CohomologicalContext`](@ref) and call
   [`solve_cohomological_equations!`](@ref).

## Arguments

- `model :: NDOrderModel` — full‑order model.  `model.linear_terms` provides
  `(B₀, B₁, …, B_ORD)`; `model.external_system` (may be `nothing`) carries
  the external eigenvalues.
- `mset :: MultiindexSet{NVAR}` — multiindex set over all `NVAR` reduced
  variables (`NVAR = ROM + N_EXT`).
- `master_eigenvalues :: SVector{ROM, ComplexF64}` — eigenvalues of the master
  modes.
- `master_modes :: Matrix{ComplexF64}` — size `FOM × ROM`; right eigenvectors
  of the master modes.  The external forcing directions are derived internally
  by solving the linear cohomological equations.
- `left_eigenmodes :: SVector{ROM, Vector{ComplexF64}}` — left eigenvectors of
  the master modes (used in the orthogonality conditions).
- `resonance_set :: ResonanceSet` — precomputed resonance look‑up table.
- `initial_W`, `initial_R` — optionally supply already-initialised objects
  (their linear monomials must be set correctly).

## Returns

`(W, R)` — the solved [`Parametrisation`](@ref) and [`ReducedDynamics`](@ref).
"""
function solve_cohomological_problem(
	model::NDOrderModel,
	mset::MultiindexSet{NVAR},
	master_eigenvalues::SVector{ROM, ComplexF64},
	master_modes::Matrix{ComplexF64},
	left_eigenmodes::SVector{ROM, Vector{ComplexF64}},
	resonance_set::ResonanceSet;
	initial_W::Union{Nothing, Parametrisation} = nothing,
	initial_R::Union{Nothing, ReducedDynamics}  = nothing,
) where {NVAR, ROM}

	# ── 1. Dimensions and eigenvalue data ─────────────────────────────────────
	# External eigenvalues come from the model's ExternalSystem (if present).
	external_eigenvalues = model.external_system === nothing ?
		ComplexF64[] :
		Vector{ComplexF64}(model.external_system.eigenvalues)
	N_EXT = length(external_eigenvalues)

	@assert NVAR == ROM + N_EXT "Multiindex set has $NVAR variables but ROM + N_EXT = $(ROM + N_EXT)"
	FOM = size(master_modes, 1)
	@assert size(master_modes, 2) == ROM "master_modes must have $ROM columns"

	ORDP1 = length(model.linear_terms)
	ORD   = ORDP1 - 1

	# Promote linear_terms to ComplexF64 so they are compatible with the complex
	# eigenvectors and Jordan matrix used throughout the cohomological equations.
	linear_terms = ntuple(i -> ComplexF64.(model.linear_terms[i]), Val(ORDP1))

	# Jordan matrix Λ: diagonal entries are master and external eigenvalues
	Λ = zeros(ComplexF64, NVAR, NVAR)
	for r in 1:ROM
		Λ[r, r] = master_eigenvalues[r]
	end
	for e in 1:N_EXT
		Λ[ROM + e, ROM + e] = external_eigenvalues[e]
	end

	# ── 2. Parametrisation and reduced-dynamics objects ────────────────────────
	if initial_W !== nothing && initial_R !== nothing
		W = initial_W
		R = initial_R
	else
		W, R = create_parametrisation_method_objects(
			mset, ORD, FOM, ROM, N_EXT, ComplexF64,
		)
		# Initialise linear monomials for master modes
		for r in 1:ROM
			e_r = SVector{NVAR, Int}(ntuple(i -> i == r ? 1 : 0, Val(NVAR)))
			idx = findfirst(==(e_r), mset.exponents)
			idx === nothing && continue
			W.poly.coefficients[:, 1, idx] .= view(master_modes, :, r)
			for k in 2:ORD
				W.poly.coefficients[:, k, idx] .= view(master_modes, :, r) .* (Λ[r, r]^(k - 1))
			end
			R.poly.coefficients[r, idx] = Λ[r, r]
		end

		# ── 2b. Solve external forcing directions ──────────────────────────────
		# For each external mode e, the external direction Φ_ext[:, e] = W[e_ext]
		# is the particular solution of the linear cohomological equation
		#
		#   L(s_ext) · W_ext = compute_multilinear_terms(model, e_ext, W)
		#
		# where s_ext = λ_{ROM+e} and W currently has only the master-mode linear
		# monomials filled in.  This correctly derives the spatial distribution
		# of the forcing response from the model rather than prescribing it externally.
		external_directions = zeros(ComplexF64, FOM, N_EXT)
		for e in 1:N_EXT
			s_ext = external_eigenvalues[e]
			# Build L(s_ext) = B₀ + s·B₁ + … + s^ORD·B_ORD via Horner
			L_sext = copy(linear_terms[ORDP1])  # start: B_ORD
			for k in (ORDP1 - 1):-1:1
				L_sext .= s_ext .* L_sext .+ linear_terms[k]
			end
			e_ext_svec = SVector{NVAR, Int}(ntuple(i -> i == ROM + e ? 1 : 0, Val(NVAR)))
			rhs_ext = ComplexF64.(compute_multilinear_terms(model, e_ext_svec, W))
			W_ext = L_sext \ rhs_ext
			external_directions[:, e] .= W_ext
			# Store in W (zeroth and higher-derivative orders at e_ext)
			idx_ext = findfirst(==(e_ext_svec), mset.exponents)
			if idx_ext !== nothing
				W.poly.coefficients[:, 1, idx_ext] .= W_ext
				for k in 2:ORD
					W.poly.coefficients[:, k, idx_ext] .= W_ext .* (s_ext^(k - 1))
				end
			end
		end
	end

	# ── 2c. Build generalised_right_eigenmodes = [master_modes | external_directions]
	# When initial_W/R were provided we recover the external columns from W.
	if initial_W !== nothing && initial_R !== nothing
		external_directions = zeros(ComplexF64, FOM, N_EXT)
		for e in 1:N_EXT
			e_ext_svec = SVector{NVAR, Int}(ntuple(i -> i == ROM + e ? 1 : 0, Val(NVAR)))
			idx_ext = findfirst(==(e_ext_svec), mset.exponents)
			idx_ext !== nothing && (external_directions[:, e] .= W.poly.coefficients[:, 1, idx_ext])
		end
	end
	generalised_right_eigenmodes = hcat(master_modes, external_directions)

	# ── 3. External-dynamics matrix (N_EXT × L) ───────────────────────────────
	# For a linear external system the only non-zero entries are at the linear
	# monomials e_{ROM+e}, where the coefficient equals the eigenvalue λ_{ROM+e}.
	L = length(mset)
	external_dynamics_matrix = zeros(ComplexF64, N_EXT, L)
	for e in 1:N_EXT
		e_ext = SVector{NVAR, Int}(ntuple(i -> i == ROM + e ? 1 : 0, Val(NVAR)))
		idx   = findfirst(==(e_ext), mset.exponents)
		idx !== nothing && (external_dynamics_matrix[e, idx] = Λ[ROM + e, ROM + e])
	end

	# ── 4. Invariance-equation operator columns ────────────────────────────────
	invariance_C_coeffs, invariance_E_coeffs = precompute_column_polynomials(
		linear_terms, generalised_right_eigenmodes, Λ, ROM,
	)

	# ── 5. Orthogonality-condition operators ───────────────────────────────────
	orthogonality_J_coeffs = precompute_orthogonality_operator_coefficients(
		linear_terms, left_eigenmodes, master_eigenvalues,
	)
	orthogonality_C_coeffs, orthogonality_E_coeffs = precompute_orthogonality_column_polynomials(
		orthogonality_J_coeffs, generalised_right_eigenmodes, Λ,
	)

	# ── 6. Build context and solve ─────────────────────────────────────────────
	ctx = CohomologicalContext{ComplexF64, ORD, ORDP1, NVAR, ROM, FOM}(
		linear_terms,
		generalised_right_eigenmodes,
		Λ,
		invariance_C_coeffs,
		invariance_E_coeffs,
		orthogonality_J_coeffs,
		orthogonality_C_coeffs,
		orthogonality_E_coeffs,
		external_dynamics_matrix,
		resonance_set,
	)

	solve_cohomological_equations!(W, R, ctx, model)

	return W, R
end

end # module CohomologicalEquations
