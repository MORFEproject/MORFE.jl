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
| [`CohomologicalContext`](@ref) | Flat struct bundling all precomputed operators, pre-allocated buffers, and the resonance set |
| [`solve_single_monomial!`](@ref) | Solve the cohomological system for one multi-index |
| [`solve_cohomological_equations!`](@ref) | Solve for all multi-indices in causal (ascending-degree) order |
| [`solve_cohomological_problem`](@ref) | High-level driver: precompute everything and solve |
"""
module CohomologicalEquations

using ..Multiindices: MultiindexSet, indices_in_box_with_bounded_degree, build_exponent_index_map
using ..ParametrisationMethod: Parametrisation, ReducedDynamics,
	create_parametrisation_method_objects, compute_higher_derivative_coefficients!,
	multiindex_set
using ..LowerOrderCouplings: compute_lower_order_couplings
using ..InvarianceEquation: assemble_cohomological_matrix_and_rhs,
	precompute_column_polynomials,
	precompute_master_column_polynomials,
	precompute_external_column_polynomials
using ..MasterModeOrthogonality: assemble_orthogonality_matrix_and_rhs,
	precompute_orthogonality_operator_coefficients,
	precompute_orthogonality_column_polynomials
using ..FullOrderModel: NDOrderModel
using ..ExternalSystems: ExternalSystem
using ..MultilinearTerms: compute_multilinear_terms, build_multilinear_terms_cache, MultilinearTermsCache
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

- `lambda_diag :: Vector{T}` — length `NVAR`.
  Diagonal entries `λᵢ` of the Jordan matrix, used to form the superharmonic
  frequency `s = ⟨λ, α⟩` for multi‑index `α`.  These are read directly from
  the reduced‑dynamics polynomial `R` at the linear monomials `e_i`.

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

- `orthogonality_J_coeffs :: Vector{Matrix{T}}` — length `ROM`.
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

## Resonance

- `resonance_set :: ResonanceSet` — look‑up table indicating which master modes
  are resonant with each monomial.

## Precomputed bookkeeping

- `linear_monomial_skip_set :: Set{Int}` — indices of all unit-vector monomials
  (`e_1, …, e_NVAR`); these are initialised before the main loop and skipped.
"""
struct CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT}
	# ── Full-order linear operators ────────────────────────────────────────────
	linear_terms::NTuple{ORDP1, Matrix{LT}}
	generalised_eigenmodes::Matrix{T}
	lambda_diag::Vector{T}                       # length NVAR; diagonal of Λ from R
	# ── Invariance-equation operators ─────────────────────────────────────────
	invariance_C_coeffs::Vector{Matrix{T}}   # length ROM,   each FOM × ORD
	invariance_E_coeffs::Vector{Matrix{T}}   # length N_EXT, each FOM × ORD
	# ── Orthogonality-condition operators ─────────────────────────────────────
	orthogonality_J_coeffs::Vector{Matrix{T}}   # length ROM, each ORD × FOM
	orthogonality_C_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × ROM
	orthogonality_E_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × N_EXT
	# ── Resonance set ──────────────────────────────────────────────────────────
	resonance_set::ResonanceSet
	# ── Skip set (linear monomials, pre-computed) ──────────────────────────────
	linear_monomial_skip_set::Set{Int}
	# ── Lower-order coupling pre-allocated resources ────────────────────────────
	multiindex_dict::Dict{SVector{NVAR, Int}, Int}
	lower_order_buffer::Vector{Vector{T}}              # length ORD, each FOM; zeroed before each call
	candidate_indices_by_monomial::Vector{Vector{Int}} # length L; candidates for _sum_higher_degree_terms!
	# ── Stacked system pre-allocated buffers ────────────────────────────────────
	system_matrix_buffer::Matrix{T}   # (FOM+ROM) × (FOM+ROM) worst-case; filled in-place each call
	rhs_buffer::Vector{T}             # length FOM+ROM; holds rhs, then solution after ldiv!
	external_rhs_buffer::Vector{T}    # length FOM; scratch buffer for evaluate_external_rhs!
end

# ==============================================================================
# 2.  Auxiliary helpers (module-private)
# ==============================================================================

# Copy the coefficients of an N_EXT-variable external polynomial into the last
# N_EXT rows of R's coefficient matrix.  Each monomial α_ext in the external
# polynomial is embedded into the NVAR-variable monomial space by prepending ROM
# zero entries:  α_full = (0,…,0, α_ext[1],…,α_ext[N_EXT]).
# The cohomological equations never modify these rows (external modes are not
# resonant), so placing them here once is sufficient.
function _embed_external_dynamics!(
	R::ReducedDynamics{ROM, NVAR, T},
	ext_poly,
	mset::MultiindexSet{NVAR},
) where {ROM, NVAR, T}
	N_EXT = NVAR - ROM
	N_EXT > 0 || return nothing
	ext_coeffs = ext_poly.coefficients
	for (j, α_ext) in enumerate(ext_poly.multiindex_set.exponents)
		α_full = SVector{NVAR, Int}(ntuple(i -> i <= ROM ? 0 : α_ext[i - ROM], Val(NVAR)))
		idx_full = findfirst(==(α_full), mset.exponents)
		idx_full === nothing && continue
		for e in 1:N_EXT
			coeff = T(ext_coeffs[e, j])
			iszero(coeff) || (R.poly.coefficients[ROM + e, idx_full] = coeff)
		end
	end
	return nothing
end

# Return the positions in `mset` of ALL unit-vector monomials eᵣ for r = 1 … NVAR.
# In GrLex order the zero vector (if present) occupies index 1, so eᵣ is at index
# r (no zero vector) or r+1 (zero vector included).  Searching only the first NVAR+1
# entries is sufficient.
function _linear_monomial_indices(mset::MultiindexSet{NVAR}) where {NVAR}
	indices = Int[]
	n_search = min(NVAR + 1, length(mset))
	for r in 1:NVAR
		e_r = SVector{NVAR, Int}(ntuple(i -> i == r ? 1 : 0, Val(NVAR)))
		idx = findfirst(==(e_r), view(mset.exponents, 1:n_search))
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
5. Retrieve the known external dynamics at this monomial from the last
   `N_EXT` rows of `R.poly.coefficients[:, idx]`.
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
- `ctx :: CohomologicalContext{T, ORD, ORDP1, NVAR, FOM}` — all precomputed data.
- `model :: NDOrderModel` — full‑order model; provides nonlinear term evaluations.
- `ml_cache :: MultilinearTermsCache` — precomputed factorisation cache; built once before the solve loop.
"""
function solve_single_monomial!(
	W::Parametrisation{ORD, NVAR, T},
	R::ReducedDynamics{ROM, NVAR, T},
	idx::Int,
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT},
	model::NDOrderModel,
	ml_cache::MultilinearTermsCache,
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT}

	multi = multiindex_set(W)[idx]

	# ── 1. Superharmonic frequency s = ⟨λ, α⟩ ────────────────────────────────
	s = sum(multi[i] * ctx.lambda_diag[i] for i in 1:NVAR)

	# ── 2. Resonance bitmask ──────────────────────────────────────────────────
	resonance = _resonance_vector(ctx.resonance_set, idx, Val(ROM))

	# ── 3. Lower-order coupling vectors ξ[j] (length FOM each) ───────────────
	for v in ctx.lower_order_buffer
		;
		fill!(v, zero(T));
	end
	lower_order_couplings = compute_lower_order_couplings(
		multi, W, R,
		ctx.multiindex_dict,
		ctx.lower_order_buffer,
		ctx.candidate_indices_by_monomial[idx],
	)

	# ── 4. Nonlinear model terms at this monomial ─────────────────────────────
	# Evaluates all nonlinear terms of the full-order model using lower-order W
	# coefficients (already solved).  Added directly to the invariance RHS.
	nonlinear_rhs = compute_multilinear_terms(model, idx, W, ml_cache)

	# ── 5. Known external dynamics at this monomial ───────────────────────────
	# External dynamics live in the last N_EXT rows of R (rows ROM+1:NVAR).
	external_dynamics = view(R.poly.coefficients, (ROM+1):NVAR, idx)

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
		ctx.external_rhs_buffer,
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

	# Write the stacked (FOM+nR) × (FOM+nR) system into the pre-allocated buffer
	# (no vcat allocation) and solve in-place with lu! + ldiv!.
	nR = count(resonance)
	n_sys = FOM + nR
	ctx.system_matrix_buffer[1:FOM, 1:n_sys] .= M_inv
	ctx.system_matrix_buffer[(FOM+1):n_sys, 1:n_sys] .= M_orth
	ctx.rhs_buffer[1:FOM] .= rhs_inv
	ctx.rhs_buffer[(FOM+1):n_sys] .= rhs_orth
	# Copy the active submatrix (1 alloc), factorize in-place, solve in-place.
	A_sys = ctx.system_matrix_buffer[1:n_sys, 1:n_sys]
	ldiv!(lu!(A_sys), view(ctx.rhs_buffer, 1:n_sys))
	sol = view(ctx.rhs_buffer, 1:n_sys)

	# ── 6a. Parametrisation coefficients (zeroth time-derivative order) ───────
	W.poly.coefficients[:, 1, idx] .= view(sol, 1:FOM)

	# ── 6b. Reduced dynamics — non-zero only for resonant master modes ─────────
	rr = 1
	for r in 1:ROM
		if resonance[r]
			R.poly.coefficients[r, idx] = sol[FOM+rr]
			rr += 1
		else
			R.poly.coefficients[r, idx] = zero(T)
		end
	end

	# ── 7. Higher time-derivative coefficients W^(j)[α], j = 1 … ORD-1 ───────
	# Pass only the master-mode rows of R (1:ROM) as the reduced-dynamics matrix;
	# external dynamics are already supplied separately via `external_dynamics`.
	compute_higher_derivative_coefficients!(
		W.poly.coefficients,
		view(R.poly.coefficients, 1:ROM, :),
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
- `ml_cache :: MultilinearTermsCache` — precomputed factorisation cache built once before the loop.
"""
function solve_cohomological_equations!(
	W::Parametrisation{ORD, NVAR, T},
	R::ReducedDynamics{ROM, NVAR, T},
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT},
	model::NDOrderModel,
	ml_cache::MultilinearTermsCache,
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT}

	nterms = length(multiindex_set(W))

	# The multiindex set is stored in GrLex order, so iterating 1:nterms is
	# already in ascending-total-degree (causal) order — no sort needed.
	for idx in 1:nterms
		idx in ctx.linear_monomial_skip_set && continue
		solve_single_monomial!(W, R, idx, ctx, model, ml_cache)
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
- `left_eigenmodes :: AbstractMatrix{ComplexF64}` — left eigenvectors of
  the master modes, size `FOM × ROM` (used in the orthogonality conditions).
- `resonance_set :: ResonanceSet` — precomputed resonance look‑up table.
- `initial_W`, `initial_R` — optionally supply already-initialised objects
  (their linear monomials must be set correctly).

## Returns

`(W, R)` — the solved [`Parametrisation`](@ref) and [`ReducedDynamics`](@ref).
"""
function solve_cohomological_problem(
	model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, LT, MT},
	mset::MultiindexSet{NVAR},
	master_eigenvalues::SVector{ROM, ComplexF64},
	master_modes::Matrix{ComplexF64},
	left_eigenmodes::AbstractMatrix{ComplexF64},
	resonance_set::ResonanceSet;
	initial_W::Union{Nothing, Parametrisation} = nothing,
	initial_R::Union{Nothing, ReducedDynamics} = nothing,
	# Caller-supplied higher-derivative master-mode coefficients.
	# Shape: FOM × (ORD-1) × ROM, where slice [:, k, r] = W^(k+1)[e_r].
	master_modes_derivatives::Union{Nothing, AbstractArray{ComplexF64, 3}} = nothing,
) where {ORD, ORDP1, N_NL, N_EXT, LT, MT, NVAR, ROM}

	# ── 1. Dimensions and consistency checks ──────────────────────────────────
	@assert NVAR == ROM + N_EXT "Multiindex set has $NVAR variables but ROM + N_EXT = $(ROM + N_EXT)"
	FOM = size(master_modes, 1)
	@assert size(master_modes, 2) == ROM "master_modes must have $ROM columns"

	# Use the model's linear terms directly; downstream functions accept mixed
	# real/complex input and Julia promotes automatically at each multiply site.
	linear_terms = model.linear_terms

	L = length(mset)

	# ── 2. Parametrisation and reduced-dynamics objects ────────────────────────
	if initial_W !== nothing && initial_R !== nothing
		W = initial_W
		R = initial_R
	else
		@assert ORD == 1 || master_modes_derivatives !== nothing """
		master_modes_derivatives must be provided for ORD > 1 systems.
		Supply a FOM × (ORD-1) × ROM array whose slice [:, k, r] = W^(k+1)[e_r].
		"""

		W, R = create_parametrisation_method_objects(
			mset, ORD, FOM, ROM, N_EXT, ComplexF64,
		)

		# GrLex order: zero vector (if present) is at index 1, so eᵣ is at index
		# r (no zero vector) or r+1.  Only the first NVAR+1 entries need checking.
		zero_vec = SVector{NVAR, Int}(ntuple(_ -> 0, Val(NVAR)))
		has_zero = length(mset) >= 1 && mset.exponents[1] == zero_vec
		unit_offset = has_zero ? 1 : 0

		# Initialise linear monomials for master modes in W and R.
		for r in 1:ROM
			idx_er = r + unit_offset
			W.poly.coefficients[:, 1, idx_er] .= view(master_modes, :, r)
			for k in 2:ORD
				W.poly.coefficients[:, k, idx_er] .= view(master_modes_derivatives, :, k - 1, r)
			end
			R.poly.coefficients[r, idx_er] = master_eigenvalues[r]
		end

		# ── 3. External dynamics: copy from model into the last N_EXT rows of R ──
		if model.external_system !== nothing
			_embed_external_dynamics!(R, model.external_system.first_order_dynamics, mset)
		end
	end

	# ── Build Λ from R (for precompute functions that need the full matrix) ────
	# R's columns at unit-vector monomials eᵣ hold the r-th column of Λ.
	# At this point the upper-right block (master↔external coupling) is zero
	# because external monomials are not resonant and will not be modified.
	zero_vec = SVector{NVAR, Int}(ntuple(_ -> 0, Val(NVAR)))
	has_zero = length(mset) >= 1 && mset.exponents[1] == zero_vec
	unit_offset = has_zero ? 1 : 0
	Λ = zeros(ComplexF64, NVAR, NVAR)
	for r in 1:NVAR
		Λ[:, r] .= R.poly.coefficients[:, r + unit_offset]
	end
	lambda_diag = [Λ[i, i] for i in 1:NVAR]

	# ── 4. Orthogonality row operators ─────────────────────────────────────────
	# J_coeffs depend only on linear_terms and left_eigenmodes, not on the
	# external directions, so they are computed once and reused in both passes.
	orthogonality_J_coeffs = precompute_orthogonality_operator_coefficients(
		linear_terms, left_eigenmodes, master_eigenvalues,
	)

	# ── 5. Solve external linear monomials via a temporary context ─────────────
	# The external forcing directions Φ_ext[:, e] = W[e_ext] solve the same
	# cohomological equation (invariance + orthogonality rows) as every other
	# monomial.  We therefore call solve_single_monomial! for each e_ext using a
	# temporary context in which the external columns of generalised_right_eigenmodes
	# are set to zero.  Since the E_e column polynomials are linear in Φ_ext,
	# setting Φ_ext = 0 zeroes all E_coeffs and reduces the invariance equation to
	#   L(s_ext) W_ext = F_nl[e_ext] + orthogonality correction,
	# which is the correct first-pass equation for the forced-response direction.
	# After this loop, generalised_right_eigenmodes is built from the results and
	# the full C/E operators are precomputed for the main solve.
	# ── 5a. Build the multilinear-terms cache (valid for the full solve) ─────────
	ml_cache = build_multilinear_terms_cache(model, W)

	# ── 5b. Precompute skip set and lower-order coupling resources ────────────
	# All three are computed once and shared between both context objects.
	linear_skip_set = Set(_linear_monomial_indices(mset))
	multiindex_dict = build_exponent_index_map(mset)
	lower_order_buffer = [zeros(ComplexF64, FOM) for _ in 1:ORD]
	system_matrix_buffer = Matrix{ComplexF64}(undef, FOM + ROM, FOM + ROM)
	rhs_buffer = Vector{ComplexF64}(undef, FOM + ROM)
	external_rhs_buffer = zeros(ComplexF64, FOM)
	candidate_indices_by_monomial = Vector{Vector{Int}}(undef, L)
	for i in 1:L
		multi_i = mset[i]
		tdeg = sum(multi_i)
		candidate_indices_by_monomial[i] = tdeg < 2 ? Int[] :
										   indices_in_box_with_bounded_degree(mset, collect(multi_i), 2, tdeg)
	end

	# ── 5c. Compute master-column invariance polynomials once (Φ_ext-independent)
	# C_coeffs depend only on master_modes and Λ[1:ROM,1:ROM]; E_coeffs depend on
	# the external directions Φ_ext and are computed in two separate passes below.
	invariance_C_coeffs, D_master_steps = precompute_master_column_polynomials(
		linear_terms, master_modes, Λ,
	)

	if initial_W === nothing || initial_R === nothing
		# E_coeffs for the partial context: external directions = 0 (Φ_ext unknown).
		# D_master_steps is reused — no master-column work repeated.
		partial_E_coeffs = precompute_external_column_polynomials(
			linear_terms, zeros(ComplexF64, FOM, N_EXT), Λ, D_master_steps,
		)
		partial_eigenmodes = hcat(master_modes, zeros(ComplexF64, FOM, N_EXT))
		partial_orth_C_coeffs, partial_orth_E_coeffs = precompute_orthogonality_column_polynomials(
			orthogonality_J_coeffs, partial_eigenmodes, Λ,
		)
		partial_ctx = CohomologicalContext{ComplexF64, ORD, ORDP1, NVAR, FOM, LT}(
			linear_terms, partial_eigenmodes, lambda_diag,
			invariance_C_coeffs, partial_E_coeffs,
			orthogonality_J_coeffs,
			partial_orth_C_coeffs, partial_orth_E_coeffs,
			resonance_set, linear_skip_set,
			multiindex_dict, lower_order_buffer, candidate_indices_by_monomial,
			system_matrix_buffer, rhs_buffer, external_rhs_buffer,
		)
		for e in 1:N_EXT
			idx_ext = ROM + e + unit_offset
			solve_single_monomial!(W, R, idx_ext, partial_ctx, model, ml_cache)
		end
	end

	# ── 6. Build generalised_right_eigenmodes = [master_modes | Φ_ext] ─────────
	# Recover the external directions from W at the external unit-vector monomials
	# (solved above, or already set by the caller via initial_W/initial_R).
	external_directions = zeros(ComplexF64, FOM, N_EXT)
	for e in 1:N_EXT
		idx_ext = ROM + e + unit_offset
		external_directions[:, e] .= W.poly.coefficients[:, 1, idx_ext]
	end
	generalised_right_eigenmodes = hcat(master_modes, external_directions)

	# ── 7. Full invariance- and orthogonality-equation operator columns ────────
	# E_coeffs now computed efficiently using saved D_master_steps (no repeated
	# master-column Horner work).
	invariance_E_coeffs = precompute_external_column_polynomials(
		linear_terms, external_directions, Λ, D_master_steps,
	)
	orthogonality_C_coeffs, orthogonality_E_coeffs = precompute_orthogonality_column_polynomials(
		orthogonality_J_coeffs, generalised_right_eigenmodes, Λ,
	)

	# ── 8. Build full context and solve all remaining monomials ────────────────
	ctx = CohomologicalContext{ComplexF64, ORD, ORDP1, NVAR, FOM, LT}(
		linear_terms,
		generalised_right_eigenmodes,
		lambda_diag,
		invariance_C_coeffs,
		invariance_E_coeffs,
		orthogonality_J_coeffs,
		orthogonality_C_coeffs,
		orthogonality_E_coeffs,
		resonance_set,
		linear_skip_set,
		multiindex_dict, lower_order_buffer, candidate_indices_by_monomial,
		system_matrix_buffer, rhs_buffer, external_rhs_buffer,
	)

	solve_cohomological_equations!(W, R, ctx, model, ml_cache)

	return W, R
end

end # module CohomologicalEquations
