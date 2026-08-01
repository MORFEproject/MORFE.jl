# =============================================================================
# The bordered cohomological system
#
# Every monomial α is solved from the same constant-size system
#
#     ┌                            ┐ ┌     ┐   ┌       ┐
#     │  L(s)      C(s) P          │ │ W_α │ = │  b_α  │   FOM rows  (invariance)
#     │  P Ĵ(s)    P Ĉ(s) P + τ Q  │ │ R_α │   │ P g_α │   ROM rows  (orthogonality)
#     └                            ┘ └     ┘   └       ┘
#
# with P = diag(resonance), Q = I − P and τ = 1.  Row FOM+r is the orthogonality
# condition when mode r is resonant, and the trivial equation R[r, α] = 0 when it
# is not.  Permuting the unknowns as [W; R_res; R_non] makes the system block
# triangular with an exactly decoupled τI block, so this is equivalent to the
# compacted (FOM+nR) system — but its size, and on the sparse path its sparsity
# pattern, no longer depend on the monomial.
#
# Why not eliminate L first.  Inner resonance is flagged by |λ_r − s| < tol, and
# det L(λ_r) = 0, so "α is resonant" means precisely "L(s_α) is numerically
# singular".  Forming L(s)⁻¹b and L(s)⁻¹C — the bordering elimination method —
# therefore has backward error scaling with κ(L) → ∞ exactly on the monomials it
# would be used for, and the two subsequent differences of large quantities cancel
# catastrophically.  Factorising the bordered matrix as a whole instead keeps the
# backward error at κ(M), which stays O(1) because the border spans L's near-null
# directions (Keller's bordering lemma; Govaerts, SIMAX 1991).  Stability relies on
# pivoting being free to swap rows across the border, so the numeric factorisation
# must re-pivot on every monomial — see `_refactorise!`.
#
# A note on solver strategy, because it is easy to break by "helping".  The bordered
# pattern is structurally symmetric whenever the L union pattern is — the border
# contributes a dense row together with its matching dense column — which is the
# common FE case.  A solver told to exploit that symmetry will constrain its
# permutation to preserve it, and so forfeit the row interchange the border depends
# on.  We therefore never *declare* symmetry to any backend; we let each one decide.
# Measured: UMFPACK's AUTO strategy, handed a structurally symmetric bordered matrix,
# selects the unsymmetric strategy (UMFPACK_STRATEGY_USED = 1, pivot tolerance 0.1),
# i.e. it reaches the right conclusion on its own.  Do not force a strategy here, and
# do not declare a symmetric matrix type to Pardiso — see ext/MORFEPardisoExt.jl.
# =============================================================================

# =============================================================================
# Sparse factorise-and-solve
# =============================================================================

"""
	_refactorise!(fact, A) -> factorisation object

Return a factorisation of `A`, reusing the cached symbolic analysis in `fact` when
one is present.

The first call performs the full analysis and stores it; subsequent calls redo only
the **numeric** factorisation, with partial pivoting, via `lu!`. That distinction
matters: the bordered matrix changes value on every monomial and goes near-singular
in its `(1,1)` block at resonances, so a frozen pivot sequence — what a pure
refactorisation such as KLU's `klu_refactor` would reuse — degrades exactly where
accuracy is needed. `lu!` reuses `F.symbolic` only.

`A`'s sparsity pattern must be identical on every call; `SparseLinearSolverState`
guarantees this by construction (only `nzval` is ever written).

On the first call the factorisation's value array is **aliased** to `A.nzval`, so the
per-monomial assembly writes straight into what UMFPACK reads and no values have to
be copied in afterwards.  At `nnz ≈ 1e7` that removes a ~160 MB memcpy from every
monomial.  Only `nzval` is shared: `colptr`/`rowval` stay the factorisation's own
copies, and they are invariant anyway, which is what keeps the cached analysis valid.
"""
function _refactorise!(fact::Ref{Any}, A::SparseMatrixCSC)
	F = fact[]
	if F === nothing
		# `check = false` so that a singular *first* monomial surfaces through the
		# caller's `issuccess` test like every later one, rather than throwing a bare
		# SingularException from a different code path.
		F = lu(A; check = false)
		# `lu` copies the values, so alias them back onto the template. Guarded
		# because it relies on `UmfpackLU.nzval` being a plain, assignable field.
		length(F.nzval) == nnz(A) || error(
			"UMFPACK value array has length $(length(F.nzval)) but the bordered " *
			"template has nnz = $(nnz(A)); cannot alias. This means `lu` no longer " *
			"preserves the pattern verbatim (e.g. it dropped stored zeros), which " *
			"would also invalidate symbolic-factorisation reuse.")
		F.nzval = A.nzval
		# Only cache a factorisation that succeeded. A failed one is returned to the
		# caller (which raises) but not kept: if that error is ever caught and the
		# solve retried, the next call must redo the analysis rather than refactorise
		# from a half-initialised object.
		issuccess(F) && (fact[] = F)
		return F
	end
	lu!(F; reuse_symbolic = true, check = false)
	return F
end

"""
	_singular_bordered_system(s)

Throw an informative error for a singular bordered cohomological matrix.

Bordering regularises `L(s)` only along the *master* directions.  When `s` sits on a
**non-master** eigenvalue of the full-order model — an outer resonance — `L(s)` is
singular in a direction the border does not span, and the bordered matrix is singular
too.  That is a property of the reduction, not of the solve: the invariant manifold is
not normally hyperbolic at that monomial and the master set is too small.  Enlarging
the master set to include the offending mode is the fix.

`ResonanceSet.outer_resonances` records exactly these monomials at construction time.
"""
function _singular_bordered_system(s)
	error("""
	Singular bordered cohomological system at superharmonic s = $s.

	The border regularises L(s) only along master directions, so this means s lies on
	a non-master eigenvalue of the full-order model — an outer resonance.  The
	reduction itself is ill-posed at this monomial, not just the linear solve.

	Enlarge the master set to include the resonant mode (check the `outer_resonances`
	field of the ResonanceSet, which flags these monomials), or lower the order so the
	offending monomial is not reached.""")
end

"""
	_bordered_solve!(ss, x, s) -> x

Solve `bordered * x = x` in place, dispatching to Pardiso when available and to the
cached UMFPACK factorisation otherwise.

Both branches reuse a symbolic analysis computed once: UMFPACK through
[`_refactorise!`](@ref), Pardiso through its phase split (`_pardiso_prepare!` once,
then numeric-factorise + solve per monomial).

The solve is routed through `ss.solve_scratch` because `ldiv!(F, x)` on an
`UmfpackLU` is defined as `ldiv!(x, F, copy(x))` — the two-argument form allocates a
copy of the right-hand side on every monomial. The three-argument form with a
persistent buffer does not.
"""
function _bordered_solve!(ss::SparseLinearSolverState, x::AbstractVector, s)
	if ss.pardiso !== nothing
		if ss.pardiso_matrix[] === nothing
			# Configure and analyse once. The matrix handed back is whatever form
			# Pardiso wants for the detected type; the pattern never changes after
			# this, so the analysis stays valid for the whole solve.
			ss.pardiso_matrix[] = _pardiso_prepare!(ss.pardiso, ss.bordered)
		end
		copyto!(ss.solve_scratch, x)
		_pardiso_factorise_solve!(ss.pardiso, ss.pardiso_matrix[], x, ss.solve_scratch)
		return x
	end
	F = _refactorise!(ss.fact, ss.bordered)
	issuccess(F) || _singular_bordered_system(s)
	copyto!(ss.solve_scratch, x)
	ldiv!(x, F, ss.solve_scratch)
	return x
end

# =============================================================================
# Shared bordered-system assembly (dense path)
# =============================================================================

"""
	_assemble_bordered_system!(ctx, s, resonance, lower_order_couplings, external_dynamics)

Assemble the `(FOM+ROM) × (FOM+ROM)` bordered cohomological system into
`ctx.buffers.system_matrix` and `ctx.buffers.rhs`.

- The first `FOM` rows come from the invariance equation (operator + nonlinear RHS).
- The last `ROM` rows come from the orthogonality conditions, with non-resonant modes
  contributing the trivial row `R[r, α] = 0`.

Called by the dense-path `_solve_monomial!`.
"""
function _assemble_bordered_system!(
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
	s,
	resonance::SVector{ROM, Bool},
	lower_order_couplings,
	external_dynamics,
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT, ROM}
	n_sys = FOM + ROM
	assemble_cohomological_matrix_and_rhs!(
		view(ctx.buffers.system_matrix, 1:FOM, 1:n_sys),
		view(ctx.buffers.rhs, 1:FOM),
		s, ctx.linear_terms,
		ctx.invariance.C_coeffs, ctx.invariance.E_coeffs,
		resonance, lower_order_couplings, external_dynamics,
		ctx.buffers.external_rhs,
	)
	view(ctx.buffers.rhs, 1:FOM) .+= ctx.buffers.ml_result
	assemble_orthogonality_matrix_and_rhs!(
		view(ctx.buffers.system_matrix, (FOM+1):n_sys, 1:n_sys),
		view(ctx.buffers.rhs, (FOM+1):n_sys),
		s, ctx.orthogonality.J_coeffs,
		ctx.orthogonality.C_coeffs, ctx.orthogonality.E_coeffs,
		resonance, lower_order_couplings, external_dynamics,
	)
	return
end

# =============================================================================
# Dense-path monomial solve
# =============================================================================

"""
	_solve_monomial!(ctx, s, resonance, lower_order_couplings, external_dynamics)

**Dense path.** Assemble the `(FOM+ROM)` bordered system via
`_assemble_bordered_system!`, then solve it in-place with `lu!` + `ldiv!`.
The solution is written into `ctx.buffers.rhs[1:FOM+ROM]`.
"""
function _solve_monomial!(
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
	s,
	resonance::SVector{ROM, Bool},
	lower_order_couplings,
	external_dynamics,
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT, ROM}
	_assemble_bordered_system!(ctx, s, resonance, lower_order_couplings, external_dynamics)
	n_sys = FOM + ROM
	F = lu!(view(ctx.buffers.system_matrix, 1:n_sys, 1:n_sys), check = false)
	issuccess(F) || _singular_bordered_system(s)
	ldiv!(F, view(ctx.buffers.rhs, 1:n_sys))
	return
end

"""
	_solve_monomial!(ctx, s, resonance, lower_order_couplings, external_dynamics)

**Sparse path** (dispatched when `MT <: SparseMatrixCSC`). Writes the bordered matrix
into the constant-pattern template held by `ctx.sparse_solver` and solves it with a
single factorisation.

The `(1,1)` block is evaluated by the untouched `build_sparse_L_and_rhs!` Horner pass
on its own workspace — it needs the transient intermediates `L[j](s)` to accumulate
the lower-order RHS — and is then scattered into the template. The border blocks are
staged in `ctx.buffers.orthogonality_rows` by the same assembly routine the dense
path uses, then scattered into their strided positions in `nzval`.

Results are written into `ctx.buffers.rhs[1:FOM+ROM]`.
"""
function _solve_monomial!(
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
	s,
	resonance::SVector{ROM, Bool},
	lower_order_couplings,
	external_dynamics,
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT <: SparseMatrixCSC, ROM}
	ss = ctx.sparse_solver::SparseLinearSolverState{T}
	M = ss.bordered
	Mnz = M.nzval
	n_sys = FOM + ROM

	# ── 1. L(s) block and the lower-order RHS accumulation ────────────────────
	rhs = view(ctx.buffers.rhs, 1:FOM)
	fill!(rhs, zero(T))
	build_sparse_L_and_rhs!(
		rhs, ss.L_template, ss.L_mappings,
		ctx.linear_terms, s, lower_order_couplings,
	)
	scatter_L_into_bordered!(M, ss.L_template)

	# ── 2. External forcing and nonlinear contributions ───────────────────────
	evaluate_external_rhs!(rhs, s, external_dynamics, ctx.invariance.E_coeffs,
		ctx.buffers.external_rhs)
	rhs .+= ctx.buffers.ml_result

	# ── 3. Invariance border columns C(s)P — contiguous runs in nzval ─────────
	@inbounds for r in 1:ROM
		base = M.colptr[FOM+r]
		column = view(Mnz, base:(base+FOM-1))
		if resonance[r]
			evaluate_column!(column, s, r, ctx.invariance.C_coeffs)
		else
			fill!(column, zero(T))
		end
	end

	# ── 4. Orthogonality rows, staged then scattered ──────────────────────────
	orth = view(ctx.buffers.orthogonality_rows, 1:ROM, 1:n_sys)
	assemble_orthogonality_matrix_and_rhs!(
		orth, view(ctx.buffers.rhs, (FOM+1):n_sys), s,
		ctx.orthogonality.J_coeffs,
		ctx.orthogonality.C_coeffs,
		ctx.orthogonality.E_coeffs,
		resonance, lower_order_couplings, external_dynamics,
	)
	# P Ĵ(s): border rows of the first FOM columns (contiguous ROM-runs on both sides).
	@inbounds for c in 1:FOM
		base = ss.border_row_base[c]
		for r in 1:ROM
			Mnz[base+r-1] = orth[r, c]
		end
	end
	# P Ĉ(s) P + τ Q: the ROM × ROM corner.
	@inbounds for q in 1:ROM
		base = M.colptr[FOM+q] + FOM
		for r in 1:ROM
			Mnz[base+r-1] = orth[r, FOM+q]
		end
	end

	# ── 5. One factorisation, one solve ───────────────────────────────────────
	_bordered_solve!(ss, view(ctx.buffers.rhs, 1:n_sys), s)
	return
end
