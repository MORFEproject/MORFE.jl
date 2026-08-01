# =============================================================================
# Solving the bordered cohomological system
#
# The system itself — its block structure, the resonance masking, and why the
# constant size is equivalent to the compacted one — is documented once, in the
# `CohomologicalEquations` module docstring.  What follows is only what is specific
# to *solving* it, and each point below is a trap someone will otherwise re-open.
#
# 1. Do not eliminate L(s) first.  Inner resonance is flagged by |λ_r − s| < tol
#    while det L(λ_r) = 0, so "α is resonant" means precisely "L(s_α) is numerically
#    singular" — and only resonant monomials have a border to eliminate.  Forming
#    L(s)⁻¹b and L(s)⁻¹C (the bordering elimination method) therefore has backward
#    error scaling with κ(L) → ∞ exactly where it would be used, and the two
#    subsequent differences of large quantities cancel catastrophically.  Factorising
#    the bordered matrix whole keeps the backward error at κ(M), which stays O(1)
#    because the border spans L's near-null directions (Keller's bordering lemma;
#    Govaerts, SIMAX 1991).
#
# 2. The factorisation must re-pivot on every monomial.  Point 1 holds only if
#    pivoting may move rows across the border, so a refactorisation that reuses a
#    frozen pivot sequence (KLU's `klu!` = klu_refactor) degrades precisely at the
#    resonances.  `_refactorise!` uses `klu_factor!`, which re-pivots while reusing
#    the cached symbolic analysis.
#
# 3. Never *declare* symmetry to a backend — let each one decide.  The bordered
#    pattern is structurally symmetric whenever the L union pattern is (the border
#    contributes a dense row together with its matching dense column), which is the
#    common FE case.  A solver told to exploit that symmetry constrains its
#    permutation to preserve it, forfeiting the row interchange point 1 depends on.
#    Given the choice, a general-purpose solver reaches the right conclusion by
#    itself.  Do not force a strategy here, and do not declare a symmetric matrix
#    type to Pardiso — see ext/MORFEPardisoExt.jl.
#
# The measurements behind points 2 and 3 — factoriser timings, fill, accuracy at a
# genuine near-singularity — are recorded outside the source, where they can be
# re-run and re-checked rather than silently ageing here.
# =============================================================================

# =============================================================================
# Sparse factorise-and-solve
# =============================================================================

"""
	_refactorise!(ss, A) -> factorisation object

Return a factorisation of `A`, reusing the symbolic analysis cached in `ss.fact` when
one is present.

The first call performs the full analysis and stores it; subsequent calls redo only
the **numeric** factorisation via `klu_factor!`, which re-pivots while reusing the
cached symbolic analysis. That distinction is the whole point: the bordered matrix
changes value on every monomial and goes near-singular in its `(1,1)` block at
resonances, so a frozen pivot sequence degrades exactly where accuracy is needed.

**Never use `klu!` here.** It maps to `klu_refactor`, which reuses the pivot sequence
chosen at the first monomial — the latent defect this whole change exists to remove.

`A`'s sparsity pattern must be identical on every call; `SparseLinearSolverState`
guarantees this by construction (only `nzval` is ever written).

`klu` takes `A.nzval` by reference rather than copying it, so the factorisation and
the template share one value array and the per-monomial assembly writes straight into
what KLU reads. Nothing has to be copied in afterwards, and no explicit aliasing step
is needed. `colptr`/`rowval` remain KLU's own (0-based) copies, and they are invariant
anyway, which is what keeps the cached analysis valid.
"""
function _refactorise!(ss::SparseLinearSolverState, A::SparseMatrixCSC)
	F = ss.fact
	if F === nothing
		# `check = false` so that a singular *first* monomial surfaces through the
		# caller's `issuccess` test like every later one, rather than throwing from a
		# different code path. `allowsingular` keeps KLU from halting inside the
		# factorisation so that `issuccess` is what decides.
		F = klu(A; check = false, allowsingular = true)
		# Only cache a factorisation that succeeded. A failed one is returned to the
		# caller (which raises) but not kept: if that error is ever caught and the
		# solve retried, the next call must redo the analysis rather than refactorise
		# from a half-initialised object.
		issuccess(F) && (ss.fact = F)
		return F
	end
	klu_factor!(F; check = false, allowsingular = true)
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
cached KLU factorisation otherwise.

Both branches reuse a symbolic analysis computed once: KLU through
[`_refactorise!`](@ref), Pardiso through its phase split (`_pardiso_prepare!` once,
then numeric-factorise + solve per monomial).

KLU's `ldiv!` is genuinely in-place, so the KLU branch needs no intermediate buffer.
`ss.solve_scratch` exists for Pardiso, whose solve requires distinct input and output
arrays.
"""
function _bordered_solve!(ss::SparseLinearSolverState, x::AbstractVector, s)
	if ss.pardiso !== nothing
		if ss.pardiso_matrix === nothing
			# Configure and analyse once. The matrix handed back is whatever form
			# Pardiso wants for the detected type; the pattern never changes after
			# this, so the analysis stays valid for the whole solve.
			ss.pardiso_matrix = _pardiso_prepare!(ss.pardiso, ss.bordered)
		end
		copyto!(ss.solve_scratch, x)
		_pardiso_factorise_solve!(ss.pardiso, ss.pardiso_matrix, x, ss.solve_scratch)
		return x
	end
	F = _refactorise!(ss, ss.bordered)
	issuccess(F) || _singular_bordered_system(s)
	ldiv!(F, x)
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
		ctx.invariance.column_coeffs, ctx.invariance.E_coeffs,
		resonance, lower_order_couplings, external_dynamics,
		ctx.buffers.external_rhs,
	)
	view(ctx.buffers.rhs, 1:FOM) .+= ctx.buffers.ml_result
	assemble_orthogonality_matrix_and_rhs!(
		view(ctx.buffers.system_matrix, (FOM+1):n_sys, 1:n_sys),
		view(ctx.buffers.rhs, (FOM+1):n_sys),
		s, ctx.orthogonality.J_coeffs,
		ctx.orthogonality.corner_coeffs, ctx.orthogonality.E_coeffs,
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
			evaluate_column!(column, s, r, ctx.invariance.column_coeffs)
		else
			fill!(column, zero(T))
		end
	end

	# ── 4. Orthogonality rows, staged then scattered ──────────────────────────
	orth = view(ctx.buffers.orthogonality_rows, 1:ROM, 1:n_sys)
	assemble_orthogonality_matrix_and_rhs!(
		orth, view(ctx.buffers.rhs, (FOM+1):n_sys), s,
		ctx.orthogonality.J_coeffs,
		ctx.orthogonality.corner_coeffs,
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
