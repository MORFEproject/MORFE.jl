# -----------------------------------------------------------------------
# Per-split accumulation — one dispatch method per symmetry class
# -----------------------------------------------------------------------
#
# Each method accumulates the contributions for one (external-system) split
# of the exponent `rem` into `accum`.  `t.f!` must use += semantics.
#
# The caller is responsible for zeroing `accum` before calling when needed
# (see accumulate_multilinear_term! below).

"""
	_accumulate_split!(accum, scratch, t, sym, W, set, rem, deg, candidate_indices, args_ext)

Accumulate contributions from one (monomial, external-split) pair into `accum`,
dispatching on the `SymmetryType` tag `sym`.

- `FullyAsymmetric`: calls `t.f!` directly into `accum` for each factorisation.
- `FullySymmetric` / `GroupwiseSymmetric`: calls `t.f!` into `scratch`, then
  `axpy!(multiplier, scratch, accum)` to apply the symmetry count.
"""
function _accumulate_split!(accum, _scratch,
	t, ::FullyAsymmetric, W, set, rem, deg, candidate_indices, args_ext)
	orders = _derivative_orders(t)
	for entry in factorisations_asymmetric(set, rem, deg, candidate_indices)
		@inbounds args = ntuple(i -> @view(W[:, orders[i], entry.factor_indices[i]]), Val(deg))
		t.f!(accum, args..., args_ext...)
	end
end

# All factor slots share one derivative; each factorisation has a symmetry count.
function _accumulate_split!(accum, scratch,
	t, ::FullySymmetric, W, set, rem, deg, candidate_indices, args_ext)
	deriv_idx = findfirst(>(0), t.multiindex)::Int
	for entry in factorisations_fully_symmetric(set, rem, deg, candidate_indices)
		fill!(scratch, 0)
		@inbounds args = ntuple(i -> @view(W[:, deriv_idx, entry.factor_indices[i]]), Val(deg))
		t.f!(scratch, args..., args_ext...)
		axpy!(entry.multiplier, scratch, accum)
	end
end

# Factor slots span several derivatives; each factorisation has a combined count.
function _accumulate_split!(accum, scratch,
	t, ::GroupwiseSymmetric, W, set, rem, deg, candidate_indices, args_ext)
	orders = _derivative_orders(t)
	for entry in
		factorisations_groupwise_symmetric(set, rem, t.multiindex, candidate_indices)
		fill!(scratch, 0)
		@inbounds args = ntuple(i -> @view(W[:, orders[i], entry.factor_indices[i]]), Val(deg))
		t.f!(scratch, args..., args_ext...)
		axpy!(entry.multiplier, scratch, accum)
	end
end

# -----------------------------------------------------------------------
# Term-level accumulation
# -----------------------------------------------------------------------

"""
	accumulate_multilinear_term!(result, scratch, temp, t, parametrisation,
								  exp, candidate_indices, external_exp, unit_vectors)

Add the contribution of nonlinear term `t` for exponent `exp` to `result`.

When `t` has no external (forcing) slots (`me = 0`) there is exactly one split,
so we skip `bounded_index_tuples` and write directly into `result`, saving one
`fill!` and one `axpy!` per term.  For `me > 0` each split is accumulated into
`temp` first, then scaled into `result` via `axpy!`.
"""
function accumulate_multilinear_term!(result, scratch, temp,
	t::MultilinearMap{ORD}, parametrisation::Parametrisation{ORD, NVAR},
	exp::SVector{NVAR}, candidate_indices, external_exp, unit_vectors) where {ORD, NVAR}
	W = parametrisation.poly.coefficients
	set = parametrisation.poly.multiindex_set
	me = t.multiplicity_external
	deg = t.deg - me
	ROM = NVAR - parametrisation.external_system_size
	sym = symmetry_type(t)

	if me == 0
		_accumulate_split!(result, scratch, t, sym, W, set, exp, deg, candidate_indices, ())
	else
		for (ext_idx, ext_multiindex_external, ext_count) in
			bounded_index_tuples(me, external_exp)
			ext_multiindex = SVector(ntuple(
				i -> i <= ROM ? 0 :
					 ext_multiindex_external[i-ROM], Val(NVAR)))
			rem = exp - ext_multiindex
			args_ext = ntuple(i -> unit_vectors[ext_idx[i]], me)
			fill!(temp, 0)
			_accumulate_split!(
				temp, scratch, t, sym, W, set, rem, deg, candidate_indices, args_ext)
			axpy!(ext_count, temp, result)
		end
	end
end

# -----------------------------------------------------------------------
# Public API — non-cached
# -----------------------------------------------------------------------

"""
	compute_multilinear_terms(model, exp, parametrisation) → Vector

Return the sum of all nonlinear-term contributions for exponent `exp`.

Scratch buffers and shared data (`unit_vectors`, `candidate_indices`,
`external_exp`) are allocated once and reused across all terms.
"""
function compute_multilinear_terms(model::NDOrderModel{ORD}, exp::SVector{NVAR},
	parametrisation::Parametrisation{ORD, NVAR}) where {ORD, NVAR}
	set = parametrisation.poly.multiindex_set
	FOM = size(parametrisation)
	T = eltype(parametrisation.poly)
	deg_max = sum(exp)

	result = zeros(T, FOM)
	scratch = similar(result)   # per-factorisation scratch (symmetric branches)
	temp = similar(result)   # per-external-split accumulator (me > 0 only)

	external_system_size = parametrisation.external_system_size
	ROM = NVAR - external_system_size
	unit_vectors = [SVector(ntuple(k -> k == j ? 1 : 0, external_system_size))
					for j in 1:external_system_size]
	candidate_indices = indices_in_box_with_bounded_degree(set, exp, 1, deg_max)
	external_exp = SVector(ntuple(i -> exp[ROM+i], external_system_size))

	for t in model.nonlinear_terms
		t.deg > deg_max && continue
		accumulate_multilinear_term!(result, scratch, temp, t, parametrisation,
			exp, candidate_indices, external_exp, unit_vectors)
	end
	return result
end
