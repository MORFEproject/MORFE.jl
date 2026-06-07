# -----------------------------------------------------------------------
# Symmetry classification
# -----------------------------------------------------------------------
#
# MultilinearMap.multiindex[k] is the number of factor slots that use the
# k-th derivative.  Three cases arise, each with a different accumulation strategy:
#
#   FullyAsymmetric    — all entries ≤ 1: distinct orders, t.f! goes directly
#                        into the accumulator (no scratch buffer needed).
#   FullySymmetric     — one positive entry > 1: all slots share one derivative,
#                        each factorisation carries a symmetry count.
#   GroupwiseSymmetric — multiple positive entries: slots span several derivatives,
#                        each factorisation carries a combined symmetry count.
#
# Dispatching on these tags lets Julia specialise the hot inner loop at
# compile time.

"""
	SymmetryType

Abstract tag type used to dispatch the inner factorisation accumulation strategy
at compile time.  The three concrete subtypes correspond to the three cases that
arise from `MultilinearMap.multiindex`:

- `FullyAsymmetric`    — all entries ≤ 1; each factor slot uses a different derivative.
- `FullySymmetric`     — exactly one entry > 1; all slots share one derivative.
- `GroupwiseSymmetric` — multiple entries > 0; slots span several derivatives.
"""
abstract type SymmetryType end

"""
	FullyAsymmetric <: SymmetryType

Tag for terms whose factor slots all use distinct derivatives (`multiindex` has no
repeated entry > 1).  No scratch buffer is needed; `t.f!` writes directly into
the accumulator.
"""
struct FullyAsymmetric <: SymmetryType end

"""
	FullySymmetric <: SymmetryType

Tag for terms where all factor slots share a single derivative order (exactly one
positive entry in `multiindex`).  Each factorisation carries a symmetry count.
"""
struct FullySymmetric <: SymmetryType end

"""
	GroupwiseSymmetric <: SymmetryType

Tag for terms whose factor slots span multiple derivative orders (multiple positive
entries in `multiindex`).  Uses `factorisations_groupwise_symmetric` with a combined
per-group symmetry count.
"""
struct GroupwiseSymmetric <: SymmetryType end

"""
	symmetry_type(t) -> SymmetryType

Classify a `AbstractMultilinearMap` as `FullyAsymmetric`, `FullySymmetric`,
or `GroupwiseSymmetric` based on its `multiindex` field.

Setting `fully_asymmetric = true` on any term overrides the `multiindex`-based
classification and always returns `FullyAsymmetric`.
"""
function symmetry_type(t::MultilinearMap)
	t.fully_asymmetric === true && return FullyAsymmetric()
	all(x -> x <= 1, t.multiindex) && return FullyAsymmetric()
	count(>(0), t.multiindex) == 1 && return FullySymmetric()
	return GroupwiseSymmetric()
end

function symmetry_type(t::AbstractMultilinearMap)
	t.fully_asymmetric === true && return FullyAsymmetric()
	all(x -> x <= 1, t.multiindex) && return FullyAsymmetric()
	count(>(0), t.multiindex) == 1 && return FullySymmetric()
	return GroupwiseSymmetric()
end

# -----------------------------------------------------------------------
# Derivative order helper
# -----------------------------------------------------------------------

"""
	_derivative_orders(t) → NTuple

Map each factor slot to its 1-based derivative index.
Example: `multiindex = (2, 1)` → `(1, 1, 2)`.
"""
function _derivative_orders(t::AbstractMultilinearMap)
	deg = sum(t.multiindex)
	return ntuple(deg) do slot
		cumulative = 0
		for (k, cnt) in enumerate(t.multiindex)
			cumulative += cnt
			slot <= cumulative && return k
		end
	end
end
