"""
Module `MultilinearMaps` — multilinear nonlinear term representations for `NDOrderModel`.

Nonlinear terms in the full-order ODE are encoded as `AbstractMultilinearMap` subtypes:

- `MultilinearMap{ORD, F}` — a single polynomial term of total degree `deg`, stored as
  a callable `f!` that evaluates the multilinear form plus metadata (`multiindex`,
  `multiplicity_external`, `deg`).
- `FEMMultilinearMap{ORD}` — abstract base for FEM-backed terms that expose element-level
  primitives (`fem_elements`, `scatter_qp!`, `accumulate_qp!`, `assemble_element!`, …).
  Implementing these methods enables the O4 batched RHS-C assembly path in `MultilinearTerms`.

See `demo/Gridap/` and `demo/Ferrite/` for reference FEM backend implementations.
"""
module MultilinearMaps

export AbstractMultilinearMap, FEMMultilinearMap, MultilinearMap, evaluate_term!,
	fem_elements, fem_n_qp, fem_ndofs_per_cell,
	scatter_qp!, accumulate_qp!, assemble_element!, fem_getdetJdV, fem_qp_buffer,
	fem_reinit!

"""
	AbstractMultilinearMap{ORD}

Abstract supertype for all multilinear terms accepted by `NDOrderModel`.

Every concrete subtype must expose the fields `multiindex`, `multiplicity_external`, and `deg`
with the same semantics as `MultilinearMap`.
"""
abstract type AbstractMultilinearMap{ORD} end

"""
	FEMMultilinearMap{ORD} <: AbstractMultilinearMap{ORD}

Abstract type for FEM-backed multilinear terms that expose element-level primitives.

Implementing the interface below enables the RHS-C batched accumulation path in
`MultilinearTerms.jl`: the mesh is traversed exactly once per (monomial, term, split)
rather than once per factorisation entry.

Required methods (extend `MORFE.*`):
- `fem_elements(t)`                                             → element iterator
- `fem_n_qp(t)`                                                 → quadrature points per element
- `fem_ndofs_per_cell(t)`                                       → DOFs per element
- `scatter_qp!(∇W_col, W_global, element, t)`                   → fill qp field values for one unique W column
- `accumulate_qp!(Fe, ∇W_args::NTuple, mult, element, q, dΩ, t)`→ add integrand at one qp
- `assemble_element!(accum, Fe, element, t)`                    → scatter element residual to global
- `fem_getdetJdV(element, q, t)`                                → integration weight at qp q
- `fem_qp_buffer(t)`                                            → pre-allocated scratch buffer for one quadrature point
"""
abstract type FEMMultilinearMap{ORD} <: AbstractMultilinearMap{ORD} end

# Interface stubs — extend these in your concrete FEM type.
function fem_elements end
function fem_n_qp end
function fem_ndofs_per_cell end
function fem_reinit! end
function scatter_qp! end
function accumulate_qp! end
function assemble_element! end
function fem_getdetJdV end
function fem_qp_buffer end

"""
	MultilinearMap{ORD, F}

Represents a single monomial term of order deg in the nonlinear function of an `NDOrderModel`.

A term is represented using a multiindex stored in the NTuple 
	`multiindex` = (i_0, ..., i_{ORD-1})  
where i_k is the multiplicity of the derivative x^(k). So the i_k specifies how many times the derivative x^(k) appears as an argument. 
In addition the function accepts `multiplicity_external` external variables r_1, r_2, ...
which satisfy the first order dynamic system r' = dynamics_external(r),
where dynamics_external is a DensePolynomial defined in NDOrderModel. The influence in f! is described by `multiplicity_external`

During evaluation the multilinear map is called as

	f!(res,
	   x^(0), ... repeated i_0 times,
	   x^(1), ... repeated i_1 times,
	   ...
	   x^(ORD-1), ...repeated i_{ORD-1} times,
	   r, ... repeated `multiplicity_external` times)

# Important Notes
- Each `MultilinearMap` **must implement a multilinear map**, i.e., it should be linear in each of its arguments independently.
- The function `f!` accumulates (adds) into `res` and must be callable with the appropriate number of arguments.
- If one i_k is larger than 1 we assume the input arguments are symmetric by permutation. For example:
	multiindex = (0, 2,...)
	f!(res, x^(1)_1, x^(1)_2, ...) = f!(res, x^(1)_2, x^(1)_1, ...)
- Set `force_asymmetric = true` to override the symmetry assumption: the term is treated as
  `FullyAsymmetric` regardless of `multiindex`, so every ordered argument permutation is
  evaluated independently with multiplier 1. Use this when `f!` is **not** symmetric in
  arguments that share a derivative order.

"""
struct MultilinearMap{ORD, F} <: AbstractMultilinearMap{ORD}
	f!::F
	multiindex::NTuple{ORD, Int}
	multiplicity_external::Int
	deg::Int
	force_asymmetric::Bool
end

"""
	MultilinearMap(f, multiindex)

Create a multilinear term for a system of order ORD without external dynamics.

# Arguments
- `f!`: in-place evaluation function
- `multiindex`: tuple specifying which derivatives are used
"""
function MultilinearMap(f!, multiindex::NTuple{ORD, Int};
	force_asymmetric::Bool = false) where {ORD}
	@assert all(multiindex .>= 0) "Terms in the multiindex cannot be negative, but multiindex=$multiindex"
	deg = sum(multiindex)
	# Check if input arguments of f matches deg
	ms = methods(f!)
	@assert length(ms) == 1 "Function $(f!) must have exactly one method to determine number of inputs"
	@assert ms[1].nargs == deg + 2 "Function $(f!) must accept $(deg+1) arguments (`res` and $deg inputs) instead of $(ms[1].nargs - 1)"
	@assert deg >= 2 "Function $(f!) must have degree at least 2, but has degree $deg"

	return MultilinearMap{ORD, typeof(f!)}(f!, multiindex, 0, deg, force_asymmetric)
end

# Create a multilinear term for a first order system.
function MultilinearMap(f!; force_asymmetric::Bool = false)
	ms = methods(f!)
	@assert length(ms) == 1 "Function $(f!) must have exactly one method to determine number of inputs"
	deg = ms[1].nargs - 2 # subtract the function itself and `res`
	@assert deg >= 2 "Function $(f!) must have degree at least 2, but has degree $deg"
	multiindex = (UInt8(deg),)
	return MultilinearMap{1, typeof(f!)}(f!, multiindex, 0, deg, force_asymmetric)
end

function MultilinearMap(
	f!, multiindex::NTuple{ORD, Int}, multiplicity_external::Int;
	force_asymmetric::Bool = false) where {ORD}
	@assert all(multiindex .>= 0) "Terms in the multiindex cannot be negative, but multiindex=$multiindex"
	@assert multiplicity_external >= 0 "The argument multiplicity_external cannot be negative, but multiplicity_external=$multiplicity_external"
	deg = sum(multiindex) + multiplicity_external
	# Check if input arguments of f matches deg
	ms = methods(f!)
	@assert length(ms) == 1 "Function $(f!) must have exactly one method to determine number of inputs"
	@assert ms[1].nargs == deg + 2 "Function $(f!) must accept $(deg+1) arguments (`res` and $deg inputs) instead of $(ms[1].nargs - 1)"
	@assert (deg >= 2) || (multiplicity_external >= 1)
	"Function $(f!) does not depend the external state, hence it must have degree at least 2, but it has degree $deg"

	return MultilinearMap{ORD, typeof(f!)}(f!, multiindex, multiplicity_external, deg,
		force_asymmetric)
end

"""
	evaluate_term!(res, term, xs, r)

Evaluate a single `MultilinearMap` and accumulate (adds) the result into `res`.

# Arguments
- `res`: output vector (modified in-place)
- `term`: multilinear term
- `xs`: tuple `(x, x^(1), …, x^(ORD-1))` of state derivatives
- `r`: external state vector (or `nothing` if not used). If `r` is `nothing` but the term expects external arguments, an error is thrown.
"""
@inline function evaluate_term!(res, term::MultilinearMap{ORD}, xs, r) where {ORD}
	inds = term.multiindex
	# me = term.multiplicity_external
	total_args = term.deg

	# Build the argument list
	args = ntuple(total_args) do k
		if k <= sum(inds)
			# Pick from xs based on multiindex
			s = 0
			for j in 1:ORD
				s += inds[j]
				if k ≤ s
					return @inbounds xs[j]
				end
			end
		else
			# Pick from external state
			if r === nothing
				error("Term expects external arguments but no external state provided")
			end
			return r
		end
	end
	term.f!(res, args...)
end

"""
	evaluate_term!(res, t::FEMMultilinearMap{ORD}, xs, r)

Direct (uncached) evaluation of a FEM-backed multilinear term at state `xs` and
external state `r`, accumulating the result into `res`. To be used in InvarianceError.jl.

Internal argument slots (determined by `t.multiindex`) are scattered to quadrature-point
field quantities via `scatter_qp!` and assembled element-wise.

For `me = 0`: `∇W_args` is a homogeneous `NTuple{N_INT, QP_TYPE}` — type-stable.

For `me > 0`: the external arg slots in `∇W_args` receive `r` directly (not scattered).
`r` is a small N_EXT-dimensional external-state vector, not a FOM displacement field.
The concrete `accumulate_qp!` evaluates the full multilinear map
`F(∇u₁,…,∇uₙ, r₁,…,rₘₑ)` at the actual inputs.
"""
function evaluate_term!(res, t::FEMMultilinearMap{ORD}, xs, r) where {ORD}
	_eval_fem_term_direct!(res, t, xs, r, Val(t.deg), Val(sum(t.multiindex)))
end

function _eval_fem_term_direct!(
	res, t::FEMMultilinearMap{ORD}, xs, r,
	::Val{DEG}, ::Val{N_INT}) where {ORD, DEG, N_INT}
	inds = t.multiindex
	me = DEG - N_INT        # number of external arg slots; compile-time constant
	∇W_qp = fem_qp_buffer(t)  # pre-allocated; ≥ DEG rows guaranteed by constructor
	n_qp = fem_n_qp(t)
	n_dofs = fem_ndofs_per_cell(t)
	Fe = zeros(eltype(res), n_dofs)

	for element in fem_elements(t)
		fem_reinit!(element, t)
		# Scatter N_INT internal arg slots to qp-level gradients (rows 1:N_INT).
		slot = 0
		for j in 1:ORD
			for _ in 1:inds[j]
				slot += 1
				scatter_qp!(@view(∇W_qp[slot, 1:n_qp]), xs[j], element, t)
			end
		end
		fill!(Fe, zero(eltype(Fe)))
		for q in 1:n_qp
			dΩ = fem_getdetJdV(element, q, t)
			if me == 0
				∇W_args = ntuple(k -> ∇W_qp[k, q], Val(N_INT))
			else
				∇W_args = ntuple(k -> k ≤ N_INT ? ∇W_qp[k, q] : r, Val(DEG))
			end
			accumulate_qp!(Fe, ∇W_args, 1.0, element, q, dΩ, t)
		end
		assemble_element!(res, Fe, element, t)
	end
end

end # module
