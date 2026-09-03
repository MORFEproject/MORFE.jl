"""
Module `MultilinearMaps` — multilinear nonlinear term representations for `NthOrderModel`.

Nonlinear terms in the full-order ODE are encoded as `AbstractMultilinearMap` subtypes:

- `MultilinearMap{ORD, F}` — a single polynomial term of total degree `deg`, stored as
  a callable `f!` that evaluates the multilinear form plus metadata (`multiindex`,
  `multiplicity_external`, `deg`).
- `FEMMultilinearMap{ORD}` — abstract base for FEM-backed terms that expose element-level
  primitives (`fem_elements`, `scatter_qp!`, `accumulate_qp!`, `assemble_element!`, …).
  Implementing these methods enables the O4 batched RHS-C assembly path in `MultilinearTerms`.

See `MORFEFerrite.jl` (StructuralSVK / FluidNavierStokes) and `examples/02_clamped_beam_gridap/` for reference FEM backend implementations.
"""
module MultilinearMaps

export AbstractMultilinearMap, FEMMultilinearMap, MultilinearMap, evaluate_term!,
       fem_elements, fem_n_qp, fem_ndofs_per_cell,
       scatter_qp!, accumulate_qp!, assemble_element!, fem_getdetJdV, fem_qp_buffer,
       fem_reinit!

"""
	AbstractMultilinearMap{ORD}

Abstract supertype for all multilinear terms accepted by `NthOrderModel`.

Every concrete subtype must expose the fields `multiindex`, `multiplicity_external`, `deg`,
and `fully_asymmetric` with the same semantics as `MultilinearMap`.
"""
abstract type AbstractMultilinearMap{ORD} end

"""
	FEMMultilinearMap{ORD} <: AbstractMultilinearMap{ORD}

Abstract type for FEM-backed multilinear terms that expose element-level primitives.

Implementing the interface below enables the RHS batched accumulation path in
`MultilinearTerms.jl`: the mesh is traversed exactly once per (monomial, term, split)
rather than once per factorisation entry.

Required fields (same semantics as `MultilinearMap`):
- `multiindex`, `multiplicity_external`, `deg`
- `fully_asymmetric::Union{Nothing, Bool}` — `nothing` = not set (triggers `@info` at `NthOrderModel`
  construction if multiindex implies symmetry); `false` = acknowledged symmetric; `true` = override to
  `FullyAsymmetric`. FEM backends whose integrand is symmetric by construction should default to `false`.

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

"""
	fem_elements(t::FEMMultilinearMap) -> iterator

Return an iterator over the mesh elements (cells) for the FEM term `t`.
Implement this for every concrete `FEMMultilinearMap` subtype.
"""
function fem_elements(t::FEMMultilinearMap)
    error("fem_elements not implemented for $(typeof(t))")
end
"""
	fem_n_qp(t::FEMMultilinearMap) -> Int

Return the number of quadrature points per element for the FEM term `t`.
"""
function fem_n_qp(t::FEMMultilinearMap)
    error("fem_n_qp not implemented for $(typeof(t))")
end
"""
	fem_ndofs_per_cell(t::FEMMultilinearMap) -> Int

Return the number of degrees of freedom per element (cell) for the FEM term `t`.
"""
function fem_ndofs_per_cell(t::FEMMultilinearMap)
    error("fem_ndofs_per_cell not implemented for $(typeof(t))")
end
"""
	fem_reinit!(element, t::FEMMultilinearMap) -> nothing

Reinitialise the FEM quadrature cache (e.g. `CellValues`) for `element`.
Called once per element before any `scatter_qp!` or `accumulate_qp!` calls.
"""
function fem_reinit!(element, t::FEMMultilinearMap)
    error("fem_reinit! not implemented for $(typeof(t))")
end
"""
	scatter_qp!(∇W_col, W_global, element, t::FEMMultilinearMap) -> nothing

Fill `∇W_col` in-place with the field values (e.g. gradients) at all quadrature
points of `element` for one column of the parametrisation `W_global`.
"""
function scatter_qp!(∇W_col, W_global, element, t::FEMMultilinearMap)
    error("scatter_qp! not implemented for $(typeof(t))")
end
"""
	accumulate_qp!(Fe, ∇W_args, mult, element, q, dΩ, t::FEMMultilinearMap) -> nothing

Accumulate the integrand contribution at quadrature point `q` (weight `dΩ`) into
the element residual vector `Fe`.  `∇W_args` is an NTuple of pre-scattered W columns;
`mult` is the combinatorial multiplicity of the term.
"""
function accumulate_qp!(Fe, ∇W_args, mult, element, q, dΩ, t::FEMMultilinearMap)
    error("accumulate_qp! not implemented for $(typeof(t))")
end
"""
	assemble_element!(accum, Fe, element, t::FEMMultilinearMap) -> nothing

Scatter the element residual `Fe` into the global accumulator `accum` using the
DOF map of `element`.
"""
function assemble_element!(accum, Fe, element, t::FEMMultilinearMap)
    error("assemble_element! not implemented for $(typeof(t))")
end

"""
	fem_getdetJdV(element, q, t::FEMMultilinearMap) -> Real

Return the integration weight `det(J) · w_q` at quadrature point index `q` of
`element`.
"""
function fem_getdetJdV(element, q, t::FEMMultilinearMap)
    error("fem_getdetJdV not implemented for $(typeof(t))")
end

"""
	fem_qp_buffer(t::FEMMultilinearMap)

Return a pre-allocated scratch buffer sized for one quadrature point of the FEM
term `t`.  Reused across calls to avoid allocation in the inner element loop.
"""
function fem_qp_buffer(t::FEMMultilinearMap)
    error("fem_qp_buffer not implemented for $(typeof(t))")
end

"""
	MultilinearMap{ORD, F}

Represents a single monomial term of order deg in the nonlinear function of an `NthOrderModel`.

A term is represented using a multiindex stored in the NTuple 
	`multiindex` = (i_0, ..., i_{ORD-1})  
where i_k is the multiplicity of the derivative x^(k). So the i_k specifies how many times the derivative x^(k) appears as an argument. 
In addition the function accepts `multiplicity_external` external variables r_1, r_2, ...
which satisfy the first order dynamic system r' = dynamics_external(r),
where dynamics_external is a DensePolynomial defined in NthOrderModel. The influence in f! is described by `multiplicity_external`

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
- Set `fully_asymmetric = true` to override the symmetry assumption: the term is treated as
  `FullyAsymmetric` regardless of `multiindex`, so every ordered argument permutation is
  evaluated independently with multiplier 1. Use this when `f!` is **not** symmetric in
  arguments that share a derivative order.

!!! note "Default: `fully_asymmetric` not set (`nothing`)"
    When this keyword is omitted, the following assumptions hold (and an `@info`
    message is emitted when the term is added to an `NthOrderModel`).
    Explicitly passing `fully_asymmetric = false` applies the same behaviour
    silently, without triggering the message.

    1. **`f!` is symmetric within each derivative-order group.** For every `k` with
       `multiindex[k] > 1`, permuting any two of the `multiindex[k]` argument slots
       that belong to derivative order `k` leaves the result unchanged.

    2. **Symmetry type is inferred automatically from `multiindex`:**
       - All entries ≤ 1 → `FullyAsymmetric`: each factor slot uses a distinct
         derivative order; `f!` is called directly with multiplier 1.
       - Exactly one entry > 1 → `FullySymmetric`: all slots share one derivative
         order; each unique unordered selection of factor indices is evaluated once,
         scaled by the multinomial coefficient `deg! / ∏ mᵢ!`.
       - Multiple entries > 1 → `GroupwiseSymmetric`: slots span several derivative
         orders; each unique unordered selection is evaluated once, scaled by the
         product of per-group multinomial coefficients.

    3. **Permutations are never evaluated separately.** Only one representative per
       equivalence class of argument orderings is passed to `f!`; the combinatorial
       count is applied as a scalar multiplier on the output.

    If `f!` does **not** satisfy assumption 1, pass `fully_asymmetric = true`.

# Fields

- `f!::F` — the in-place map itself, accumulating into its first argument.  Stored
  as a type parameter so calls through it are statically dispatched.
- `multiindex::NTuple{ORD, Int}` — `multiindex[k]` is how many times the derivative
  `x^(k-1)` appears as an argument.
- `multiplicity_external::Int` — how many external variables `r` are passed, after
  the derivative arguments.
- `deg::Int` — combined degree, `sum(multiindex) + multiplicity_external`.  Cached
  because it is compared against the monomial degree on every factorisation.
- `fully_asymmetric::Union{Nothing, Bool}` — overrides the symmetry inferred from
  `multiindex`.  `nothing` means "not stated", which behaves as `false` but also
  emits the `@info` note described above; that three-valued form is what lets the
  model warn about an assumption the caller may not have realised it was making.

# Construction

Four constructors are available.  The keyword form is the recommended one — it names
every argument and can infer `multiindex` from the system order, the total degree or
a per-slot list of derivative orders:

```julia
MultilinearMap(f!; multiindex, derivatives, order, degree,
    multiplicity_external, fully_asymmetric)   # recommended
MultilinearMap(f!)                             # shape inferred from the arity of f!
MultilinearMap(f!, multiindex; fully_asymmetric)
MultilinearMap(f!, multiindex, multiplicity_external; fully_asymmetric)
```

In the keyword form an omitted `order` means `ORD = 2`, the second-order mechanical
setting, and a one-factor `f!` is read as a pure external forcing term.  Every assumed
value is reported through an `@info`; the positional forms assume nothing and stay silent.
"""
struct MultilinearMap{ORD, F} <: AbstractMultilinearMap{ORD}
    f!::F
    multiindex::NTuple{ORD, Int}
    multiplicity_external::Int
    deg::Int
    fully_asymmetric::Union{Nothing, Bool}
end

# -----------------------------------------------------------------------
# Construction helpers
# -----------------------------------------------------------------------
#
# Every public constructor funnels into `_build_multilinear_map`, so the validation
# rules live in exactly one place.  The keyword constructor only resolves the
# (multiindex, multiplicity_external, ORD) triple from whichever keywords the caller
# supplied; the kernel then checks it and builds the struct.

"""
	_definition_site(f!) -> String

Return `" @ file.jl:12"` for the first method of `f!`, or `""` when `f!` exposes no
methods.  Used by `show` and by `FullOrderModel._term_label` to point a diagnostic at
the definition of a term.  Deliberately uses `first` rather than `only`: `f!` is allowed
to carry several methods, and a display helper must never throw.
"""
function _definition_site(f!)
    ms = methods(f!)
    isempty(ms) && return ""
    m = first(ms)
    return " @ $(basename(String(m.file))):$(m.line)"
end

"""
	_method_arities(f!) -> (fixed, va)

Split the methods of `f!` by argument count, *including* `res` but excluding the
callable itself.  `fixed` lists the exact counts of the fixed-arity methods; `va` lists,
for each varargs method, the minimum number of arguments it accepts.
"""
function _method_arities(f!)
    fixed = Int[]
    va = Int[]
    for m in methods(f!)
        n = m.nargs - 1                    # drop the callable itself
        m.isva ? push!(va, n - 1) : push!(fixed, n)
    end
    return fixed, va
end

"""
	_arity_description(fixed, va) -> String

Human-readable summary of the argument counts a callable accepts, e.g. `"3, ≥ 1"`.
"""
function _arity_description(fixed, va)
    parts = vcat([string(n) for n in sort(unique(fixed))],
        ["≥ $n" for n in sort(unique(va))])
    return isempty(parts) ? "none" : join(parts, ", ")
end

"""
	_call_signature(multiindex, multiplicity_external) -> String

Render the call `f!` receives, e.g. `"f!(res, x^(0), x^(0), x^(1), r)"` for
`multiindex = (2, 1)` and `multiplicity_external = 1`.
"""
function _call_signature(multiindex, multiplicity_external)
    slots = String[]
    for (k, n) in enumerate(multiindex), _ in 1:n

        push!(slots, "x^($(k - 1))")
    end
    append!(slots, fill("r", multiplicity_external))
    return isempty(slots) ? "f!(res)" : "f!(res, " * join(slots, ", ") * ")"
end

"""
	_symmetry_label(multiindex, fully_asymmetric) -> String

Name the symmetry class a term will be given by the parametrisation solver.

Mirrors `symmetry_type` in
`src/ParametrisationMethod/RightHandSide/MultilinearTerms/Symmetry.jl`, which cannot be
called from here: `MultilinearMaps` is loaded twelve includes earlier.  The testset
"show agrees with symmetry_type" holds the two in step.
"""
function _symmetry_label(multiindex, fully_asymmetric)
    fully_asymmetric === true && return "FullyAsymmetric (forced)"
    all(<=(1), multiindex) && return "FullyAsymmetric"
    count(>(0), multiindex) == 1 && return "FullySymmetric"
    return "GroupwiseSymmetric"
end

# -- error messages -----------------------------------------------------
#
# Built by helpers rather than inline so the formatter leaves them alone and the
# validation kernel stays readable.  Each one states the rule, the offending value,
# and the fix.

function _msg_negative_multiindex(multiindex)
    return "MultilinearMap: multiindex entries must be non-negative, got $multiindex.\n" *
           "multiindex[k] counts how many factor slots use the derivative x^(k-1), so\n" *
           "multiindex = (2, 1) means f!(res, x^(0), x^(0), x^(1))."
end

function _msg_empty_multiindex()
    return "MultilinearMap: multiindex must have at least one entry — its length is the\n" *
           "order ORD of the system the term belongs to.  For a first-order system pass\n" *
           "a 1-tuple, e.g. multiindex = (2,)."
end

function _msg_negative_external(multiplicity_external)
    return "MultilinearMap: multiplicity_external must be non-negative, got " *
           "$multiplicity_external.\nIt counts how many times the external state r is " *
           "passed to f!, after the derivative arguments."
end

function _msg_degree_too_low(multiindex, multiplicity_external, deg)
    return "MultilinearMap: a term with no external factors must have degree at least 2,\n" *
           "but multiindex = $multiindex gives deg = $deg.  Linear contributions belong in\n" *
           "the `linear_terms` matrices of `NthOrderModel`, not in a `MultilinearMap`.  If\n" *
           "this term is meant to depend on the external state r, pass " *
           "`multiplicity_external`."
end

function _msg_arity(f!, want, fixed, va, multiindex, multiplicity_external)
    deg = want - 1
    return "MultilinearMap: `f!` must accept $want arguments — `res` plus $deg factors —\n" *
           "but its methods accept $(_arity_description(fixed, va)) " *
           "(defined$(_definition_site(f!))).\n" *
           "  multiindex = $multiindex, multiplicity_external = $multiplicity_external\n" *
           "  ⇒ deg = sum(multiindex) + multiplicity_external = $deg\n" *
           "Expected call: $(_call_signature(multiindex, multiplicity_external))\n" *
           "Either change the arity of `f!`, or correct `multiindex`."
end

function _msg_infer_degree(f!, fixed)
    found = isempty(fixed) ? "only varargs methods" :
            "methods of arity $(_arity_description(fixed, Int[]))"
    return "MultilinearMap: cannot infer the degree of `f!`; it has $found" *
           "$(_definition_site(f!)).\n" *
           "Pass `multiindex`, `derivatives` or `degree` explicitly."
end

function _msg_multiindex_and_derivatives()
    return "MultilinearMap: pass either `multiindex` or `derivatives`, not both.\n" *
           "`multiindex` counts slots per derivative order; `derivatives` lists the\n" *
           "derivative order of each slot in call order, e.g. derivatives = (0, 0, 1)\n" *
           "is the same term as multiindex = (2, 1)."
end

function _msg_derivatives_negative(derivatives)
    return "MultilinearMap: `derivatives` entries must be non-negative, got $derivatives.\n" *
           "Each entry is the 0-based derivative order of one argument slot, so\n" *
           "derivatives = (0, 0, 1) means f!(res, x, x, xdot)."
end

function _msg_derivatives_unsorted(derivatives)
    return "MultilinearMap: `derivatives` must be non-decreasing, got $derivatives.\n" *
           "`evaluate_term!` always passes factors grouped by ascending derivative order,\n" *
           "so `derivatives` describes the call signature of `f!` and cannot be reordered\n" *
           "for you.  Either write derivatives = $(Tuple(sort(collect(derivatives)))) and " *
           "define `f!` to match,\nor reorder the arguments inside `f!` yourself."
end

function _msg_order_truncates(base, order)
    return "MultilinearMap: order = $order is smaller than length(multiindex) = " *
           "$(length(base))\n(multiindex = $base).  `order` may only zero-pad a multiindex " *
           "up to the system\norder, never truncate it — dropping an entry would silently " *
           "drop a derivative slot."
end

function _msg_degree_mismatch(degree, multiindex, multiplicity_external)
    total = sum(multiindex) + multiplicity_external
    return "MultilinearMap: degree = $degree disagrees with multiindex = $multiindex and\n" *
           "multiplicity_external = $multiplicity_external, which give\n" *
           "sum(multiindex) + multiplicity_external = $total.  Drop `degree`, or correct it."
end

function _msg_mixed_split(f!, total, internal, multiplicity_external)
    guess = _call_signature((internal,), multiplicity_external)
    return "MultilinearMap: cannot infer how the $total factors of `f!` split between the " *
           "state\nand the external state when multiplicity_external = " *
           "$multiplicity_external — that would mean\nguessing $guess.\n" *
           "State the internal shape explicitly:\n" *
           "  MultilinearMap(f!; multiindex = ($internal, 0), multiplicity_external = " *
           "$multiplicity_external)\n" *
           "  MultilinearMap(f!, ($internal, 0), $multiplicity_external)" *
           "                # positional form\n" *
           "Only a pure forcing term — no derivative factors at all — is inferred " *
           "automatically."
end

"""
	_msg_assumed_compact(f!, multiindex, multiplicity_external, assumed) -> String

One-line report of the values the constructor had to default.  `assumed` is the list of
field names that were not stated; only those are named, so the closing sentence ("state
them explicitly to silence this message") is always true.
"""
function _msg_assumed_compact(f!, multiindex, multiplicity_external, assumed)
    deg = sum(multiindex) + multiplicity_external
    return "MultilinearMap$(_definition_site(f!)) — assumed " * join(assumed, ", ") *
           ".\n" *
           "⇒ deg = $deg, $(_call_signature(multiindex, multiplicity_external)).  " *
           "State them explicitly to silence this message."
end

"""
	_msg_assumed_forcing(f!, multiindex, multiplicity_external) -> String

Full report for the one inference that can silently reinterpret a linear term as a forcing
term.  Every call it presents as silent really is silent under the resolution rules — the
testset "suggested silent forms are silent" pins that.
"""
function _msg_assumed_forcing(f!, multiindex, multiplicity_external)
    ord = length(multiindex)
    deg = sum(multiindex) + multiplicity_external
    return "MultilinearMap$(_definition_site(f!)) — assuming a pure external forcing term.\n" *
           "  multiindex = $multiindex         — no derivative factors, system order " *
           "ORD = $ord\n" *
           "  multiplicity_external = $multiplicity_external   — f! receives the external " *
           "state $(multiplicity_external == 1 ? "once" : "$multiplicity_external times")\n" *
           "  ⇒ deg = $deg, called as $(_call_signature(multiindex, multiplicity_external))\n" *
           "Nothing in `f!` reveals the system order or the external multiplicity, so both " *
           "were\nassumed.  Either of these states the term in full, and is silent:\n" *
           "  MultilinearMap(f!; multiindex = $multiindex, multiplicity_external = " *
           "$multiplicity_external)\n" *
           "  MultilinearMap(f!, $multiindex, $multiplicity_external)" *
           "                              # positional form\n" *
           "(`MultilinearMap(f!; multiplicity_external = $multiplicity_external, " *
           "order = $ord)` builds the same term\nbut still reports the assumed multiindex.)\n" *
           "If the single factor of `f!` is a state vector rather than the external state, " *
           "this\nterm is LINEAR and belongs in `linear_terms`, not in a `MultilinearMap`."
end

"""
	_info_assumed(f!, multiindex, multiplicity_external, mi_source, assumed_order,
	              assumed_me, forcing)

Emit an `@info` naming every value the constructor had to default, or nothing at all when
the caller pinned them.  `mi_source` is `:stated`, `:arity` or `:degree`, and names where an
assumed `multiindex` came from.

`fully_asymmetric` is deliberately not reported here: it already has a dedicated diagnostic,
`FullOrderModel._info_implicit_symmetry`, which fires at `NthOrderModel` construction exactly
when the flag changes the result.
"""
function _info_assumed(f!, multiindex, multiplicity_external, mi_source, assumed_order,
        assumed_me, forcing)
    assumed_mi = mi_source !== :stated
    (assumed_mi || assumed_order || assumed_me) || return nothing
    if forcing
        @info _msg_assumed_forcing(f!, multiindex, multiplicity_external)
        return nothing
    end
    assumed = String[]
    if assumed_mi
        origin = mi_source === :degree ? "from `degree`" : "from the arity of f!"
        push!(assumed, "multiindex = $multiindex ($origin)")
    end
    assumed_order && push!(assumed, "order = $(length(multiindex))")
    assumed_me && push!(assumed, "multiplicity_external = $multiplicity_external")
    @info _msg_assumed_compact(f!, multiindex, multiplicity_external, assumed)
    return nothing
end

function _msg_degree_below_external(degree, multiplicity_external)
    return "MultilinearMap: degree = $degree is smaller than multiplicity_external = " *
           "$multiplicity_external.\n`degree` is the *total* degree, external factors " *
           "included, so it cannot be exceeded\nby the number of external factors alone."
end

# -- validation ---------------------------------------------------------

"""
	_check_arity(f!, deg, multiindex, multiplicity_external)

Throw an `ArgumentError` unless `f!` can be called with `deg + 1` arguments.

`hasmethod` is the fast accept path, but it returns `false` for methods with concrete
argument annotations, so a scan of the method table decides rejection.  A varargs method
that can absorb the arguments is accepted — that is what admits closures built by
`MORFESymbolicsExt` and callable structs.  A callable exposing no methods at all is
trusted rather than rejected.
"""
function _check_arity(f!, deg, multiindex, multiplicity_external)
    want = deg + 1
    hasmethod(f!, NTuple{want, Any}) && return nothing
    fixed, va = _method_arities(f!)
    isempty(fixed) && isempty(va) && return nothing
    (want in fixed || any(<=(want), va)) && return nothing
    throw(ArgumentError(_msg_arity(
        f!, want, fixed, va, multiindex, multiplicity_external)))
end

"""
	_infer_degree(f!) -> Int

Number of factors `f!` takes, i.e. its argument count less `res`.  Requires a single
fixed arity; varargs or conflicting arities are ambiguous and raise an `ArgumentError`
telling the caller to state the degree explicitly.
"""
function _infer_degree(f!)
    fixed, _ = _method_arities(f!)
    arities = unique(fixed)
    length(arities) == 1 && return only(arities) - 1        # subtract `res`
    throw(ArgumentError(_msg_infer_degree(f!, fixed)))
end

# -- keyword normalisation ----------------------------------------------

_as_index_tuple(x::Tuple{Vararg{Integer}}, ::AbstractString) = map(Int, x)
function _as_index_tuple(x::AbstractVector{<:Integer}, ::AbstractString)
    return ntuple(k -> Int(x[k]), length(x))
end
function _as_index_tuple(x, what::AbstractString)
    throw(ArgumentError("MultilinearMap: `$what` must be a tuple or vector of " *
                        "integers, got $(typeof(x))."))
end

"""
	_counts_from_derivatives(derivatives) -> NTuple

Convert a per-slot list of 0-based derivative orders into a `multiindex` of counts:
`(0, 0, 1) → (2, 1)`.  The list must be non-decreasing, since it *is* the argument
order `f!` will be called with.
"""
function _counts_from_derivatives(derivatives::NTuple{N, Int}) where {N}
    N == 0 && return (0,)
    any(<(0), derivatives) &&
        throw(ArgumentError(_msg_derivatives_negative(derivatives)))
    issorted(derivatives) ||
        throw(ArgumentError(_msg_derivatives_unsorted(derivatives)))
    return ntuple(k -> count(==(k - 1), derivatives), derivatives[end] + 1)
end

"""
	_pad_to_order(base, order) -> NTuple

Zero-pad `base` to length `order`.  Padding is symmetry-neutral: trailing zeros change
neither `all(<=(1), mi)` nor `count(>(0), mi)`, so `symmetry_type` classifies `(2,)` and
`(2, 0, 0)` identically.  Truncation is refused — it would silently drop a slot.
"""
function _pad_to_order(base::NTuple{N, Int}, order::Int) where {N}
    order < N && throw(ArgumentError(_msg_order_truncates(base, order)))
    order == N && return base
    return ntuple(k -> k <= N ? base[k] : 0, order)
end

function _internal_degree(degree::Int, multiplicity_external::Int)
    d = degree - multiplicity_external
    d >= 0 || throw(ArgumentError(_msg_degree_below_external(
        degree, multiplicity_external)))
    return d
end

"""
	_build_multilinear_map(f!, multiindex, multiplicity_external, fully_asymmetric)

Validate a fully resolved term and build it.  The single place where the invariants of
`MultilinearMap` are enforced; every public constructor ends up here.

Checks run in the order negatives → degree → arity so that `sum(multiindex)` is
meaningful in every message.
"""
function _build_multilinear_map(f!, multiindex::NTuple{ORD, Int},
        multiplicity_external::Int,
        fully_asymmetric::Union{Nothing, Bool}) where {ORD}
    ORD >= 1 || throw(ArgumentError(_msg_empty_multiindex()))
    all(>=(0), multiindex) ||
        throw(ArgumentError(_msg_negative_multiindex(multiindex)))
    multiplicity_external >= 0 ||
        throw(ArgumentError(_msg_negative_external(multiplicity_external)))
    deg = sum(multiindex) + multiplicity_external
    (deg >= 2 || multiplicity_external >= 1) ||
        throw(ArgumentError(_msg_degree_too_low(
            multiindex, multiplicity_external, deg)))
    _check_arity(f!, deg, multiindex, multiplicity_external)
    return MultilinearMap{ORD, typeof(f!)}(
        f!, multiindex, multiplicity_external, deg, fully_asymmetric)
end

"""
	MultilinearMap(f!; multiindex = nothing, derivatives = nothing, order = nothing,
	               degree = nothing, multiplicity_external = nothing,
	               fully_asymmetric = nothing)

Create a multilinear term, naming every argument.  This is the recommended constructor.

# Keyword arguments

- `multiindex`: tuple (or vector) of counts, `multiindex[k]` being how many argument
  slots use the derivative `x^(k-1)`.  Its length is the order `ORD` of the system.
- `derivatives`: the alternative spelling — the 0-based derivative order of *each*
  argument slot, in call order.  `derivatives = (0, 0, 1)` is the same term as
  `multiindex = (2, 1)`, i.e. `f!(res, x, x, ẋ)`.  Must be non-decreasing, because it
  describes the order in which `evaluate_term!` passes the factors.  Mutually exclusive
  with `multiindex`.
- `order`: the system order `ORD`.  Zero-pads a shorter `multiindex` up to it, so a
  quadratic term of a third-order model can be written `multiindex = (2,), order = 3`
  instead of `(2, 0, 0)`.  It may pad but never truncate.  Defaults to `2` — see below.
- `degree`: the total degree, external factors included.  Use it when the arity of `f!`
  cannot be introspected (a varargs closure), or as a cross-check against `multiindex`.
- `multiplicity_external`: how many times the external state `r` is passed to `f!`,
  after the derivative arguments.  Defaults to `0`, or to `1` under the forcing rule below.
- `fully_asymmetric`: overrides the symmetry inferred from `multiindex`; see the note in
  the [`MultilinearMap`](@ref) docstring.  Defaults to `nothing` ("not stated").

# Defaults for omitted arguments

| Value | Assumed unless… | Default |
|:---|:---|:---|
| `multiindex` | `multiindex` or `derivatives` given | from `degree`, else from the arity of `f!`, with every non-external factor on `x^(0)` |
| `order` | `order` given, **or** `multiindex` given | `2` — the second-order mechanical setting |
| `multiplicity_external` | given | `0`, or `1` under the forcing rule |

`multiindex` pins the order exactly, since its length *is* `ORD`; `derivatives` does not,
because it lists argument slots rather than the system order.

Two rules apply only when neither `multiindex` nor `derivatives` was given:

- **Forcing rule.**  A total degree of 1 with `multiplicity_external` unstated is read as a
  pure external forcing term (`multiplicity_external = 1`, no derivative factors).  Without
  it, `MultilinearMap(f!)` on `f!(res, r)` would resolve to a degree-1 term in the state,
  which is *linear* and cannot be represented here — linear contributions belong in the
  `linear_terms` matrices of `NthOrderModel`.
- **Mixed terms are never inferred.**  With `multiplicity_external >= 1` and a non-zero
  internal degree, splitting the factors would mean guessing `f!(res, x, r)`; that is an
  `ArgumentError`.  Only the pure-forcing split is inferable.  Stating `multiindex` lifts
  the restriction — mixed terms are perfectly legal, just not guessable.

**Every assumed value is reported through an `@info`.**  A call that pins `multiindex` (or
`derivatives` plus `order`) and `multiplicity_external` is silent, as are all the positional
constructors.  `fully_asymmetric` is not reported here: it has its own diagnostic at
`NthOrderModel` construction, which fires exactly when the flag changes the result.

# Examples

```julia
# f!(res, x, x, ẋ) in a second-order system
MultilinearMap(f!; multiindex = (2, 1))
MultilinearMap(f!; derivatives = (0, 0, 1))          # identical

# cubic term of a third-order system, without hand-written trailing zeros
MultilinearMap(f!; multiindex = (3,), order = 3)     # ⇒ (3, 0, 0)
MultilinearMap(f!; degree = 3, order = 3)            # ⇒ (3, 0, 0)

# pure external forcing, f!(res, r): shape, order and multiplicity all assumed
MultilinearMap(f!)                                   # ⇒ (0, 0), me 1, ORD 2

# the same term stated in full — silent
MultilinearMap(f!; multiindex = (0, 0), multiplicity_external = 1)

# a first-order term must say so, since the order defaults to 2
MultilinearMap(f!; order = 1)                        # ⇒ (2,), ORD 1

# f! is not symmetric in its two x^(0) slots
MultilinearMap(f!; multiindex = (2, 0), fully_asymmetric = true)
```

# Errors

Throws `ArgumentError` if `multiindex`/`derivatives` entries are negative, if
`derivatives` is not sorted, if `order` would truncate, if `degree` disagrees with
`multiindex`, if a mixed internal/external split would have to be guessed, if the resulting
degree is below 2 with no external factors, or if `f!` cannot be called with `deg + 1`
arguments.
"""
Base.@constprop :aggressive function MultilinearMap(f!;
        multiindex = nothing,
        derivatives = nothing,
        order = nothing,
        degree = nothing,
        multiplicity_external = nothing,
        fully_asymmetric::Union{Nothing, Bool} = nothing)
    (multiindex === nothing || derivatives === nothing) ||
        throw(ArgumentError(_msg_multiindex_and_derivatives()))

    stated_shape = multiindex !== nothing || derivatives !== nothing
    me = multiplicity_external === nothing ? 0 : Int(multiplicity_external)
    me >= 0 || throw(ArgumentError(_msg_negative_external(me)))

    assumed_me = false
    base = if derivatives !== nothing
        _counts_from_derivatives(_as_index_tuple(derivatives, "derivatives"))
    elseif multiindex !== nothing
        _as_index_tuple(multiindex, "multiindex")
    else
        total = degree === nothing ? _infer_degree(f!) : Int(degree)
        # Total degree 1 with no external factors is a *linear* term, which a
        # `MultilinearMap` cannot represent — `linear_first_order_matrices` builds the
        # companion pair from `linear_terms` alone, so such a term would drive the RHS
        # while staying invisible to the eigenproblem.  When the caller stated neither the
        # internal shape nor an external count, the only reading that makes sense is a
        # pure forcing term, so assume it rather than erroring out.
        if total == 1 && multiplicity_external === nothing
            me = 1
            assumed_me = true
        end
        internal = _internal_degree(total, me)
        # Which of the `total` factors are internal and which are external is not something
        # `f!` reveals.  Only the pure-forcing split (none internal) is inferable.
        internal >= 1 && me >= 1 &&
            throw(ArgumentError(_msg_mixed_split(f!, total, internal, me)))
        (internal,)
    end

    # ORD = 2 — the second-order mechanical setting — unless the caller pinned the order,
    # either with `order` or with a `multiindex` whose length states it exactly.
    assumed_order = order === nothing && multiindex === nothing
    ord = order !== nothing ? Int(order) :
          (multiindex !== nothing ? length(base) : max(2, length(base)))
    mi = _pad_to_order(base, ord)

    if degree !== nothing && Int(degree) != sum(mi) + me
        throw(ArgumentError(_msg_degree_mismatch(Int(degree), mi, me)))
    end
    term = _build_multilinear_map(f!, mi, me, fully_asymmetric)
    # Report only after the term validates, so a failed construction never announces an
    # assumption that did not survive.
    mi_source = stated_shape ? :stated : (degree === nothing ? :arity : :degree)
    _info_assumed(f!, mi, me, mi_source, assumed_order,
        multiplicity_external === nothing, assumed_me)
    return term
end

"""
	MultilinearMap(f!, multiindex; fully_asymmetric = nothing)

Create a multilinear term for a system of order `ORD` without external dynamics.

# Arguments
- `f!`: in-place evaluation function, accumulating into its first argument
- `multiindex::NTuple{ORD, Int}`: how many argument slots use each derivative;
  `multiindex[k]` counts the slots taking `x^(k-1)`

# Keyword arguments
- `fully_asymmetric`: overrides the symmetry inferred from `multiindex`; see the note in
  the [`MultilinearMap`](@ref) docstring

Equivalent to `MultilinearMap(f!; multiindex = multiindex)`.
"""
function MultilinearMap(f!, multiindex::NTuple{ORD, Int};
        fully_asymmetric::Union{Nothing, Bool} = nothing) where {ORD}
    return _build_multilinear_map(f!, multiindex, 0, fully_asymmetric)
end

"""
	MultilinearMap(f!, multiindex, multiplicity_external; fully_asymmetric = nothing)

Create a multilinear term for a system of order `ORD` that also takes the external state.

`f!` is called with the derivative arguments selected by `multiindex` first, then the
external state `r` repeated `multiplicity_external` times.  A term that depends on the
external state may have total degree 1 — a pure forcing term is written
`MultilinearMap(f!, (0, 0), 1)`, for which `MultilinearMap(f!)` is a shorthand.

The model this term goes into must have an external system; `NthOrderModel` rejects a
`multiplicity_external > 0` term otherwise.

# Arguments
- `f!`: in-place evaluation function, accumulating into its first argument
- `multiindex::NTuple{ORD, Int}`: how many argument slots use each derivative
- `multiplicity_external::Int`: how many times `r` is passed to `f!`

# Keyword arguments
- `fully_asymmetric`: overrides the symmetry inferred from `multiindex`; see the note in
  the [`MultilinearMap`](@ref) docstring

Equivalent to
`MultilinearMap(f!; multiindex = multiindex, multiplicity_external = multiplicity_external)`,
and — like every positional form — it assumes nothing, so it never emits the `@info` the keyword
constructor uses to report defaulted values.

A pure forcing term of a second-order system, `MultilinearMap(f!, (0, 0), 1)`, is what the keyword
constructor infers from a one-factor `f!` when nothing else is stated; see the keyword
constructor's docstring for that shorthand and the assumptions it makes.
"""
function MultilinearMap(
        f!, multiindex::NTuple{ORD, Int}, multiplicity_external::Int;
        fully_asymmetric::Union{Nothing, Bool} = nothing) where {ORD}
    return _build_multilinear_map(
        f!, multiindex, multiplicity_external, fully_asymmetric)
end

function Base.show(io::IO, t::MultilinearMap{ORD}) where {ORD}
    print(io, "MultilinearMap{ORD=", ORD, "} multiindex=", t.multiindex,
        ", multiplicity_external=", t.multiplicity_external, ", deg=", t.deg,
        ", ", _symmetry_label(t.multiindex, t.fully_asymmetric))
end

function Base.show(io::IO, ::MIME"text/plain", t::MultilinearMap{ORD}) where {ORD}
    println(io, "MultilinearMap{ORD=", ORD, "}", _definition_site(t.f!))
    println(io, "  multiindex: ", t.multiindex, "  →  ",
        _call_signature(t.multiindex, t.multiplicity_external))
    println(io, "  multiplicity_external: ", t.multiplicity_external)
    println(io, "  deg: ", t.deg)
    println(io, "  fully_asymmetric: ", t.fully_asymmetric)
    print(io, "  symmetry: ", _symmetry_label(t.multiindex, t.fully_asymmetric))
end

"""
	evaluate_term!(res, term, xs, r)

Evaluate a single `MultilinearMap` and accumulate (adds) the result into `res`.

# Arguments
- `res`: output vector (modified in-place)
- `term`: multilinear term
- `xs`: tuple `(x, x^(1), …, x^(ORD-1))` of state derivatives
- `r`: external state vector (or `nothing` if not used). If `r` is `nothing` but the term expects external arguments, an error is thrown.

!!! note "`r` is the *physical* external state"
    See [`evaluate_nonlinear_terms!`](@ref): when the external system was re-based, the
    caller must convert the reduced coordinates `r′` with
    `ExternalSystems.to_physical_external` first.  During the cohomological solve this
    argument is a *basis direction* rather than a state — a unit vector `eⱼ`, or the column
    `Q[:, j]` after a re-basing — supplied by
    `ExternalSystems.external_argument_vectors`, so `f!` may receive a complex-valued
    external argument and must not assume an integer one.
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
