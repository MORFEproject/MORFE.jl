"""
Module `MultilinearMaps` — multilinear nonlinear term representations for `NthOrderModel`.

Nonlinear terms in the full-order ODE are encoded as `AbstractMultilinearMap` subtypes:

- `MultilinearMap{ORD, F}` — a single polynomial term of total degree `deg`, stored as
  a callable `f!` that evaluates the multilinear form plus metadata (`multiindex`,
  `multiplicity_external`, `deg`).
- `FEMMultilinearMap{ORD}` — abstract base for FEM-backed terms that expose element-level
  primitives (`fem_elements`, `scatter_qp!`, `accumulate_qp!`, `assemble_element!`, …).
  Implementing these methods enables the single-pass assembly of the right-hand side in
  `MultilinearTerms`.

For FEM backends, see the MORFEExamples tutorial `writing_a_finite_element_nonlinearity`,
which writes one from scratch, and MORFEFerrite.jl (`StructuralSVK`, `FluidNavierStokes`).
"""
module MultilinearMaps

export AbstractMultilinearMap, FEMMultilinearMap, MultilinearMap, evaluate_term!,
       fem_elements, fem_n_qp, fem_ndofs_per_cell,
       scatter_qp!, accumulate_qp!, assemble_element!, fem_getdetJdV, fem_qp_buffer,
       fem_reinit!, fem_element_loop, fem_flux_zero, flux_qp, contract_qp!,
       accumulate_terms_qp!

"""
	AbstractMultilinearMap{ORD}

Abstract supertype for all multilinear terms accepted by `NthOrderModel`.

Every concrete subtype must expose the fields `multiindex`, `multiplicity_external`, `deg`,
and `fully_asymmetric` with the same semantics as `MultilinearMap`.
"""
abstract type AbstractMultilinearMap{ORD} end

"""
	FEMMultilinearMap{ORD} <: AbstractMultilinearMap{ORD}

Abstract type for nonlinear terms that a finite element backend integrates element by
element. A concrete term evaluates the same multilinear form as a `MultilinearMap`, but
instead of one function of full vectors it exposes the steps of an element loop. MORFE runs
the loop itself, which lets one traversal of the cells serve every factorisation of a
monomial and every FEM term on those cells: the single-pass assembly of `MultilinearTerms`.

# Vocabulary

- *coefficient*: one column `W[:, order, col]` of the parametrisation, a complex vector over
  the free DOFs. Constrained DOFs have no entry: the backend maps its global DOF numbers to
  free indices and reads zero where a DOF is constrained.
- *element*: one item of `fem_elements(term)`, usually a mesh cell. It is passed back to every
  other method, so it carries what they need (for Ferrite, a `CellCache` with the cell's
  coordinates and global DOF numbers).
- *quadrature-point value*: what `scatter_qp!` computes from one coefficient at one quadrature
  point: a displacement gradient for a structure, a velocity value and gradient for
  convection. Its type is the element type of `fem_qp_buffer(term)`.
- `q`: the index of a quadrature point within the element, in `1:fem_n_qp(term)`.
- `dΩ`: the integration weight at `q`, the quadrature weight times the Jacobian determinant
  of the element map, as returned by `fem_getdetJdV`.
- `Fe`: the element vector, of length `fem_ndofs_per_cell(term)` and indexed in the
  element's local DOF order. MORFE zeroes it before each element.
- `qp_args`: the arguments of one factorisation at `q`, an `NTuple` of length `term.deg`:
  the quadrature-point values of its internal factors, in the order `term.multiindex` lists
  the derivatives, followed by one external argument per external slot.
- `multiplier`: the number of argument orderings one factorisation stands for, times the
  multiplicity of its external split. The term scales its contribution by it.

# The element loop

A term runs on one element loop: cells of one element type, with one discretisation of the
field (interpolation, quadrature, DOF map), so `fem_n_qp` and `fem_ndofs_per_cell` are
constants of the term. A map acting on several element types or domains is passed as one term
per element loop, which is exact because the weak form is a sum over cells.

MORFE groups the FEM terms of a model by [`fem_element_loop`](@ref) and, per monomial, runs
the loop below once per group. The methods are called on the group's first term, the driver,
except `accumulate_qp!`, which is called on each term of the group in turn; in the direct
evaluation `evaluate_term!` the term drives its own loop.

	for element in fem_elements(driver)
	    fem_reinit!(element, driver)
	    for each distinct coefficient c read at this monomial
	        scatter_qp!(qp_values_of_c, c, element, driver)
	    end
	    Fe .= 0
	    for q in 1:fem_n_qp(driver)
	        dΩ = fem_getdetJdV(element, q, driver)
	        for each term of the group, for each of its factorisations
	            accumulate_qp!(Fe, qp_args, multiplier, element, q, dΩ, term)
	        end
	    end
	    assemble_element!(result, Fe, element, driver)
	end

The cache build checks that the terms of a group agree on `fem_n_qp` and
`fem_ndofs_per_cell`. Terms that opt into flux summation replace the inner
`accumulate_qp!` calls by one `contract_qp!` per quadrature point (see
[`fem_flux_zero`](@ref)), and a backend may take over the whole quadrature-point step with
[`accumulate_terms_qp!`](@ref).

# Required fields

Same semantics as `MultilinearMap`:
- `multiindex`, `multiplicity_external`, `deg`
- `fully_asymmetric::Union{Nothing, Bool}` — `nothing` = not set (triggers `@info` at `NthOrderModel`
  construction if multiindex implies symmetry); `false` = acknowledged symmetric; `true` = override to
  `FullyAsymmetric`. FEM backends whose integrand is symmetric by construction should default to `false`.

# Required methods (extend `MORFE.*`)

- [`fem_elements`](@ref)`(term)`: iterator over the elements
- [`fem_reinit!`](@ref)`(element, term)`: prepare one element
- [`fem_n_qp`](@ref)`(term)`: quadrature points per element
- [`fem_ndofs_per_cell`](@ref)`(term)`: DOFs per element, the length of `Fe`
- [`fem_qp_buffer`](@ref)`(term)`: buffer whose element type is the quadrature-point value type
- [`scatter_qp!`](@ref)`(qp_values, coefficient, element, term)`: one coefficient at every quadrature point
- [`fem_getdetJdV`](@ref)`(element, q, term)`: the weight `dΩ` at `q`
- [`accumulate_qp!`](@ref)`(Fe, qp_args, multiplier, element, q, dΩ, term)`: one factorisation at `q`
- [`assemble_element!`](@ref)`(result, Fe, element, term)`: add `Fe` to the free-DOF result

# Optional methods

- [`fem_element_loop`](@ref)`(term)`: a label shared by the terms on the same element loop,
  which then share one pass; the default, the term itself, gives each term its own

The others let the single-pass assembly test the integrands of all terms and factorisations
against the shape functions once per quadrature point (see [`fem_flux_zero`](@ref)):
- [`fem_flux_zero`](@ref)`(term)`: zero flux, or `nothing` (default) to opt out
- [`flux_qp`](@ref)`(qp_args, element, q, term)`: flux of one factorisation at `q`
- [`contract_qp!`](@ref)`(Fe, flux, element, q, dΩ, term)`: add the tested flux at `q`
- [`accumulate_terms_qp!`](@ref)`(flux, Fe, coefficient_values, external_arguments, entries, terms, element, q, dΩ)`:
  all factorisations of all terms at `q`, to share work between them
"""
abstract type FEMMultilinearMap{ORD} <: AbstractMultilinearMap{ORD} end

# Interface stubs — extend these in your concrete FEM type.

"""
	fem_elements(term::FEMMultilinearMap) -> iterator

Iterator over the elements (cells) of the mesh. Each item is handed back as `element` to
[`fem_reinit!`](@ref), [`scatter_qp!`](@ref), [`fem_getdetJdV`](@ref),
[`accumulate_qp!`](@ref) and [`assemble_element!`](@ref), so it must carry what they need,
for instance the element's global DOF numbers.
"""
function fem_elements(term::FEMMultilinearMap)
    error("fem_elements not implemented for $(typeof(term))")
end
"""
	fem_n_qp(term::FEMMultilinearMap) -> Int

Number of quadrature points per element, the same for every element.
"""
function fem_n_qp(term::FEMMultilinearMap)
    error("fem_n_qp not implemented for $(typeof(term))")
end
"""
	fem_ndofs_per_cell(term::FEMMultilinearMap) -> Int

Number of DOFs of one element, which is the length of the element vector `Fe`.
"""
function fem_ndofs_per_cell(term::FEMMultilinearMap)
    error("fem_ndofs_per_cell not implemented for $(typeof(term))")
end
"""
	fem_reinit!(element, term::FEMMultilinearMap) -> nothing

Prepare `term` for `element`. MORFE calls it once per element, before any other method
receives that element. A Ferrite backend reinitialises its `CellValues` here, which maps the
shape-function values and gradients to the element.
"""
function fem_reinit!(element, term::FEMMultilinearMap)
    error("fem_reinit! not implemented for $(typeof(term))")
end
"""
	scatter_qp!(qp_values, coefficient, element, term::FEMMultilinearMap) -> nothing

Evaluate one coefficient of the parametrisation at every quadrature point of `element`.

# Arguments
- `qp_values`: output, a vector (a view into MORFE's buffer) of length `fem_n_qp(term)`
  with the element type of `fem_qp_buffer(term)`; `qp_values[q]` receives the
  quadrature-point value at `q`.
- `coefficient`: one column `W[:, order, col]` of the parametrisation, a complex vector over
  the free DOFs. Constrained DOFs have no entry and contribute zero.
- `element`: the current element, already passed to [`fem_reinit!`](@ref).
"""
function scatter_qp!(qp_values, coefficient, element, term::FEMMultilinearMap)
    error("scatter_qp! not implemented for $(typeof(term))")
end
"""
	accumulate_qp!(Fe, qp_args, multiplier, element, q, dΩ, term::FEMMultilinearMap) -> nothing

Add the integrand of one factorisation at quadrature point `q` of `element`, scaled by
`multiplier * dΩ`, to the element vector `Fe`.

# Arguments
- `Fe`: the element vector, of length `fem_ndofs_per_cell(term)`, in local DOF order.
- `qp_args`: an `NTuple` of `term.deg` arguments: the quadrature-point values at `q` of
  the internal factors, in the order `term.multiindex` lists the derivatives, then one
  external argument per external slot (see [`evaluate_term!`](@ref)).
- `multiplier`: the number of argument orderings the factorisation stands for, times the
  multiplicity of its external split; `1` in the direct evaluation.
- `q`, `dΩ`: the quadrature-point index and its weight, `fem_getdetJdV(element, q, term)`.

The integrand must be linear in each argument. The arguments are complex, so it must not
conjugate them.
"""
function accumulate_qp!(Fe, qp_args, multiplier, element, q, dΩ, term::FEMMultilinearMap)
    error("accumulate_qp! not implemented for $(typeof(term))")
end
"""
	assemble_element!(result, Fe, element, term::FEMMultilinearMap) -> nothing

Add the element vector `Fe` to `result`, a vector over the free DOFs, through the DOF map of
`element`. Entries of constrained DOFs are dropped.
"""
function assemble_element!(result, Fe, element, term::FEMMultilinearMap)
    error("assemble_element! not implemented for $(typeof(term))")
end

"""
	fem_getdetJdV(element, q, term::FEMMultilinearMap) -> Real

Integration weight `dΩ` at quadrature point `q` of `element`: the quadrature weight times
the Jacobian determinant of the element map. The name follows Ferrite's `getdetJdV`.
"""
function fem_getdetJdV(element, q, term::FEMMultilinearMap)
    error("fem_getdetJdV not implemented for $(typeof(term))")
end

"""
	fem_qp_buffer(term::FEMMultilinearMap)

Return a pre-allocated matrix for quadrature-point values, with at least `term.deg` rows and
`fem_n_qp(term)` columns. The direct evaluation `evaluate_term!` scatters each argument into
one row of it; the single-pass assembly reads only its element type and sizes its own
buffer. Make the element type concrete (for Tensors.jl, all four parameters, as in
`Tensor{2, 3, ComplexF64, 9}`): an abstract one boxes every value stored in the buffer.
"""
function fem_qp_buffer(term::FEMMultilinearMap)
    error("fem_qp_buffer not implemented for $(typeof(term))")
end

"""
	fem_element_loop(term::FEMMultilinearMap) -> label

A value identifying the element loop of `term`: the cells it integrates over, and how
(element type, interpolation, quadrature, DOF map). The single-pass assembly runs one pass
per group of terms with `isequal` labels, driven by the group's first term, so equal labels
promise that either term can run on the other's loop:

1. the same cells;
2. `fem_reinit!` on either term prepares everything the other reads;
3. `scatter_qp!` on either yields the quadrature-point values the other expects;
4. the same quadrature points and weights;
5. the same local DOF order and `assemble_element!`;
6. if their fluxes have one type (see [`fem_flux_zero`](@ref)), the same `contract_qp!`.

A backend returns, for instance, its DOF handler and cell values, compared by identity. A
label that is too coarse integrates a term over the wrong cells; one that is too fine only
costs passes. The default, the term itself, gives each term its own pass: always correct,
but cells shared by several terms are then traversed once per term.

A map acting on several element types or domains (materials, physics) is passed as one term
per element loop: the weak form is a sum over cells, so the restrictions add up to the map
exactly. Splitting all maps along the same boundaries visits each cell once per monomial.
"""
fem_element_loop(term::FEMMultilinearMap) = term

"""
	fem_flux_zero(term::FEMMultilinearMap) -> flux or nothing

Opt the FEM term `term` into flux summation. Many integrands test a single quadrature-point
quantity, the flux, against the shape functions: the first Piola-Kirchhoff stress against the
shape-function gradients, for instance. A term of that kind returns its zero flux here and
implements [`flux_qp`](@ref) and [`contract_qp!`](@ref) so that, for any arguments,

	accumulate_qp!(Fe, qp_args, multiplier, element, q, dΩ, term)
	≡ contract_qp!(Fe, multiplier * flux_qp(qp_args, element, q, term), element, q, dΩ, term)

with `contract_qp!` linear in the flux. At each quadrature point, the single-pass assembly
sums `multiplier * flux_qp(…)` over every factorisation of every term whose zero flux has the
type of the driver's (the first term of its group, see [`fem_element_loop`](@ref)), and tests
the sum once with the driver's
`contract_qp!`: a factorisation costs one flux instead of one flux and one pass over the
element's test functions. Terms whose fluxes share a type must therefore share their
`contract_qp!`; a term tested against other shape functions returns a flux of another type,
or `nothing`, and is integrated with `accumulate_qp!`. The flux must support `+` and
multiplication by an integer, with results of the type of the zero returned here.

The default, `nothing`, keeps the term on `accumulate_qp!`. The direct evaluation
`evaluate_term!` always uses `accumulate_qp!`, which a term that opts in must still implement.
"""
fem_flux_zero(term::FEMMultilinearMap) = nothing

"""
	flux_qp(qp_args, element, q, term::FEMMultilinearMap) -> flux

Flux of one factorisation at quadrature point `q`: the quantity `accumulate_qp!` tests
against the shape functions, without the multiplier and the weight. `qp_args` is as in
[`accumulate_qp!`](@ref). Required only when [`fem_flux_zero`](@ref) returns a flux.
"""
function flux_qp(qp_args, element, q, term::FEMMultilinearMap)
    error("flux_qp not implemented for $(typeof(term))")
end

"""
	contract_qp!(Fe, flux, element, q, dΩ, term::FEMMultilinearMap) -> nothing

Test `flux` against the shape functions at quadrature point `q`, with weight `dΩ`, and add
the result to the element vector `Fe`, as `accumulate_qp!` does with the flux of one
factorisation. Must be linear in `flux`. Required only when [`fem_flux_zero`](@ref) returns
a flux.
"""
function contract_qp!(Fe, flux, element, q, dΩ, term::FEMMultilinearMap)
    error("contract_qp! not implemented for $(typeof(term))")
end

"""
	accumulate_terms_qp!(flux, Fe, coefficient_values, external_arguments,
	                     entries::Tuple, terms::Tuple, element, q, dΩ) -> flux

Add, at quadrature point `q`, every factorisation of every FEM term in `terms`: the
single-pass assembly calls it once per quadrature point with the FEM terms of one element loop
(see [`fem_element_loop`](@ref)), `flux` being the driver's [`fem_flux_zero`](@ref). It returns `flux` plus the summed flux of
the factorisations, to be tested once with the driver's `contract_qp!`, and adds to `Fe`
whatever it integrates itself (weight `dΩ`).

- `coefficient_values[i]` is the quadrature-point value at `q` of the `i`-th distinct
  coefficient of the monomial, as `scatter_qp!` computed it.
- `external_arguments[j]` is the external argument of external variable `j`.
- `entries[k]` lists the factorisations of `terms[k]`. Each has a `multiplier::Int`, the
  number of orderings times the multiplicity of its external split; `factors`, an `NTuple`
  of indices into `coefficient_values`, one per internal slot; and `externals`, an `NTuple`
  of indices into `external_arguments`, one per external slot.

The default adds the terms one at a time: `multiplier * flux_qp(…)` for a term whose flux has
the type of `flux`, `accumulate_qp!` otherwise. A backend may add a method for its own tuple
of terms to share work between factorisations and terms evaluated at the same point: the
SVK backend, for instance, collects strains per coefficient so that the quadratic and cubic
stresses cost one constitutive evaluation and one product with each displacement gradient.
"""
function accumulate_terms_qp! end

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
	evaluate_term!(res, term::FEMMultilinearMap{ORD}, xs, r)

Direct evaluation of a FEM term at the state `xs = (x, x^(1), …, x^(ORD-1))` and the
physical external state `r`, added to `res`. The term drives its own element loop (see
[`FEMMultilinearMap`](@ref)) with multiplier `1`. This is the path of
[`evaluate_nonlinear_terms!`](@ref), used outside the cohomological solve, for instance by
the invariance error, where no factorisation cache exists.

Each internal slot of `term.multiindex` is scattered from the matching derivative in `xs`
with `scatter_qp!`, into one row of `fem_qp_buffer(term)`. Each external slot of `qp_args`
receives `r` itself, a vector of length `N_EXT` rather than a field, so `accumulate_qp!`
evaluates the form at the actual external state.
"""
function evaluate_term!(res, term::FEMMultilinearMap{ORD}, xs, r) where {ORD}
    _eval_fem_term_direct!(res, term, xs, r, Val(term.deg), Val(sum(term.multiindex)))
end

function _eval_fem_term_direct!(
        res, term::FEMMultilinearMap{ORD}, xs, r,
        ::Val{DEG}, ::Val{N_INT}) where {ORD, DEG, N_INT}
    multiindex = term.multiindex
    me = DEG - N_INT              # number of external slots; compile-time constant
    qp_buffer = fem_qp_buffer(term)   # one row per internal slot, one column per qp
    n_qp = fem_n_qp(term)
    n_dofs = fem_ndofs_per_cell(term)
    Fe = zeros(eltype(res), n_dofs)

    for element in fem_elements(term)
        fem_reinit!(element, term)
        # Row `slot` holds the quadrature-point values of the slot's state derivative.
        slot = 0
        for j in 1:ORD
            for _ in 1:multiindex[j]
                slot += 1
                scatter_qp!(@view(qp_buffer[slot, 1:n_qp]), xs[j], element, term)
            end
        end
        fill!(Fe, zero(eltype(Fe)))
        for q in 1:n_qp
            dΩ = fem_getdetJdV(element, q, term)
            if me == 0
                qp_args = ntuple(k -> qp_buffer[k, q], Val(N_INT))
            else
                qp_args = ntuple(k -> k ≤ N_INT ? qp_buffer[k, q] : r, Val(DEG))
            end
            accumulate_qp!(Fe, qp_args, 1.0, element, q, dΩ, term)
        end
        assemble_element!(res, Fe, element, term)
    end
end

end # module
