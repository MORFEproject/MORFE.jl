module SymbolicsExtension

export model_from_symbolics, externalsystem_from_symbolics

"""
    model_from_symbolics(exprs, groups)
    model_from_symbolics(f!, order::Int, nvars::Int; p = ())
    model_from_symbolics(exprs, groups, ext_var, ext_exprs)

Generates an `NthOrderModel` from a symbolic ODE.

# Methods

- `model_from_symbolics(exprs, groups)`: `groups` is an `NTuple` where every
  entry is a vector of the state variables or their derivatives, e.g.
  `([z1, z2, z3], [dz1, dz2, dz3], ...)`. `exprs` is the ODE, written as a
  vector supposed equal to zero.

- `model_from_symbolics(f!, order, nvars; p = ())`: mirrors
  DifferentialEquations.jl's convention of defining ODEs, generalised to
  order `order`. `f!` is in-place (arity == `order + 1`) and mutates its
  first argument:

      f!(dᵏu, dᵏ⁻¹u, ..., du, u, p, t)

  Each `dⁱu` must be a vector of length `nvars`.

- `model_from_symbolics(exprs, groups, ext_var, ext_exprs)`: as the first
  method, but the nonlinear forcing terms may also depend on external
  variables `ext_var` (the state of an `ExternalSystem`, defined by the
  autonomous ODE `ext_exprs`). The resulting `MultilinearMap`s get an extra
  multiindex entry counting the degree in `ext_var`.

Requires Symbolics.jl. Load it with `using Symbolics` to activate the MORFE
extension.
"""
function model_from_symbolics end

"""
    externalsystem_from_symbolics(exprs, var)
    externalsystem_from_symbolics(f, nvars::Int; p = ())

Generates an `ExternalSystem` from a symbolic, autonomous, polynomial ODE.

# Methods

- `externalsystem_from_symbolics(exprs, var)`: expects the ODE describing the
  external system in the form `dvar/dt = exprs`.

- `externalsystem_from_symbolics(f, nvars; p = ())`: mirrors
  DifferentialEquations.jl's convention of defining ODEs. `f` may be given in
  either of two layouts:
  1. in-place, mutating the first argument: `f(dr, r, p, t)`
  2. out-of-place, returning the derivative: `f(r, p, t) -> dr`

Requires Symbolics.jl. Load it with `using Symbolics` to activate the MORFE
extension.
"""
function externalsystem_from_symbolics end

end # module SymbolicsExtension