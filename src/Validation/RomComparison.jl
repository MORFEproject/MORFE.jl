"""
Module `RomComparison` — regression comparison of reduced-dynamics coefficients.

Two modes, matching the two situations that occur in practice:

- `:exact` — rows are matched by exponent tuple over the **shared** monomial
  set and compared with a relative tolerance. Because the DPIM solve is graded
  (degree-N coefficients never depend on degrees > N), a lower-order run is an
  exact truncation of a higher-order reference — so a FAST/smoke run can be
  validated against a FULL reference on their shared monomials.

- `:gauge_invariant` — eigensolvers return eigenvectors up to a complex
  scale (the "gauge"), and ARPACK's gauge is not reproducible across runs or
  machines. Raw ROM coefficients therefore are **not** run-comparable in
  general. What survives any diagonal gauge `z_r → a_r z_r`:
  coefficients of monomials with total *modal* degree 1 (the eigenvalue row
  and every linear-in-external coupling row, e.g. the Stuart-Landau `λ` and
  `c₁₀₁`), and — for a real gauge scale — the Im/Re ratio of higher rows.
  This mode compares exactly those quantities: all modal-degree-1 rows, plus
  the Im/Re ratio of the lowest-order purely-modal nonlinear row.
"""
module RomComparison

using ..RomIO: read_rom_coefficients

export compare_rom_coefficients

_load(x::AbstractString) = read_rom_coefficients(x)
_load(x::Tuple) = x

# rows keyed by exponent tuple
function _index(exponents::Matrix{Int})
    Dict(Tuple(exponents[r, :]) => r for r in 1:size(exponents, 1))
end

"""
	compare_rom_coefficients(new, ref; mode = :exact, rtol = 1e-6,
	                          n_master = nothing) -> (pass, max_dev, report)

`new`/`ref` are CSV paths (or pre-parsed `(exponents, coefficients)` tuples).

- `mode = :exact`: max relative deviation over the shared monomial rows,
  each coefficient compared against `max(|ref|, 1e-12)`.
- `mode = :gauge_invariant`: compares only gauge-safe quantities (see module
  docstring). `n_master` gives the number of leading exponent columns that
  are modal coordinates (default: all columns). Requires ≥ 1 modal-degree-1
  shared row.

Returns `(pass::Bool, max_dev::Float64, report::String)`.
"""
function compare_rom_coefficients(new, ref; mode::Symbol = :exact,
        rtol::Real = 1e-6, n_master::Union{Nothing, Int} = nothing)
    (en, cn) = _load(new)
    (er, cr) = _load(ref)
    size(en, 2) == size(er, 2) ||
        return (false, Inf, "exponent arity differs: $(size(en,2)) vs $(size(er,2))")
    idx_ref = _index(er)
    shared = [(r, idx_ref[Tuple(en[r, :])])
              for r in 1:size(en, 1)
              if haskey(idx_ref, Tuple(en[r, :]))]
    n_new_only = size(en, 1) - length(shared)
    n_ref_only = size(er, 1) - length(shared)

    if mode === :exact
        isempty(shared) && return (false, Inf, "no shared monomials")
        max_dev = 0.0
        for (rn, rr) in shared
            for i in 1:min(size(cn, 2), size(cr, 2))
                dev = abs(cn[rn, i] - cr[rr, i]) / max(abs(cr[rr, i]), 1e-12)
                max_dev = max(max_dev, dev)
            end
        end
        pass = max_dev < rtol
        report = "exact: $(length(shared)) shared rows " *
                 "($(n_new_only) new-only, $(n_ref_only) ref-only), " *
                 "max rel dev = $(max_dev) (rtol = $(rtol)) → " *
                 (pass ? "PASS" : "FAIL")
        return (pass, max_dev, report)

    elseif mode === :gauge_invariant
        nm = n_master === nothing ? size(en, 2) : n_master
        max_dev = 0.0
        nlin = 0
        for (rn, rr) in shared
            modal_deg = sum(@view en[rn, 1:nm])
            if modal_deg == 1
                nlin += 1
                for i in 1:min(size(cn, 2), size(cr, 2))
                    dev = abs(cn[rn, i] - cr[rr, i]) / max(abs(cr[rr, i]), 1e-12)
                    max_dev = max(max_dev, dev)
                end
            end
        end
        nlin == 0 && return (false, Inf, "gauge_invariant: no modal-degree-1 shared rows")
        # Im/Re ratio of the lowest-order purely-modal nonlinear row.
        nl = [(rn, rr)
              for (rn, rr) in shared
              if sum(@view en[rn, 1:nm]) ≥ 2 && sum(@view en[rn, (nm + 1):end]) == 0]
        if !isempty(nl)
            (rn, rr) = first(sort(nl; by = t -> sum(@view en[t[1], 1:nm])))
            i = findfirst(i -> abs(cr[rr, i]) > 1e-12, 1:size(cr, 2))
            if i !== nothing && abs(real(cn[rn, i])) > 1e-12 && abs(real(cr[rr, i])) > 1e-12
                ratio_dev = abs(imag(cn[rn, i]) / real(cn[rn, i]) -
                                imag(cr[rr, i]) / real(cr[rr, i])) /
                            max(abs(imag(cr[rr, i]) / real(cr[rr, i])), 1e-12)
                max_dev = max(max_dev, ratio_dev)
            end
        end
        pass = max_dev < rtol
        report = "gauge_invariant: $(nlin) modal-degree-1 rows" *
                 (isempty(nl) ? "" : " + Im/Re of leading nonlinear row") *
                 ", max rel dev = $(max_dev) (rtol = $(rtol)) → " *
                 (pass ? "PASS" : "FAIL")
        return (pass, max_dev, report)
    else
        throw(ArgumentError("unknown mode :$mode (use :exact or :gauge_invariant)"))
    end
end

end # module RomComparison
