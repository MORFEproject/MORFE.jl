# Allocation-conscious terminal progress reporting.

"""
	_SimpleProgress

Lightweight progress state for the `\r`-based terminal progress indicator.

# Fields

- `n_total::Int` — number of monomials that will actually be solved, which is fewer
  than the multiindex-set size whenever linear or conjugate-secondary monomials are
  skipped. The reported fraction is against this, not against the set size.
- `enabled::Bool` — `false` when `stderr` is not a TTY, so CI logs stay clean
  without every call site having to test for it.
- `max_nl_degree::Int` — highest nonlinearity degree in the model. Work per
  monomial grows with degree, so the fraction is raised to this power to make the
  displayed percentage track elapsed time rather than monomial count.
"""
struct _SimpleProgress
    n_total::Int
    enabled::Bool
    max_nl_degree::Int
end

"""
	_make_progress(n_total, show_progress, max_nl_degree) -> _SimpleProgress

Construct a `_SimpleProgress` tracker. `max_nl_degree` controls the work-weighted
percentage. Output is disabled automatically when `stderr` is not a TTY.
"""
function _make_progress(n_total::Int, show_progress::Bool, max_nl_degree::Int)
    return _SimpleProgress(n_total, show_progress && stderr isa Base.TTY, max_nl_degree)
end

"""
	_progress_tick!(progress, completed, degree)

Print an in-place `\r`-overwritten progress line to `stderr` showing the current
polynomial degree and the fraction of monomials solved. This is a no-op when progress is
disabled.
"""
function _progress_tick!(progress::_SimpleProgress, completed::Int, degree::Int)
    progress.enabled || return
    percentage = round(
        100.0 * (completed / progress.n_total)^progress.max_nl_degree; digits = 2)
    print(stderr,
        "\rSolving: order $degree \t Monomials: $completed/$(progress.n_total) \t Progress: $percentage%   ")
end

"""
	_progress_done!(progress, completed)

Print the final completion line to `stderr` and clear trailing characters from the last
progress update. This is a no-op when progress is disabled.
"""
function _progress_done!(progress::_SimpleProgress, completed::Int)
    progress.enabled || return
    println(stderr, "\rSolved $completed monomials." * " "^50)
end
