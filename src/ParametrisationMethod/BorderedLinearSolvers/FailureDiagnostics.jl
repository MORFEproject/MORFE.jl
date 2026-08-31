# Failure categories and contextual diagnostics for bordered solves.

"""
	_BorderedSolveFailure

Internal contextual failure raised by a bordered cohomological solve.

# Fields

- `category::Symbol` — `:outer_resonance`, `:factorisation`, `:solve`, or `:accuracy`.
- `backend::Symbol` — selected linear solver (`:dense`, `:klu`, `:umfpack`, or
  `:pardiso`).
- `index::Int` — position of the monomial in the active multiindex set.
- `multiindex` — exponent vector of the failing monomial.
- `superharmonic` — canonical superharmonic used to assemble the matrix.
- `inner_resonance_mask` — master-mode resonance mask.
- `outer_resonance_targets::Vector{Int}` — configured outer targets flagged at this
  monomial.
- `recovery_attempted::Bool` — whether a failed cached factorisation was discarded and
  retried from a fresh symbolic analysis.
- `detail::String` — backend status or failed accuracy criterion.
- `cause` — underlying backend exception, or `nothing` when failure was status-based.
"""
struct _BorderedSolveFailure{MI, S, R, C} <: Exception
    category::Symbol
    backend::Symbol
    index::Int
    multiindex::MI
    superharmonic::S
    inner_resonance_mask::R
    outer_resonance_targets::Vector{Int}
    recovery_attempted::Bool
    detail::String
    cause::C
end

function Base.showerror(io::IO, failure::_BorderedSolveFailure)
    heading = failure.category === :outer_resonance ?
              "Bordered cohomological system failed at a configured outer resonance." :
              failure.category === :factorisation ?
              "Bordered cohomological factorisation failed." :
              failure.category === :solve ? "Bordered cohomological solve failed." :
              "Bordered cohomological solution failed its accuracy check."
    println(io, heading)
    println(io, "  selected backend: ", failure.backend)
    println(io, "  multiindex position: ", failure.index)
    println(io, "  multiindex: ", failure.multiindex)
    println(io, "  superharmonic: ", failure.superharmonic)
    println(io, "  inner resonance mask: ", failure.inner_resonance_mask)
    println(io, "  outer resonance targets: ", failure.outer_resonance_targets)
    println(io, "  cached-factor recovery attempted: ", failure.recovery_attempted)
    println(io, "  detail: ", failure.detail)
    if failure.category === :outer_resonance
        println(io,
            "The configured resonance set flags an off-manifold target here, so a " *
            "mathematical outer resonance is the likely cause.")
    elseif failure.category === :factorisation
        println(io,
            "This is a backend-reported numerical factorisation failure; it is not by " *
            "itself proof that the bordered system is mathematically singular.")
    end
    failure.cause === nothing || begin
        print(io, "Caused by: ")
        showerror(io, failure.cause)
    end
end

"""Return whether a caught runtime failure must pass through without solver wrapping."""
function _is_unrecoverable_failure(error)
    error isa InterruptException ||
        error isa OutOfMemoryError ||
        error isa StackOverflowError
end

"""Return configured outer-resonance target indices for one monomial position."""
function _outer_resonance_targets(resonance_set, index::Int)
    outer = resonance_set.outer_resonances
    outer === nothing && return Int[]
    return findall(view(outer, :, index))
end

"""Construct and throw a contextual [`_BorderedSolveFailure`](@ref)."""
function _throw_bordered_failure(category::Symbol, backend::Symbol,
        index::Int, multiindex, superharmonic, inner_resonance_mask, resonance_set;
        recovery_attempted::Bool = false,
        detail::AbstractString,
        cause = nothing)
    outer_targets = _outer_resonance_targets(resonance_set, index)
    effective_category = !isempty(outer_targets) &&
                         category in (:factorisation, :solve, :accuracy) ?
                         :outer_resonance : category
    throw(_BorderedSolveFailure(
        effective_category, backend, index, multiindex, superharmonic,
        inner_resonance_mask, outer_targets, recovery_attempted, String(detail), cause))
end
