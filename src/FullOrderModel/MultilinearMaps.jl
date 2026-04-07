module MultilinearMaps

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

"""
struct MultilinearMap{ORD,F}
    f!::F
    multiindex::NTuple{ORD,Int}
    multiplicity_external::Int
    deg::Int
end

"""
	MultilinearMap(f, multiindex)

Create a multilinear term for a system of order ORD without external dynamics.

# Arguments
- `f!`: in-place evaluation function
- `multiindex`: tuple specifying which derivatives are used
"""
function MultilinearMap(f!, multiindex::NTuple{ORD,Int}) where {ORD}
    @assert all(multiindex .>= 0) "Terms in the multiindex cannot be negative, but multiindex=$multiindex"
    deg = sum(multiindex)
    # Check if input arguments of f matches deg
    ms = methods(f!)
    @assert length(ms) == 1 "Function $(f!) must have exactly one method to determine number of inputs"
    @assert ms[1].nargs == deg + 2 "Function $(f!) must accept $(deg+1) arguments (`res` and $deg inputs) instead of $(ms[1].nargs - 1)"
    @assert deg >= 2 "Function $(f!) must have degree at least 2, but has degree $deg"

    return MultilinearMap{ORD,typeof(f!)}(f!, multiindex, 0, deg)
end

# Create a multilinear term for a first order system.
function MultilinearMap(f!)
    ms = methods(f!)
    @assert length(ms) == 1 "Function $(f!) must have exactly one method to determine number of inputs"
    deg = ms[1].nargs - 2 # subtract the function itself and `res`
    @assert deg >= 2 "Function $(f!) must have degree at least 2, but has degree $deg"
    multiindex = (UInt8(deg),)
    return MultilinearMap{1,typeof(f!)}(f!, multiindex, 0, deg)
end

function MultilinearMap(
    f!, multiindex::NTuple{ORD,Int}, multiplicity_external::Int) where {ORD}
    @assert all(multiindex .>= 0) "Terms in the multiindex cannot be negative, but multiindex=$multiindex"
    @assert multiplicity_external >= 0 "The argument multiplicity_external cannot be negative, but multiplicity_external=$multiplicity_external"
    deg = sum(multiindex) + multiplicity_external
    # Check if input arguments of f matches deg
    ms = methods(f!)
    @assert length(ms) == 1 "Function $(f!) must have exactly one method to determine number of inputs"
    @assert ms[1].nargs == deg + 2 "Function $(f!) must accept $(deg+1) arguments (`res` and $deg inputs) instead of $(ms[1].nargs - 1)"
    @assert (deg >= 2) || (multiplicity_external >= 1)
    "Function $(f!) does not depend the external state, hence it must have degree at least 2, but it has degree $deg"

    return MultilinearMap{ORD,typeof(f!)}(f!, multiindex, multiplicity_external, deg)
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

end # module