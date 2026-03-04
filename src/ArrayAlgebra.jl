module ArrayAlgebra

# Zero for tuple types and values
Base.zero(::Type{NTuple{N,T}}) where {N,T} = ntuple(_ -> zero(T), N)
Base.zero(t::NTuple{N,T}) where {N,T} = ntuple(_ -> zero(T), N)

# Check if tuple is all zeros
Base.iszero(t::NTuple{N,T}) where {N,T} = all(iszero, t)

# Scalar multiplication (both orders)
Base.:*(s::Number, t::NTuple{N,T}) where {N,T} = ntuple(i -> s * t[i], N)
Base.:*(t::NTuple{N,T}, s::Number) where {N,T} = s * t

# Addition of two tuples (component‑wise)
Base.:+(t1::NTuple{N,T}, t2::NTuple{N,T}) where {N,T} = ntuple(i -> t1[i] + t2[i], N)

# Subtraction of two tuples (component‑wise)
Base.:-(t1::NTuple{N,T}, t2::NTuple{N,T}) where {N,T} = ntuple(i -> t1[i] - t2[i], N)

# Absolute value
Base.abs(t::NTuple{N,T}) where {N,T} = ntuple(i -> abs(t[i]), N)

end