module ParametrisationModule

using ..Polynomials: DensePolynomial

export Parametrisation
    
"""
    Parametrisation{N,T} = DensePolynomial{NTuple{N,T}}

A dense polynomial whose coefficients are `NTuple{N,T}`.

# Type parameters
- `N`: dimension of the tuple (typically `N` matches the order of the ODE).
- `T`: type of each coefficient component (typically an `AbstractVector`, e.g., `Vector{ComplexF64}`)

# Examples
```julia
coeffs = [([1.0, 2.0], [3.0, 4.0]), # polynomial coefficients of the first monomial
          ([0.0, 0.0], [1.0, 1.0])] # polynomial coefficients of the second monomial
W = Parametrisation{2,Vector{Float64}}(coeffs, multiindex_set)
````
"""
const Parametrisation{N,T} = DensePolynomial{NTuple{N,T}}

end # module