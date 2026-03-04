include(joinpath(@__DIR__, "../src/Realification.jl"))
using .Realification

# 1. Define multiindex set by number of variables and max degree
nvars = 3
max_degree = 2
multiindex_set = all_multiindices_up_to(nvars, max_degree)   # default Grlex order
println("Multiindex set (columns are exponents):")
for (idx, exp) in enumerate(eachcol(multiindex_set.exponents))
    println("  idx = $idx :\texponent = $exp")
end



# 2. Create zero polynomial
dimension = 2
coeff_type = NTuple{dimension, ComplexF64}
poly = zero(DensePolynomial{coeff_type}, multiindex_set)
println("\nZero polynomial created")
for (idx, exp) in enumerate(eachcol(poly.multiindex_set.exponents))
    println("  idx = $idx :\texponent = $exp, \tcoefficient = ", poly.coeffs[idx])
end



# 3. Change some coefficients *in place*
i = find_term(poly, [1,1,0])         # exponent must be in the set
new_coeff_i = (1.0, 0.0 + 2.0im)
poly.coeffs[i] = new_coeff_i

i = find_term(poly, [0,0,2])
new_coeff_i = (3.0 + 4.0im, 5.0)
poly.coeffs[i] = new_coeff_i

println("\nPolynomial after in‑place modification (non‑zero terms only):")
for (idx, exp) in enumerate(eachcol(poly.multiindex_set.exponents))
    println("  idx = $idx :\texponent = $exp, \tcoefficient = ", poly.coeffs[idx])
end



# 4. Evaluate at a point (z1, z2, z3)
z1 = 3.0 + 4.0im
z2 = conj(z1)
z3 = 5.0 # real variable
vals = [z1, z2, real(z3)]

# Full vector result (both components)
full = evaluate(poly, vals)
println("\nEvaluation at (z1=$z1, z2=$z2, z3=$z3):")
println("  full evaluation = ", full)

# Evaluate directly one component
comp = 1 
println("  directly evalute component $comp = ", evaluate(poly, vals, comp))



# 5. Build a new polynomial only with one component
poly_scalar = extract_component(poly, 1)
println("\nScalar polynomial corresponding to component $comp:")
for (idx, exp) in enumerate(eachcol(poly_scalar.multiindex_set.exponents))
    println("  idx = $idx :\texponent = $exp, \tcoefficient = ", poly_scalar.coeffs[idx])
end

# Full evaluation of the smaller polynomial -> should match the previous result
println("\nEvaluation at (z1=$z1, z2=$z2, z3=$z3):")
println("  full evaluation = ", evaluate(poly_scalar, vals))



# 6. Realify

conjugate_map = [2,1,3] 
# conj(z1) = z2
# conj(z2) = z1     (z1 and z2 form a conjugate pair)
# conj(z3) = z3     (z3 is real)

poly_realified = realify(poly, conjugate_map)
println("\nRealified polynomial:")
for (idx, exp) in enumerate(eachcol(poly_realified.multiindex_set.exponents))
    println("  idx = $idx :\texponent = $exp, \tcoefficient = ", poly_realified.coeffs[idx])
end

# Full vector result (both components)
x = real(z1)
y = imag(z1)
z3 = real(z3)
full_realified = evaluate(poly_realified, [x,y,z3])
println("\nEvaluation at (x=real(z1)=$x, y=imag(z1)=$y, z3=real(z3)=$z3):")
println("  full evaluation = ", full_realified)

# Compare with the result pre realification
println("  error = ",  abs(full_realified - full))