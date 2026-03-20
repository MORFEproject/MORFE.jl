include(joinpath(@__DIR__, "../src/MORFE.jl"))
using .MORFE.Multiindices: all_multiindices_up_to
using .MORFE.Polynomials #: DensePolynomial, zero, find_term, evaluate
using .MORFE.Realification

# 1. Create a dense polynomial with 3 variables, max degree 2, and 2‑component coefficients
nvars, max_degree = 3, 2
multiindex_set = all_multiindices_up_to(nvars, max_degree)   # all exponents up to degree 2
poly = zero(DensePolynomial{NTuple{2,ComplexF64}}, multiindex_set)
println("Zero polynomial initialised")

# 2. Set two non‑zero coefficients (in place)
poly.coeffs[find_term(poly, [1,1,0])] = (1.0, 2.0im)
poly.coeffs[find_term(poly, [0,0,2])] = (3.0+4.0im, 5.0)
println("\nPolynomial after in‑place modification:")
for (idx, exp) in enumerate(eachcol(poly.multiindex_set.exponents))
    println("  idx = $idx :\texponent = $exp, \tcoefficient = ", poly.coeffs[idx])
end

# 3. Evaluate at a point with conjugate pair (z1, z2) and real variable z3
z1 = 3.0+4.0im
z2 = conj(z1)
z3 = 5.0
vals = [z1, z2, real(z3)]
full_eval = evaluate(poly, vals)
println("\nEvaluation at (z1=$z1, z2=$z2, z3=$z3):")
println("  full result (both components) = ", full_eval)
println("  only component 1 = ", evaluate(poly, vals, 1)) # Directly evaluate one component

# 4. Build a new polynomial only with one component of the original
poly_scalar = extract_component(poly, 2)
println("  only component 2 = ", evaluate(poly_scalar, vals))

# 5. Realify using conjugation map: z1 ↔ z2, z3 is real
conj_map = [2, 1, 3]
realf_poly = realify(poly, conj_map)   # now in variables (x, y, w)

# 6. Evaluate the realified polynomial at (x,y,w) = (real(z1), imag(z1), z3)
x, y, w = real(z1), imag(z1), z3
realf_eval = evaluate(realf_poly, [x, y, w])
println("\n Realified evaluation at (x=$x, y=$y, w=$w):")
println("  result = ", realf_eval)
println("  error compared to original = ", abs(realf_eval - full_eval))

# 7. Compose the polynomial with a linear transformation f(Mx)
M = [1.0    1.0im   0.0;
     1.0   -1.0im   0.0;
     0.0    0.0     1.0]
comp_poly = compose_linear(poly, M)
println("\n Evaluate f(Mx) at (x=$x, y=$y, w=$w):")
println("  result = ", evaluate(realf_poly, [x, y, w]))