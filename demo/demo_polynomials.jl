using StaticArrays: SVector

include(joinpath(@__DIR__, "../src/MORFE.jl"))
using .MORFE.Multiindices: all_multiindices_up_to
using .MORFE.Polynomials: DensePolynomial, zero, find_term, evaluate, extract_component, linear_matrix_of_polynomial, nmonomials, coeff_shape
using .MORFE.Realification

# 1. Create a dense polynomial with 3 variables, max degree 2, and 2‑component coefficients
nvars, max_degree = 3, 2
# all monomials in 3 variables up to degree 2
multiindex_set = all_multiindices_up_to(nvars, max_degree)

# zero(DensePolynomial{T}, coeff_shape, mset) — new API for vector-valued polynomials
poly = zero(DensePolynomial{ComplexF64}, (2,), multiindex_set)
println("Zero polynomial initialised: coeff_shape=$(coeff_shape(poly)), nmonomials=$(nmonomials(poly))")

# 2. Set two non‑zero coefficients in place.
# The coefficient matrix has shape (2, L); columns are indexed by monomial position.
poly.coefficients[:, find_term(poly, [1, 1, 0])] = [1.0, 2.0im]
poly.coefficients[:, find_term(poly, [0, 0, 2])] = [3.0+4.0im, 5.0]
println("\nPolynomial after in‑place modification:")
for (idx, exp) in enumerate(poly.multiindex_set.exponents)
	println("Index = $idx:\texponent = $exp\tcoefficient = ", poly.coefficients[:, idx])
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
x, y, w = real(z1), imag(z1), real(z3)
realf_eval = evaluate(realf_poly, [x, y, w])
println("\n Realified evaluation at (x=$x, y=$y, w=$w):")
println("  result = ", realf_eval)
using LinearAlgebra
println("  relative error = ", norm(realf_eval - full_eval)/norm(full_eval))

# 7. Compose the polynomial with a linear transformation f(Mx)
M = [ 1.0    1.0im   0.0;
	1.0   -1.0im   0.0;
	0.0    0.0     1.0]
comp_poly = compose_linear(poly, M)
println("\n Evaluate f(Mx) at (x=$x, y=$y, w=$w):")
comp_eval = evaluate(comp_poly, [x, y, w])
println("  result = ", comp_eval)
println("  relative error = ", norm(comp_eval - full_eval)/norm(full_eval))

# -------------------------------------------------------------------
# 8. Extract the linear part of a random polynomial
# -------------------------------------------------------------------

println("\n\n=== Random Polynomial ===\n")
output_size, input_size = 7, 3
max_degree = 2
multiindex_set = all_multiindices_up_to(input_size, max_degree)
deleteat!(multiindex_set.exponents, 1) # delete the constant term
idx = rand(1:input_size)
println("Deleting exponent $(multiindex_set.exponents[idx])\n")
deleteat!(multiindex_set.exponents, idx) # delete a random linear exponent

# Vector{SVector{K,T}} constructor converts to a contiguous (K × L) matrix automatically
rand_poly = DensePolynomial([randn(SVector{output_size, Float64}) for _ in 1:length(multiindex_set)], multiindex_set)
for (i, exp) in enumerate(rand_poly.multiindex_set.exponents)
	println("Index = $i:\texponent = $exp\tcoefficient=$(rand_poly.coefficients[:, i])")
end
println("\nLinear part =\n", repr("text/plain", linear_matrix_of_polynomial(rand_poly)))

println("\nDisplay coefficients array (shape $(size(rand_poly.coefficients))):")
display(rand_poly.coefficients)

# -------------------------------------------------------------------
println("\n" * "="^80 * "\n")
println("Demo finished successfully.")
