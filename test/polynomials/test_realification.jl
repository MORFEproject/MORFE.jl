using Test
using Random
using StaticArrays: SVector

include(joinpath(@__DIR__, "../../src/MORFE.jl"))
using .MORFE.Multiindices
using .MORFE.Polynomials
using .MORFE.Realification
using .MORFE.Realification: _multinomial, _compositions, _reorder_canonical, _realify_term

# ---------- Helper functions for testing ----------

"""
	polynomial_to_dict(poly::DensePolynomial) -> Dict{Vector{Int}, coeff}

Convert a polynomial to a dictionary of non‑zero coefficients.
Uses `each_term` so it works for both scalar (N=1) and vector-valued (N=2)
polynomials without relying on `eltype` or raw array indexing.

- Scalar polynomial: values are `T`.
- Vector polynomial: values are `Vector{T}` (concrete copies of the column views).
"""
function polynomial_to_dict(poly::DensePolynomial{T, NVAR, 1}) where {T, NVAR}
	d = Dict{Vector{Int}, T}()
	for (exp, coeff) in each_term(poly)
		d[collect(exp)] = coeff
	end
	return d
end

function polynomial_to_dict(poly::DensePolynomial{T, NVAR, N}) where {T, NVAR, N}
	d = Dict{Vector{Int}, Vector{T}}()
	for (exp, coeff) in each_term(poly)
		d[collect(exp)] = collect(coeff)   # materialise the SubArray view
	end
	return d
end

# ---------- Tests for internal helpers (type‑agnostic) ----------

@testset "multinomial" begin
	@test _multinomial(0, [0]) == 1
	@test _multinomial(5, [5]) == 1
	@test _multinomial(5, [0, 5]) == 1
	@test _multinomial(5, [1, 4]) == 5
	@test _multinomial(5, [2, 3]) == 10
	@test _multinomial(5, [1, 1, 3]) == 20
	@test _multinomial(6, [2, 2, 2]) == 90
end

@testset "compositions" begin
	comps = collect(_compositions(2, 2))
	@test length(comps) == 3
	@test Set(comps) == Set([[0, 2], [1, 1], [2, 0]])

	comps3 = collect(_compositions(3, 3))
	@test length(comps3) == 10
	for k in comps3
		@test length(k) == 3
		@test sum(k) == 3
		@test all(>=(0), k)
	end
end

# ---------- Tests for compose_linear on all implemented polynomial types ----------

@testset "compose_linear" begin
	@testset "for $PolyType" for PolyType in [DensePolynomial]
		@testset "linear identity" begin
			f = Dict([1, 0] => 1.0, [0, 1] => 2.0)
			M = [1 0; 0 1]
			p = 2
			poly = DensePolynomial(f)
			result = compose_linear(poly, M)
			@test result isa PolyType
			result_dict = polynomial_to_dict(result)
			expected = Dict([1, 0] => 1.0, [0, 1] => 2.0)
			@test result_dict == expected
		end

		@testset "quadratic example" begin
			f = Dict([2, 0] => 3.0, [1, 1] => 5.0)
			M = [1 2; 3 4]
			p = 2
			poly = DensePolynomial(f)
			result = compose_linear(poly, M)
			result_dict = polynomial_to_dict(result)
			expected = Dict([2, 0] => 18.0, [1, 1] => 62.0, [0, 2] => 52.0)
			@test result_dict == expected
		end

		@testset "constant" begin
			f = Dict([0, 0] => 7.0)
			M = [1 2; 3 4]
			p = 2
			poly = DensePolynomial(f)
			result = compose_linear(poly, M)
			result_dict = polynomial_to_dict(result)
			expected = Dict([0, 0] => 7.0)
			@test result_dict == expected
		end

		@testset "higher degree" begin
			f = Dict([2, 1, 0] => 1.0, [0, 0, 3] => 1.0)
			M = [1 0; 0 1; 1 1]
			p = 2
			poly = DensePolynomial(f)
			result = compose_linear(poly, M)
			result_dict = polynomial_to_dict(result)
			expected = Dict([3, 0] => 1.0, [2, 1] => 4.0, [1, 2] => 3.0, [0, 3] => 1.0)
			@test result_dict == expected
		end

		@testset "vector output" begin
			# Use SVector instead of plain Vector for coefficients
			f = Dict([2, 0] => SVector{2, Float64}(1.0, 0.0),
				[1, 1] => SVector{2, Float64}(0.0, 1.0))

			M = [1 2; 3 4]
			p = 2
			# x = z + 2w
			# y = 3z + 4w
			# f(z,w) = (1,0) (z^2 + 4zw + 4w^2) + (0,1) (3z^2 + 10zw + 8w^2) 
			#        = (1,3) z^2 + (4,10) zw + (4,8) w^2
			poly = DensePolynomial(f)
			result = compose_linear(poly, M)
			result_dict = polynomial_to_dict(result)
			@test result_dict[[2, 0]] ≈ SVector{2, Float64}(1.0, 3.0)
			@test result_dict[[1, 1]] ≈ SVector{2, Float64}(4.0, 10.0)
			@test result_dict[[0, 2]] ≈ SVector{2, Float64}(4.0, 8.0)
		end

		@testset "zero matrix" begin
			f = Dict([1, 0] => 2.0, [0, 1] => 3.0, [0, 0] => 5.0)
			M = zeros(2, 2)
			p = 2
			poly = DensePolynomial(f)
			result = compose_linear(poly, M)
			result_dict = polynomial_to_dict(result)
			@test result_dict[[0, 0]] ≈ 5.0
			for (k, v) in result_dict
				if k != [0, 0]
					@test v ≈ 0.0
				end
			end
		end

		@testset "identity with extra variables" begin
			f = Dict([2, 0] => 1.0)
			M = [1 0 0; 0 1 0]
			p = 3
			poly = DensePolynomial(f)
			result = compose_linear(poly, M)
			result_dict = polynomial_to_dict(result)
			@test result_dict[[2, 0, 0]] ≈ 1.0
			for (k, v) in result_dict
				if k != [2, 0, 0]
					@test v ≈ 0.0
				end
			end
		end

		@testset "binomial expansion" begin
			f = Dict([5] => 1.0)
			poly = DensePolynomial(f)
			M = [1 1]
			p = 2
			result = compose_linear(poly, M)
			result_dict = polynomial_to_dict(result)
			for k in 0:5
				@test result_dict[[k, 5-k]] ≈ binomial(5, k)
			end
		end
	end
end

# ---------- Tests for realify on all implemented polynomial types ----------

@testset "realify" begin
	@testset "for $PolyType" for PolyType in [DensePolynomial]
		@testset "reorder_canonical - empty" begin
			mset = MultiindexSet(Vector{SVector{3, Int}}())
			poly = zero(DensePolynomial{Float64}, mset)
			conj_map = [2, 1, 3]
			canonical, n, m = _reorder_canonical(poly, conj_map)
			@test nvars(canonical) == 3
			@test n == 1
			@test m == 1
		end

		@testset "reorder_canonical - single pair" begin
			f = Dict([1, 1] => 1.0)
			poly = DensePolynomial(f)
			conj_map = [2, 1]
			newpoly, n, m = _reorder_canonical(poly, conj_map)
			@test length(newpoly) == 1
			@test n == 1
			@test m == 0
			d = polynomial_to_dict(newpoly)
			@test haskey(d, [1, 1])
		end

		@testset "reorder_canonical - two pairs" begin
			f = Dict([2, 1, 1, 0, 0] => 2.0)
			poly = DensePolynomial(f)
			conj_map = [3, 2, 1, 5, 4]
			newpoly, n, m = _reorder_canonical(poly, conj_map)
			@test n == 2
			@test m == 1
			d = polynomial_to_dict(newpoly)
			@test haskey(d, [2, 0, 1, 0, 1])
			@test d[[2, 0, 1, 0, 1]] ≈ 2.0
		end

		@testset "realify_term - z*zbar" begin
			exp_vec = SVector(1, 1)
			coeff = 2.0 + 3.0im
			n = 1
			result = _realify_term(exp_vec, coeff, n)
			@test result[[2, 0]] ≈ coeff
			@test result[[0, 2]] ≈ coeff
		end

		@testset "realify_term - z^2" begin
			exp_vec = SVector(2, 0)
			coeff = 1.0 + 0.0im
			n = 1
			result = _realify_term(exp_vec, coeff, n)
			@test result[[2, 0]] ≈ 1.0 + 0.0im
			@test result[[0, 2]] ≈ -1.0 + 0.0im
			@test result[[1, 1]] ≈ 0.0 + 2.0im
		end

		@testset "realify_term - with real variable" begin
			exp_vec = SVector(1, 1, 2)
			coeff = 2.0 + 0.0im
			n = 1
			result = _realify_term(exp_vec, coeff, n)
			@test result[[2, 0, 2]] ≈ 2.0 + 0.0im
			@test result[[0, 2, 2]] ≈ 2.0 + 0.0im
		end

		@testset "realify - simple example from docs" begin
			f = Dict([1, 1, 0] => 2.0+1.0im, [2, 0, 0] => 1.0-1.0im, [0, 0, 0] => 3.0+0.0im)
			poly = DensePolynomial(f)
			conj_map = [2, 1, 3]
			result = realify(poly, conj_map)
			@test result isa PolyType
			result_dict = polynomial_to_dict(result)
			@test result_dict[[2, 0, 0]] ≈ (2.0+1.0im) + (1.0-1.0im)
			@test result_dict[[1, 1, 0]] ≈ (1.0-1.0im)*2im
			@test result_dict[[0, 2, 0]] ≈ (2.0+1.0im) + (-1.0+1.0im)
			@test result_dict[[0, 0, 0]] ≈ 3.0 + 0.0im
		end

		@testset "realify - evaluation consistency" begin
			Random.seed!(42)
			for _ in 1:5
				n = 2
				m = 1
				N = 2n + m
				nterms = 5
				maxdeg = 7

				poly_orig = Dict{Vector{Int}, ComplexF64}()
				for _ in 1:nterms
					exp = rand(0:maxdeg, N)
					coeff = randn() + randn() * im
					poly_orig[exp] = coeff
				end

				conj_map = [2, 1, 4, 3, 5]

				poly = DensePolynomial(poly_orig)
				poly_real = realify(poly, conj_map)

				x1 = randn()
				y1 = randn()
				x3 = randn()
				y3 = randn()
				w = randn()

				z1 = x1 + im*y1
				z2 = conj.(z1)
				z3 = x3 + im*y3
				z4 = conj.(z3)

				val_orig = evaluate(poly, [z1, z2, z3, z4, w])
				val_real = evaluate(poly_real, [x1, x3, y1, y3, w])

				@test val_orig ≈ val_real atol=1e-10
			end
		end

		@testset "realify - idempotence (real variables only)" begin
			f = Dict([2, 0] => 1.0, [1, 1] => 2.0, [0, 2] => 3.0)
			poly = DensePolynomial(f)
			conj_map = [1, 2]
			result = realify(poly, conj_map)
			result_dict = polynomial_to_dict(result)
			@test result_dict[[2, 0]] ≈ 1.0
			@test result_dict[[1, 1]] ≈ 2.0
			@test result_dict[[0, 2]] ≈ 3.0
		end
	end
end

# ---------- Tests for realify_via_linear on all implemented polynomial types ----------

@testset "realify_via_linear" begin
	@testset "for $PolyType" for PolyType in [DensePolynomial]
		@testset "equivalence with realify" begin
			Random.seed!(123)
			for _ in 1:5
				n = 2
				m = 1
				N = 2n + m
				nterms = 5
				maxdeg = 3

				f = Dict{Vector{Int}, ComplexF64}()
				for _ in 1:nterms
					exp = rand(0:maxdeg, N)
					coeff = randn() + randn() * im
					f[exp] = coeff
				end

				conj_map = [2, 1, 4, 3, 5]
				poly = DensePolynomial(f)

				res_direct = realify(poly, conj_map)
				res_linear = realify_via_linear(poly, conj_map)

				@test res_direct isa PolyType
				@test res_linear isa PolyType

				x = randn(n)
				y = randn(n)
				w = randn(m)
				vals = vcat(x, y, w)

				val_direct = evaluate(res_direct, vals)
				val_linear = evaluate(res_linear, vals)

				@test val_direct ≈ val_linear atol=1e-10
			end
		end
	end
end

# ---------- Additional tests for DensePolynomial ----------

@testset "DensePolynomial specific" begin
	@testset "basis sharing" begin
		f1 = Dict([1, 0] => 1.0, [0, 1] => 2.0)
		f2 = Dict([1, 0] => 3.0, [0, 1] => 4.0)
		p1 = DensePolynomial(f1)
		p2 = DensePolynomial(f2)
		@test multiindex_set(p1).exponents == multiindex_set(p2).exponents
	end

	@testset "empty polynomial" begin
		mset = MultiindexSet(Vector{SVector{3, Int}}())
		poly = DensePolynomial(Float64[], mset)
		@test length(poly) == 0
		@test nvars(poly) == 3
		@test polynomial_to_dict(poly) == Dict{Vector{Int}, Float64}()
	end
end

println("All tests passed!")
