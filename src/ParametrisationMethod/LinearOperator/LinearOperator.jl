module LinearOperator

using LinearAlgebra

struct PrecomputedData{T, ROM}
	D::NTuple{ROM, Vector{Vector{T}}}  # D[r][k+1] = coefficient for s^k
	FOM::Int      # dimension of the full order model 
	ORD::Int      # native order of the full order model 
end

function precompute(B::Vector{Matrix{T}}, Y::Dict{R, Vector{T}}, λ::Dict{R, T}) where {T <: Number, R}
	ORD = length(B) - 1
	m, n = size(B[1])
	D = Dict{R, Vector{Vector{T}}}()
	for r in keys(Y)
		Dr = [zeros(T, m) for _ in 1:ORD]
		yr = Y[r]
		λr = λ[r]
		for L in 1:ORD                      # degree L (L = 1 … ORD)
			V = B[L+1] * yr                 # B_L * Y_r
			λpow = one(T)
			# accumulate into D_{k,r} for k = L-1 … 0
			for k in (L-1):-1:0
				Dr[k+1] .+= V .* λpow
				λpow *= λr
			end
		end
		D[r] = Dr
	end
	return PrecomputedData(D, m, n, ORD)
end

function build_matrix(s::T, B::Vector{Matrix{T}}, data::PrecomputedData{T, R}, resonance_set) where {T, R}
	ORD = data.ORD
	FOM = data.FOM
	# ---------- compute A(s) = Σ_{l=0}^{ORD} B_l s^l ----------
	A = B[end]                     # B_ORD
	for l in (ORD-1):-1:0
		A = A * s + B[l+1]         # Horner's method
	end
	# ---------- collect all columns ----------
	cols = [A]                     # first block: m×n matrix
	for r in resonance_set
		Dr = data.D[r]
		if ORD == 0
			C = zeros(T, FOM)        # no f terms when ORD=0
		else
			C = Dr[end]            # coefficient of s^{ORD-1}
			for k in (ORD-2):-1:0
				C = C * s + Dr[k+1]   # Horner for C_r(s)
			end
		end
		push!(cols, reshape(C, FOM, 1))   # add as a column
	end
	return hcat(cols...)
end

end   # module
