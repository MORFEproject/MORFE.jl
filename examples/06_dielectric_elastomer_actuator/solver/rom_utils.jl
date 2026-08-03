"""
rom_utils.jl — ROM evaluation and small fixed-step integrators (Phase 8).

Hand-rolled polynomial evaluation and integrators (no external ODE dependency):
- `eval_poly`    : evaluate a DensePolynomial coefficient matrix at a point z
- `rk4`          : classic RK4 for small non-stiff systems (the reduced dynamics)
- `imex_cn_ab2`  : Crank–Nicolson (linear part, prefactored) + Adams–Bashforth-2
				   (nonlinear part) for the stiff full-order validation systems
"""

using LinearAlgebra

"""
	eval_poly(coeffs, mset, z) -> Vector{ComplexF64}

`coeffs :: (ncomp, L)` aligned to `mset.exponents`; returns Σ_l coeffs[:,l]·z^α_l.
"""
function eval_poly(coeffs::AbstractMatrix, mset, z)
    acc = zeros(ComplexF64, size(coeffs, 1))
    for (l, α) in enumerate(mset.exponents)
        mon = one(ComplexF64)
        @inbounds for j in eachindex(z)
            αj = α[j]
            αj == 0 && continue
            mon *= z[j]^αj
        end
        acc .+= view(coeffs, :, l) .* mon
    end
    return acc
end

"""
	rk4(f, z0, t0, t1, dt) -> (ts, Z)

Fixed-step RK4; `f(z, t)` returns ż. `Z[k]` is the state at `ts[k]`.
"""
function rk4(f, z0, t0, t1, dt)
    nsteps = ceil(Int, (t1 - t0) / dt)
    ts = collect(range(t0, t0 + nsteps * dt; length = nsteps + 1))
    Z = Vector{typeof(z0)}(undef, nsteps + 1)
    Z[1] = copy(z0)
    z = copy(z0)
    for k in 1:nsteps
        t = ts[k]
        k1 = f(z, t)
        k2 = f(z .+ (dt / 2) .* k1, t + dt / 2)
        k3 = f(z .+ (dt / 2) .* k2, t + dt / 2)
        k4 = f(z .+ dt .* k3, t + dt)
        z = z .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        Z[k + 1] = copy(z)
    end
    return ts, Z
end

"""
	imex_cn_ab2(Lmat, N, s0, t0, t1, dt) -> (ts, S)

Solve ṡ = Lmat·s + N(s, t) with Crank–Nicolson on the (stiff, constant) linear part and
Adams–Bashforth-2 on the (mild) nonlinear part. `Lmat` is prefactored once.
"""
function imex_cn_ab2(Lmat::AbstractMatrix, N, s0, t0, t1, dt)
    nsteps = ceil(Int, (t1 - t0) / dt)
    ts = collect(range(t0, t0 + nsteps * dt; length = nsteps + 1))
    n = length(s0)
    Aplus = I + (dt / 2) .* Lmat
    F = lu(I - (dt / 2) .* Lmat)
    S = Vector{Vector{Float64}}(undef, nsteps + 1)
    S[1] = copy(s0)
    s = copy(s0)
    Nprev = N(s, t0)
    # first step: explicit Euler on N (AB2 needs history)
    s = F \ (Aplus * s .+ dt .* Nprev)
    S[2] = copy(s)
    for k in 2:nsteps
        t = ts[k]
        Ncur = N(s, t)
        s = F \ (Aplus * s .+ dt .* (1.5 .* Ncur .- 0.5 .* Nprev))
        Nprev = Ncur
        S[k + 1] = copy(s)
    end
    return ts, S
end

"""
	integrate_rom(R, mset, z0, t1, dt) -> (ts, Z)

Integrate the reduced dynamics ż = R(z) (complex state, includes external components,
whose exact linear dynamics ±iΩ are part of R).
"""
function integrate_rom(R, mset, z0::Vector{ComplexF64}, t1, dt)
    coeffs = R.poly.coefficients
    f(z, t) = eval_poly(coeffs, mset, z)
    return rk4(f, z0, 0.0, t1, dt)
end

"""
	reconstruct_tip(W, mset, Z, idx_wtip) -> Vector{Float64}

Map reduced trajectories through the displacement block of the parametrisation W
(coefficients (FOM, ORD, L); block 1 = displacement) and extract the tip DOF.
"""
function reconstruct_tip(W, mset, Z, idx_wtip)
    Wdisp = W.poly.coefficients[:, 1, :]      # FOM × L
    wrow = Wdisp[idx_wtip, :]                 # 1 × L functional — evaluate scalar directly
    out = Vector{Float64}(undef, length(Z))
    for (k, z) in enumerate(Z)
        acc = zero(ComplexF64)
        for (l, α) in enumerate(mset.exponents)
            mon = one(ComplexF64)
            @inbounds for j in eachindex(z)
                αj = α[j]
                αj == 0 && continue
                mon *= z[j]^αj
            end
            acc += wrow[l] * mon
        end
        out[k] = real(acc)
    end
    return out
end

"""
	steady_amplitude(ts, w, T_period; n_periods = 5) -> Float64

(max − min)/2 of the signal over the last `n_periods` forcing periods.
"""
function steady_amplitude(ts, w, T_period; n_periods = 5)
    t_cut = ts[end] - n_periods * T_period
    idx = findall(>=(t_cut), ts)
    seg = w[idx]
    return (maximum(seg) - minimum(seg)) / 2
end
