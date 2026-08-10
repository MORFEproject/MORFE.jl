using MORFE, LinearAlgebra, SparseArrays, StaticArrays, Printf
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver, select_master_modes_by_sorting
using MORFE.CohomologicalEquations: solve_cohomological_problem
using MORFE.Resonance: build_resonance_set
using MORFE.SpectralDecomposition: SpectralData

cub = MultilinearMap((res,x1,x2,x3)->(@. res += -1.0*x1*x2*x3), (3,0))
println("  n   order   |mset|    positional      spectral     delta")
for (n, order) in ((6,3), (6,7), (30,5), (30,7))
    B0 = Matrix(SymTridiagonal(fill(2.0,n), fill(-1.0,n-1)))
    B2 = Matrix(1.0I,n,n); B1 = 0.002*B2
    dense = NthOrderModel((B0,B1,B2), (cub,))
    model = NthOrderModel((sparse(B0),sparse(B1),sparse(B2)), (cub,))
    ep = spectrum(dense; solver=DefaultEigensolver())
    select_master_modes_by_sorting(ep, 2)
    m = ep.master_modes
    λ = SVector{2,ComplexF64}(ep.eigenvalues[m]); Ψ = ep.eigenmodes[:,1,m]
    ℓ = ep.left_eigenmodes[:,m]
    mmd = @view(ep.eigenmodes[:,2:end,m]); lmd = @view(ep.left_eigenmodes_orders[:,1:end-1,m])
    mset = all_multiindices_up_to(2, order; min_degree=1)
    rset = build_resonance_set(model, :complex_normal_form, mset, ep, 0.05, nothing)
    sd = SpectralData(model, ep; master=findall(m), conjugate_permutation=[2,1])
    pos() = solve_cohomological_problem(model, mset, λ, Ψ, ℓ, rset;
        master_modes_derivatives=mmd, left_modes_derivatives=lmd,
        conjugate_permutation=[2,1], show_progress=false)
    spc() = solve_cohomological_problem(model, mset, sd, rset; show_progress=false)
    pos(); spc()
    a = @allocated pos(); b = @allocated spc()
    @printf("%4d %5d %8d %14d %13d %+9d\n", n, order, length(mset), a, b, b-a)
end
