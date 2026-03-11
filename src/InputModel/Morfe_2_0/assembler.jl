
#=
struct SparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    colptr::Vector{Ti}      # Column j is in colptr[j]:(colptr[j+1]-1)
    rowval::Vector{Ti}      # Row indices of stored values
    nzval::Vector{Tv}       # Stored values, typically nonzeros
end
=#

"""
> assembly_H!(Cp, entry,Ψ₁, Ψ₂, Ψ₃,mesh, U, mult = 1.0)
It assemblies cubic nonlinearities operator

\$ G(Ψ₁,Ψ₂,Ψ₃) = \\frac{1}{6} ∫_{Ω} γ(Ψ₁,Ψ₂):\\mathcal{A}:γ(Ψ₃,w) + γ(Ψ₁,Ψ₃):\\mathcal{A}:γ(Ψ₂,w) + γ(Ψ₃,Ψ₂):\\mathcal{A}:γ(Ψ₁,w) dΩ \$

- Cp : parametrisation data structure
- entry : entrance of the reference array
- Ψ₁ : mapping
- Ψ₂ : mapping
- Ψ₃ : mapping
- mesh : mesh data structure
- U : displacement field
- mult : integral multiplier
"""
function assembly_H!(res,Ψ₁,Ψ₂,Ψ₃,mesh::Grid,U::Field,mult = 1.0)
  #
  neq=U.neq
  X = zeros(Float64,nne_max*dim)
  dofs = zeros(Int64,nne_max*dim)
  Fₑ = zeros(ComplexF64,nne_max*dim)
  #
  Ψ₁ₑ = zeros(ComplexF64,nne_max*dim)
  Ψ₂ₑ = zeros(ComplexF64,nne_max*dim)  
  Ψ₃ₑ = zeros(ComplexF64,nne_max*dim) 
  #
  N     = zeros(Float64,nne_max)
  ∂N∂a  = zeros(Float64,(nne_max,dim))
  ∂N∂x  = zeros(Float64,(nne_max,dim))
  Jac   = zeros(Float64,(dim,dim))
  Jac⁻¹ = zeros(Float64,(dim,dim))
  #
  ∇U₁ = zeros(ComplexF64,(dim,dim))
  ∇U₂ = zeros(ComplexF64,(dim,dim))
  ∇U₃ = zeros(ComplexF64,(dim,dim))
  #
  e₁₂ = zeros(ComplexF64,(dim,dim)) 
  e₂₃ = zeros(ComplexF64,(dim,dim)) 
  e₁₃ = zeros(ComplexF64,(dim,dim)) 
  #
  eᵛ₁₂ = zeros(ComplexF64,dim*(dim-1)) 
  eᵛ₂₃ = zeros(ComplexF64,dim*(dim-1)) 
  eᵛ₁₃ = zeros(ComplexF64,dim*(dim-1))
  #
  σᵛ₁₂ = zeros(ComplexF64,dim*(dim-1))
  σᵛ₂₃ = zeros(ComplexF64,dim*(dim-1))
  σᵛ₁₃ = zeros(ComplexF64,dim*(dim-1))
  #
  sym∇ⁿˡ₁ = zeros(ComplexF64,(nne_max*dim,dim*(dim-1)))
  sym∇ⁿˡ₂ = zeros(ComplexF64,(nne_max*dim,dim*(dim-1)))
  sym∇ⁿˡ₃ = zeros(ComplexF64,(nne_max*dim,dim*(dim-1)))
  #
  for iΩ ∈ mesh.Ω
    Dᵢⱼₖₗ = iΩ.mat.Dᵢⱼₖₗ
    for set = 1:iΩ.Sen
      etype = iΩ.Set[set]
      qr = select_quadrature_rule(etype)
      nn = iΩ.Senn[set]
      skip = iΩ.eskip[set]
      for e = 1:iΩ.ne[set]
        conn = @view iΩ.e2n[skip+1+(e-1)*nn:skip+e*nn]
        get_coor!(mesh, conn, X, nn)
        dofs!(U,nn,conn,dofs)
        #
        @inbounds for i = 1:nn*dim
          if (dofs[i]>0)
            Ψ₁ₑ[i] = Ψ₁[dofs[i]]
            Ψ₂ₑ[i] = Ψ₂[dofs[i]]
            Ψ₃ₑ[i] = Ψ₃[dofs[i]]
          else
            Ψ₁ₑ[i] = 0.0
            Ψ₂ₑ[i] = 0.0
            Ψ₃ₑ[i] = 0.0
          end
        end
        #
        integrate_H!(Fₑ,X,Dᵢⱼₖₗ,Ψ₁ₑ,Ψ₂ₑ,Ψ₃ₑ,
                    N,∂N∂a,∂N∂x,Jac,Jac⁻¹,
                    ∇U₁,∇U₂,∇U₃,
                    e₁₂,e₂₃,e₁₃,
                    eᵛ₁₂,eᵛ₂₃,eᵛ₁₃,
                    σᵛ₁₂,σᵛ₂₃,σᵛ₁₃,
                    sym∇ⁿˡ₁,sym∇ⁿˡ₂,sym∇ⁿˡ₃,
                    nn,etype,qr)
        #
        for i = 1:nn*dim
          if (dofs[i]>0)
            @inbounds res[neq+dofs[i]] -= Fₑ[i]*mult
          end
        end
        #
      end
    end
  end
  #
  return nothing
  #
end



"""
> assembly_G!(Cp, entry,Ψ₁, Ψ₂,mesh, U, mult = 1.0)
It assemblies quadratic nonlinearities operator

\$ G(Ψ₁,Ψ₂) = \\frac{1}{2} ∫_{Ω} γ(Ψ₁,Ψ₂):\\mathcal{A}:ε(w) + γ(Ψ₁,w):\\mathcal{A}:ε(Ψ₂) + γ(w,Ψ₂):\\mathcal{A}:ε(Ψ₁) dΩ \$

- Cp : parametrisation data structure
- entry : entrance of the reference array
- Ψ₁ : mapping
- Ψ₂ : mapping
- mesh : mesh data structure
- U : displacement field
- mult : integral multiplier
"""
function assembly_G!(res, Ψ₁, Ψ₂,mesh::Grid, U::Field, mult = 1.0)
  #
  neq=U.neq
  if neq==info.nk    # what is info.nk? =info.nK?
    println("Gleich")
  end
  X = zeros(Float64,nne_max*dim)
  dofs = zeros(Int64,nne_max*dim)
  Fₑ = zeros(ComplexF64,nne_max*dim)
  #
  Ψ₁ₑ = zeros(ComplexF64,nne_max*dim)
  Ψ₂ₑ = zeros(ComplexF64,nne_max*dim)
  #
  N     = zeros(Float64,nne_max)
  ∂N∂a  = zeros(Float64,(nne_max,dim))
  ∂N∂x  = zeros(Float64,(nne_max,dim))
  Jac   = zeros(Float64,(dim,dim))
  Jac⁻¹ = zeros(Float64,(dim,dim))
  #
  ∇U₁    = zeros(ComplexF64,(dim,dim))
  ∇U₂    = zeros(ComplexF64,(dim,dim))
  sym∇U₁ = zeros(ComplexF64,(dim,dim))
  sym∇U₂ = zeros(ComplexF64,(dim,dim))
  e₁₂    = zeros(ComplexF64,(dim,dim))
  #
  εᵛ₁ = zeros(ComplexF64,dim*(dim-1)) 
  εᵛ₂ = zeros(ComplexF64,dim*(dim-1)) 
  eᵛ₁₂ = zeros(ComplexF64,dim*(dim-1)) 
  #
  σᵛ₁ = zeros(ComplexF64,dim*(dim-1))
  σᵛ₂ = zeros(ComplexF64,dim*(dim-1))
  σᵛ₁₂ = zeros(ComplexF64,dim*(dim-1))
  #
  sym∇ = zeros(ComplexF64,(nne_max*dim,dim*(dim-1)))
  sym∇ⁿˡ₁ = zeros(ComplexF64,(nne_max*dim,dim*(dim-1)))
  sym∇ⁿˡ₂ = zeros(ComplexF64,(nne_max*dim,dim*(dim-1)))
  #
  for iΩ ∈ mesh.Ω
    Dᵢⱼₖₗ = iΩ.mat.Dᵢⱼₖₗ
    for set = 1:iΩ.Sen
      etype = iΩ.Set[set]
      qr = select_quadrature_rule(etype)
      nn = iΩ.Senn[set]
      skip = iΩ.eskip[set]
      for e = 1:iΩ.ne[set]
        conn = @view iΩ.e2n[skip+1+(e-1)*nn:skip+e*nn]
        get_coor!(mesh, conn, X, nn)
        dofs!(U,nn,conn,dofs)
        #
        @inbounds for i = 1:nn*dim
          if (dofs[i]>0)
            Ψ₁ₑ[i] = Ψ₁[dofs[i]]
            Ψ₂ₑ[i] = Ψ₂[dofs[i]]
          else
            Ψ₁ₑ[i] = 0.0
            Ψ₂ₑ[i] = 0.0
          end
        end
        #
        integrate_G!(Fₑ,X,Dᵢⱼₖₗ,Ψ₁ₑ,Ψ₂ₑ,
                     N,∂N∂a,∂N∂x,Jac,Jac⁻¹,
                     ∇U₁,∇U₂,sym∇U₁,sym∇U₂,e₁₂,
                     εᵛ₁,εᵛ₂,eᵛ₁₂,σᵛ₁,σᵛ₂,σᵛ₁₂,
                     sycolptrm∇,sym∇ⁿˡ₁,sym∇ⁿˡ₂,
                     nn,etype,qr)
        #
        for i = 1:nn*dim
          if (dofs[i]>0)
            @inbounds res[neq+dofs[i]] -= Fₑ[i]*mult
          end
        end
        #
      end
    end
  end
  #
  return nothing
  #
end


"""
"""
function assembler_MK!(mesh::Grid,U::Field,K::SparseMatrixCSC{Float64,Int64},M::SparseMatrixCSC{Float64,Int64})

  fill!(K.nzval,0.0)
  fill!(M.nzval,0.0)
                            #
  X = zeros(Float64,nne_max*dim)
  dofs = zeros(Int64,nne_max*dim)
  Kₑ = zeros(Float64,(nne_max*dim,nne_max*dim))
  Mₑ = zeros(Float64,(nne_max*dim,nne_max*dim))
  #
  N = zeros(Float64,nne_max)
  ∂N∂a = zeros(Float64,(nne_max,dim))
  ∂N∂x = zeros(Float64,(nne_max,dim))
  sym∇ = zeros(Float64,(nne_max*dim,dim*(dim-1)))
  Jac = zeros(Float64,(dim,dim))
  Jac⁻¹ = zeros(Float64,(dim,dim))
  #
  for iΩ ∈ mesh.Ω
    Dᵢⱼₖₗ = iΩ.mat.Dᵢⱼₖₗ
    ρ = iΩ.mat.ρ
    for set = 1:iΩ.Sen
      etype = iΩ.Set[set]
      qr = select_quadrature_rule(etype)
      nn = iΩ.Senn[set]
      skip = iΩ.eskip[set]
      for e = 1:iΩ.ne[set]
        conn = @view iΩ.e2n[skip+1+(e-1)*nn:skip+e*nn]
        get_coor!(mesh,conn,X,nn)
        dofs!(U,nn,conn,dofs)
        integrate_MK!(Mₑ,Kₑ,X,ρ,Dᵢⱼₖₗ,N,∂N∂a,∂N∂x,sym∇,Jac,Jac⁻¹,nn,etype,qr)
        for irow = 1:nn*dim
          if dofs[irow]>0
            for jcol = 1:nn*dim
              if dofs[jcol]>0
                K[dofs[irow],dofs[jcol]]+=Kₑ[irow,jcol]
                M[dofs[irow],dofs[jcol]]+=Mₑ[irow,jcol]
              end
            end
          end
        end
      end
    end
  end
  return 
end
