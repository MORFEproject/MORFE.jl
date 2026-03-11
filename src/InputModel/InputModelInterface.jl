#=
AbstractInputModel expects first order ODE-System:
    B d_t X = AX + F_nl(X) + F(t)
=#
abstract type AbstractInputModel end

function assemble!(fem::AbstractInputModel)
    error("assemble! not implemented for $(typeof(fem))")
end

function a_matrix(fem::AbstractInputModel)
    error("a_matrix not implemented for $(typeof(fem))")
end

function b_matrix(fem::AbstractInputModel)
    error("b_matrix not implemented for $(typeof(fem))")
end

function f_vector(fem::AbstractInputModel)
    error("f_vector not implemented for $(typeof(fem))")
end


function evaluate_nonlinearity(fem::AbstractInputModel, Ψ...)
    error("evaluate_nonlinearity not implemented for $(typeof(fem))")
end

function ndofs(fem::AbstractInputModel)
    error("ndofs not implemented for $(typeof(fem))")
end

function field_from_vector(fem::AbstractInputModel, u)
    error("field_from_vector not implemented for $(typeof(fem))")
end

function visualize(fem::AbstractInputModel, u; kwargs...)
    error("visualize not implemented for $(typeof(fem))")
end

#=
Representation of second order ODE-System:
    M d_t² U  +  C d_t U + KU  +  F_nl(U) = F(t)
=#
abstract type AbstractInputModelSecondOrder <: AbstractInputModel end

function assemble!(fem::AbstractInputModelSecondOrder)
    assemble_mass_matrix!(fem)
    assemble_stiffness_matrix!(fem)
    assemble_damping_matrix!(fem)
    assemble_load_vector!(fem)
end

function assemble_mass_matrix!(fem::AbstractInputModelSecondOrder)
    error("assemble_mass_matrix! not implemented for $(typeof(fem))")
end

function assemble_stiffness_matrix!(fem::AbstractInputModelSecondOrder)
    error("assemble_stiffness_matrix! not implemented for $(typeof(fem))")
end

function assemble_damping_matrix!(fem::AbstractInputModelSecondOrder)
    error("assemble_damping_matrix! not implemented for $(typeof(fem))")
end

function assemble_load_vector!(fem::AbstractInputModelSecondOrder)
    error("assemble_load_vector! not implemented for $(typeof(fem))")
end

function a_matrix(fem::AbstractInputModelSecondOrder)
    error("For $(typeof(fem)) use structure of second order system!")
end

function b_matrix(fem::AbstractInputModelSecondOrder)
    error("For $(typeof(fem)) use structure of second order system!")
end

function f_vector(fem::AbstractInputModelSecondOrder)
    error("For $(typeof(fem)) use structure of second order system!")
end

function mass_matrix(fem::AbstractInputModelSecondOrder)
    error("mass_matrix not implemented for $(typeof(fem))")
end

function stiffness_matrix(fem::AbstractInputModelSecondOrder)
    error("stiffness_matrix not implemented for $(typeof(fem))")
end

function damping_matrix(fem::AbstractInputModelSecondOrder)
    error("damping_matrix not implemented for $(typeof(fem))")
end

function load_vector(fem::AbstractInputModelSecondOrder)
    error("load_vector not implemented for $(typeof(fem))")
end