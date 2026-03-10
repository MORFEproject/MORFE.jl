#=
InputModelAbstract expects first order ODE-System:
    B d_t X = AX + F_nl(X) + F(t)
=#
abstract type InputModelAbstract end

function assemble!(fem::InputModelAbstract)
    assemble_mass_matrix!(fem)
    assemble_stiffness_matrix!(fem)
    assemble_load_vector!(fem)
end

function assemble_mass_matrix!(fem::InputModelAbstract)
    error("assemble_mass_matrix! not implemented for $(typeof(fem))")
end

function assemble_stiffness_matrix!(fem::InputModelAbstract)
    error("assemble_stiffness_matrix! not implemented for $(typeof(fem))")
end

function assemble_load_vector!(fem::InputModelAbstract)
    error("assemble_load_vector! not implemented for $(typeof(fem))")
end

function get_a_matrix(fem::InputModelAbstract)
    error("get_a_matrix not implemented for $(typeof(fem))")
end

function get_b_matrix(fem::InputModelAbstract)
    error("get_b_matrix not implemented for $(typeof(fem))")
end

function get_f_vector(fem::InputModelAbstract)
    error("get_f_vector not implemented for $(typeof(fem))")
end

function evaluate_nonlinearity(fem::InputModelAbstract, Ψ...)
    error("evaluate_nonlinearity not implemented for $(typeof(fem))")
end

function ndofs(fem::InputModelAbstract)
    error("ndofs not implemented for $(typeof(fem))")
end

function field_from_vector(fem::InputModelAbstract, u)
    error("field_from_vector not implemented for $(typeof(fem))")
end

function visualize(fem::InputModelAbstract, u; kwargs...)
    error("visualize not implemented for $(typeof(fem))")
end