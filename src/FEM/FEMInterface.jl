abstract type AbstractFEM end

function assemble_system(fem::AbstractFEM)
    assemble_mass_matrix(fem)
    assemble_stiffness_matrix(fem)
    assemble_load_vector(fem)
end

function assemble_mass_matrix!(fem::AbstractFEM)
    error("assemble_mass_matrix! not implemented for $(typeof(fem))")
end

function assemble_stiffness_matrix!(fem::AbstractFEM)
    error("assemble_stiffness_matrix! not implemented for $(typeof(fem))")
end

function assemble_load_vector!(fem::AbstractFEM)
    error("assemble_load_vector! not implemented for $(typeof(fem))")
end

function mass_matrix(fem::AbstractFEM)
    error("mass_matrix not implemented for $(typeof(fem))")
end

function stiffness_matrix(fem::AbstractFEM)
    error("stiffness_matrix not implemented for $(typeof(fem))")
end

function load_vector(fem::AbstractFEM)
    error("load_vector not implemented for $(typeof(fem))")
end

function evaluate_quadratic_nonlinearity(fem::AbstractFEM)
    error("evaluate_quadratic_nonlinearity not implemented for $(typeof(fem))")
end

function evaluate_cubic_nonlinearity(fem::AbstractFEM)
    error("evaluate_cubic_nonlinearity not implemented for $(typeof(fem))")
end

function ndofs(fem::AbstractFEM)
    error("ndofs not implemented for $(typeof(fem))")
end

function field_from_vector(fem::AbstractFEM, u)
    error("field_from_vector not implemented for $(typeof(fem))")
end

function visualize(fem::AbstractFEM, u; kwargs...)
    error("visualize not implemented for $(typeof(fem))")
end