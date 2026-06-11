function build_exact_nonlinear_maps(dh, cv, free_to_local::Dict{Int,Int}, n_free::Int,
                                     λ::Float64, μ::Float64;
                                     max_unique_cols::Int)
    term_quad = FerriteGeometricNonlinearity{2}(dh, cv, free_to_local, n_free, λ, μ;
                                                max_unique_cols = max_unique_cols)
    term_cube = FerriteGeometricNonlinearity{3}(dh, cv, free_to_local, n_free, λ, μ;
                                                max_unique_cols = max_unique_cols)
    return (term_quad, term_cube)
end
