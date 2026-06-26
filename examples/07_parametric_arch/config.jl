# Shared configuration — edit ONLY here.
# Consumed by: main.jl, reference/compute_references.jl,
#              validation/backbone.jl, validation/plot_backbone.py

const h0_L_ratio = 0.005   # base arch rise / span  (θ = 0 configuration)
const N_INCREMENTS = 2      # intervals in [0 … 2·h0_L_ratio]; gives N+1 reference arches
#                              θ = h_ratio / h0_L_ratio − 1  maps [0, 2h0] → [−1, +1]
const a_max_mm = 100.0        # physical amplitude display cap (mm) — applied in plots, not data
