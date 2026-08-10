using Symbolics
using MORFE

@variables z[1:2] dz[1:2] r[1:2]
groups = (collect(z), collect(dz))
ext_var = collect(r)

exprs = [
    -dz[1] - z[1] - 1.5*z[1] - 2*z[1]^2 + 3*z[1]*r[1] + 5*r[1],
    -dz[2] - 3.5*z[2] + z[1]^2 - 5*z[1]*z[2] + (21//4)*z[2]^2 + 2*r[2]
]

# dt/dr ext_var = ext_exprs
ext_exprs = [
    1 * r[1] + 1.5 * r[2] + 3 * r[1] * r[2] + 4 * r[2]^2 * r[1] + 4 * r[1]^2,
    4 * r[2] + r[2] * r[1] + 1 * r[2]^3 * r[1] + r[2]^2 * r[1]
]

model = symbolics_to_NthOrderModel(exprs, groups, ext_var, ext_exprs)
