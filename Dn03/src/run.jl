include("Dn03.jl"); using .Dn03
using Plots

∇F1(x, y, z) = [4x^3, y, 2z]
∇F2(x, y, z) = [2x, 2y, -8z]

u0 = [1.758, 2.215, 0.0]
traj = Dn03.integrate_explicit_euler(u0, 1e-3, ∇F1, ∇F2)

# extract coordinates
xs = [p[1] for p in traj]
ys = [p[2] for p in traj]
zs = [p[3] for p in traj]

# plot
plot3d(xs, ys, zs, seriestype = :line, linewidth = 2, color = :blue, label = "Implicit curve")



println("Curve length: ", curve_length(traj))