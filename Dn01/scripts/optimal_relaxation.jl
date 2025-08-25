using Dn01
using Plots

gr()

# Build a small test system (same one you used earlier)
M = [4.0 -1.0  0.0;
     -1.0 4.0 -1.0;
      0.0 -1.0 3.0]

A = sparseFromDense(M)
b = [15.0, 10.0, 10.0]
x0 = zeros(3)

# Sweep over ω values
ws = 0.5:0.05:1.9
iters = Int[]

for w in ws
    try
        _, k = sor(A, b, x0, w; tol=1e-8, maxiter=10_000)
        push!(iters, k)
    catch
        # If it doesn't converge, record NaN
        push!(iters, NaN)
    end
end

# Plot
plot(ws, iters,
     xlabel="ω",
     ylabel="Iterations to Converge",
     title="SOR Convergence vs ω",
     legend=false,
     lw=2,
     marker=:circle)