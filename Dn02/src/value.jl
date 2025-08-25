using QuadGK

# --- 2-point Gauss-Legendre nodes (on [-1,1]) ---
const nodes = [-1/sqrt(3), 1/sqrt(3)]
const weights = [1.0, 1.0]   # both weights are 1

"""
Single interval Gauss–Legendre 2-point rule on [a,b].
"""
function gauss2_interval(f::Function, a::Float64, b::Float64)
    mid = (a + b) / 2
    half = (b - a) / 2
    s = 0.0
    for i in 1:2
        x = mid + half * nodes[i]
        s += weights[i] * f(x)
    end
    return half * s
end

"""
Composite Gauss–Legendre 2-point quadrature.
Splits [a,b] into m subintervals.
"""
function gauss2_composite(f::Function, a::Float64, b::Float64, m::Int)
    h = (b - a) / m
    s = 0.0
    for j in 0:(m-1)
        xj, xj1 = a + j*h, a + (j+1)*h
        s += gauss2_interval(f, xj, xj1)
    end
    return s
end

f(x) = sin(x)/x

# Approximate sin(x)/x dx with m=2, m=4
val2 = gauss2_composite(f, 0.0, 5.0, 2)
val4 = gauss2_composite(f, 0.0, 5.0, 70)

println("m=2 → $val2")
println("m=4 → $val4")

true_val, err = quadgk(f, 0.0, 5.0)   # adaptive Gauss-Kronrod
println("True ≈ $val (error estimate $err)")


abs_err = abs(val4 - true_val)
rel_err = abs_err / max(abs(val4), eps())

println("Error  = $(abs_err)")
println("Relative Error  = $(rel_err)")

function gauss2_error_estimate(f, a, b, m)
    I_m  = gauss2_composite(f, a, b, m)
    I_2m = gauss2_composite(f, a, b, 2m)
    err_est = abs(I_2m - I_m)
    return I_2m, err_est
end

f(x) = sin(x)/x

val, err_est = gauss2_error_estimate(f, 0.0, 5.0, 40)

println("Approx integral ≈ $val")
println("Estimated error ≈ $err_est")