function cdf_asymptotic(x::Float64; terms::Int)
    if x <= 0
        error("Use this only for large positive x")
    end
    factor = exp(-x^2 / 2) / (x * sqrt(2π))
    series = 1.0
    term = 1.0
    for n in 1:(terms-1)
        term *= -(2n - 1) / (x^2)   # recursive generation of coefficients
        series += term
    end
    Q =  factor * series
    return 1.0 - Q
end
