# Taylor expansion of erf(x) around 0
function erf_taylor_slow(x::Float64; terms::Int)
    result = 0.0
    for n in 0:(terms-1)
        term = ((-1)^n / (factorial(n) * (2n + 1))) * (x^(2n + 1))
        result += term
    end
    return (2 / sqrt(pi)) * result
end

function erf_taylor(x::Float64; terms::Int=10)
    result = 0.0
    term = x
    for n in 0:(terms-1)
        result += term / (2n+1)
        term *= -x^2 / (n+1)
    end
    return (2/sqrt(pi)) * result
end

# CDF of standard normal using erf_taylor
function cdf_taylor(x::Float64; terms::Int)
    return 0.5 * (1 + erf_taylor(x / sqrt(2), terms=terms))
end

