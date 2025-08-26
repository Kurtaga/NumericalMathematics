module Dn02

using SpecialFunctions
using Distributions  # for Normal() and cdf()

export erf_taylor, cdf_taylor, cdf_legendre, cdf_asymptotic, my_cdf

include("taylor.jl")
include("gauss_legendre.jl")
include("asymptotic.jl")

function my_cdf(x::Float64; 
                  taylor_terms::Int=10,
                  gauss_terms::Int=12, 
                  asymptotic_terms::Int=10)

    if x < 0
        return 1.0 - my_cdf(-x; 
                              taylor_terms=taylor_terms,
                              gauss_terms=gauss_terms,
                              asymptotic_terms=asymptotic_terms)
    end

    # from here on, x ≥ 0
    if x < 2.0
        return cdf_taylor(x; terms=taylor_terms)

    elseif x <= 5.0
        return cdf_legendre(x, n=gauss_terms)

    else
        return cdf_asymptotic(x; terms=asymptotic_terms)
    end
end

# Demo vs. Distributions.jl
for x in [-10, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 2.75, 3.25, 3.5, 5.0, 10.0]
    approx = my_cdf(x)
    actual = cdf(Normal(0,1), x)
    abs_err = abs(approx - actual)
    rel_err = abs_err / max(abs(actual), eps())  # eps() prevents div by 0
    println("x=$x, approx=$approx, actual=$actual, abs_err=$abs_err, rel_err=$rel_err")
end

end # module Dn02
