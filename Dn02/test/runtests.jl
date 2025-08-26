using Test
using Dn02
using Distributions   # for reference Normal CDF

Φ(x) = cdf(Normal(0,1), x)

@testset "Dn02 tests" begin

    @testset "Taylor approximation" begin
        @test isapprox(cdf_taylor(0.0; terms=10), Φ(0.0); atol=1e-12)
        @test isapprox(cdf_taylor(1.0; terms=12), Φ(1.0); atol=1e-8)
        @test isapprox(cdf_taylor(-1.0; terms=12), Φ(-1.0); atol=1e-8)
    end

    @testset "Gauss–Legendre approximation" begin
        for x in 2.0:0.5:5.0
            @test isapprox(cdf_legendre(x; n=12), Φ(x); atol=1e-9)
            @test isapprox(cdf_legendre(-x; n=14), Φ(-x); atol=1e-9)
        end
    end

    @testset "Asymptotic expansion" begin
        for x in 6.0:2.0:12.0
            @test isapprox(cdf_asymptotic(x; terms=10), Φ(x); atol=1e-12)
        end
    end

    @testset "My CDF" begin
        for x in -8.0:0.5:8.0
            @test isapprox(my_cdf(x; taylor_terms=12, gauss_terms=12, asymptotic_terms=10),
                           Φ(x); atol=1e-9)
        end
    end

end