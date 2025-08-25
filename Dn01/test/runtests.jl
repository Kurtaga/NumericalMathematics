using Test
using Dn01

@testset "SparseMatrix basics" begin
    # Small dense matrix
    M = [1.0 0.0 2.0;
         0.0 3.0 0.0;
         4.0 0.0 5.0]

    A = sparseFromDense(M)

    # Roundtrip: dense → sparse → dense (before modifications)
    M_back = [A[i,j] for i in 1:3, j in 1:3]
    @test M_back == M

    # Test size
    @test size(A) == (3,3)

    # Test getindex
    @test A[1,1] == 1.0
    @test A[1,3] == 2.0
    @test A[2,2] == 3.0
    @test A[2,3] == 0.0
    @test A[3,1] == 4.0
    @test A[3,3] == 5.0

    # Test setindex! (update existing)
    A[1,1] = 9.0
    @test A[1,1] == 9.0

    # Test setindex! (insert new nonzero)
    A[2,3] = 7.0
    @test A[2,3] == 7.0

    # Now check dense form reflects the updates
    M_updated = [A[i,j] for i in 1:3, j in 1:3]
    @test M_updated == [9.0 0.0 2.0;
                        0.0 3.0 7.0;
                        4.0 0.0 5.0]

    # Test multiplication
    x = [1.0, 2.0, 3.0]
    y = A * x
    @test y ≈ (M_updated * x)

    # Test negation
    B = -A
    for i in 1:3, j in 1:3
        @test B[i,j] == -A[i,j]
    end
end


@testset "SOR method" begin
    # Small 2x2 diagonally dominant system
    M = [4.0 1.0;
         2.0 3.0]
    A = sparseFromDense(M)
    b = [1.0, 2.0]

    # Test ω=1 (Gauss–Seidel) converges
    x, iters = sor(A, b, 1.0)
    @test isapprox(A * x, b; atol=1e-8)
    @test iters > 0

    # Test under-relaxation (ω≈0.8) converges but slower
    x3, iters3 = sor(A, b, 0.8)
    @test isapprox(A * x3, b; atol=1e-8)
    @test iters3 > iters

    # Test zero diagonal triggers error
    Mbad = [0.0 1.0;
            1.0 2.0]
    Abad = sparseFromDense(Mbad)
    @test_throws ErrorException sor(Abad, b, 1.0)

    Mdiv = [1.0 2.0;
            2.0 1.0]   # not diagonally dominant
    Adiv = sparseFromDense(Mdiv)
    bdiv = [1.0, 1.0]

    @test_throws ErrorException sor(Adiv, bdiv, 1.5)

    # 3x3 system test
    M3 = [4.0 -1.0  0.0;
          -1.0  4.0 -1.0;
           0.0 -1.0  3.0]
    A3 = sparseFromDense(M3)
    b3 = [15.0, 10.0, 10.0]
    x3, iters3 = sor(A3, b3, 1.2)
    @test isapprox(A3 * x3, b3; atol=1e-8)
end