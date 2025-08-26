using Test
using Dn03

@testset "Geometry utilities" begin
    # cross3 should match known cross product
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    @test cross3(a, b) ≈ [0.0, 0.0, 1.0]

    # norm3 should match sqrt of sum of squares
    v = [3.0, 4.0, 12.0]
    @test norm3(v) ≈ 13.0
end

@testset "RHS function" begin
    ∇F1(x, y, z) = [1.0, 0.0, 0.0]
    ∇F2(x, y, z) = [0.0, 1.0, 0.0]
    u = [0.0, 0.0, 0.0]
    # cross of (1,0,0) and (0,1,0) is (0,0,1)
    @test rhs(u, ∇F1, ∇F2) ≈ [0.0, 0.0, 1.0]
end

@testset "Explicit Euler step" begin
    ∇F1(x, y, z) = [1.0, 0.0, 0.0]
    ∇F2(x, y, z) = [0.0, 1.0, 0.0]
    u0 = [0.0, 0.0, 0.0]
    h = 0.1
    # step direction should be along +z
    u1 = explicit_euler_step(u0, h, ∇F1, ∇F2)
    @test isapprox(u1, [0.0, 0.0, h], atol=1e-10)
end

@testset "Integration" begin
    # For this toy case, tangent always points +z, so trajectory is a straight line
    ∇F1(x, y, z) = [1.0, 0.0, 0.0]
    ∇F2(x, y, z) = [0.0, 1.0, 0.0]
    u0 = [0.0, 0.0, 0.0]
    h = 0.1
    traj = integrate_explicit_euler(u0, h, ∇F1, ∇F2; max_steps=5, min_steps=10)

    # Expected straight line points
    expected = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.3],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 0.5]]

    @test length(traj) == length(expected)
    @test all(isapprox(t, e; atol=1e-10) for (t, e) in zip(traj, expected))
end