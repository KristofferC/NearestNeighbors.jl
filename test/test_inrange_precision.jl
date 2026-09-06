module TestInrangePrecision
using NearestNeighbors, Test

@testset "BallTree query sphere preserves coordinate precision" begin
    tree = BallTree([0 2])
    periodic = PeriodicTree(tree, [0], [4])
    for t in (tree, periodic)
        @test inrange(t, [0.5], 1) == [1]
        @test inrangecount(t, [0.5], 1) == 1
    end
    tree = BallTree(reshape(Float32[1], 1, :))
    periodic = PeriodicTree(tree, Float32[0], Float32[4])
    q = [1.0 + 0.6 * eps(1.0f0)]
    radius = Float32(0.75 * eps(1.0f0))
    @test abs(q[1] - 1) < radius
    for t in (tree, periodic)
        @test inrange(t, q, radius) == [1]
        @test inrangecount(t, q, radius) == 1
        @test inrange(t, reshape(q, 1, :), radius) == [[1]]
    end
end

end
