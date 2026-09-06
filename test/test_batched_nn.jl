module TestBatchedNN
using NearestNeighbors, StaticArrays, Test

@testset "Batched scalar nearest-neighbor search" begin
    for Tree in (KDTree, BallTree, BruteTree), T in (Float32, Float64), reorder in (false, true)
        data = T[0.1 0.3 0.6 0.9; 0.2 0.8 0.5 0.1]
        tree = Tree(data; leafsize=1, reorder)
        periodic = PeriodicTree(tree, zeros(T, 2), ones(T, 2))
        queries = [0.12 0.7 0.91; 0.21 0.6 0.05]
        for t in (tree, periodic), skip in (Returns(false), ==(1))
            expected = [nn(t, q, skip) for q in eachcol(queries)]
            for points in (queries, view(queries, :, :), collect.(eachcol(queries)), SVector{2}.(eachcol(queries)))
                idx, dist = nn(t, points, skip)
                @test idx == first.(expected)
                @test dist ≈ T.(last.(expected))
                @test eltype(dist) === T
            end
            @test nn(t, zeros(2, 0), skip) == (Int[], T[])
            @test nn(t, SVector{2,Float64}[], skip) == (Int[], T[])
            @test_throws ArgumentError nn(t, queries, Returns(true))
            @test_throws ArgumentError nn(t, collect.(eachcol(queries)), Returns(true))
            @test_throws ArgumentError nn(t, zeros(3, 2), skip)
            @test_throws ArgumentError nn(t, [[0.0, 0.0], [0.0]], skip)
            @test_throws ArgumentError nn(t, [NaN 0.0; 0.0 0.0], skip)
        end
    end
end
end
