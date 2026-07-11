module TestTreeNan
# Tests for KDTree, BallTree, BruteTree that reject data containing NaNs
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "TestSetup.jl"))
using ..Main.TestSetup
using NearestNeighbors
using StaticArrays
using Test

@testset "Trees reject NaNs" begin
    data_vec = [SVector{2,Float64}(NaN, 0.0), SVector{2,Float64}(1.0, 1.0)]
    data_mat = [NaN 0.0; 1.0 1.0]

    for TreeType in (KDTree, BallTree, BruteTree)
        @test_throws ArgumentError TreeType(data_vec)
        @test_throws ArgumentError TreeType(data_mat)
    end
end

@testset "Trees handle points at infinity" begin
    # Points at infinity are allowed: they are never returned as neighbors of
    # finite queries but must not corrupt the results for the finite points
    # (issue #78: Inf points used to poison BallTree bounding spheres with NaNs)
    coords = [29882.5 25974.3 Inf Inf 17821.8 Inf Inf Inf Inf Inf 16322.0;
              9279.86 9286.35 Inf Inf 10320.4 Inf Inf Inf Inf Inf 11459.0;
              0.0     0.0     Inf Inf 0.0     Inf Inf Inf Inf Inf 0.0]
    point = [17889.55, 2094.45, 0.0]
    for TreeType in (KDTree, BallTree, BruteTree)
        idxs, dists = knn(TreeType(coords), point, 1)
        @test idxs == [5]
        @test dists ≈ [8226.228994198984]
    end

    # Random stress with full-Inf and partial-Inf points of mixed signs
    for trial in 1:20
        dim = rand(1:3)
        n = rand(5:60)
        data = randn(dim, n) .* 10
        for _ in 1:rand(0:div(n, 3))
            j = rand(1:n)
            if rand() < 0.5
                data[:, j] .= rand((-Inf, Inf))
            else
                data[rand(1:dim), j] = rand((-Inf, Inf))
            end
        end
        btree = BruteTree(data)
        q = randn(dim) .* 10
        bidx, bdist = knn(btree, q, 2, true)
        for TreeType in (KDTree, BallTree)
            tree = TreeType(data; leafsize = rand(1:10))
            idx, dist = knn(tree, q, 2, true)
            @test idx == bidx
            @test dist ≈ bdist
            @test inrange(tree, q, 15.0, true) == inrange(btree, q, 15.0, true)
        end
    end
end

@testset "knn rejects NaNs" begin
    for TreeType in (KDTree, BallTree, BruteTree)
        data = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 1.0)]
        tree = TreeType(data)

        # Single query point (vector) containing NaN
        @test_throws ArgumentError knn(tree, [NaN, 0.0], 1)

        # Vector-of-vectors query containing NaN
        query_vec = [SVector{2,Float64}(NaN, 0.0)]
        @test_throws ArgumentError knn(tree, query_vec, 1)

        # Matrix query containing NaN
        query_mat = [NaN 0.0; 0.0 1.0]
        @test_throws ArgumentError knn(tree, query_mat, 1)
    end
end

@testset "inrange rejects NaNs" begin
    for TreeType in (KDTree, BallTree, BruteTree)
        data = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 1.0)]
        tree = TreeType(data)

        # Single query point (vector) containing NaN
        @test_throws ArgumentError inrange(tree, [NaN, 0.0], 1.0)

        # Vector-of-vectors query containing NaN
        query_vec = [SVector{2,Float64}(NaN, 0.0)]
        @test_throws ArgumentError inrange(tree, query_vec, 1.0)

        # Matrix query containing NaN
        query_mat = [NaN 0.0; 0.0 1.0]
        @test_throws ArgumentError inrange(tree, query_mat, 1.0)
    end
end

@testset "inrangecount rejects NaNs" begin
    for TreeType in (KDTree, BallTree, BruteTree)
        data = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 1.0)]
        tree = TreeType(data)

        # Single query point (vector) containing NaN
        @test_throws ArgumentError inrangecount(tree, [NaN, 0.0], 1.0)

        # Vector-of-vectors query containing NaN
        query_vec = [SVector{2,Float64}(NaN, 0.0)]
        @test_throws ArgumentError inrangecount(tree, query_vec, 1.0)

        # Matrix query containing NaN
        query_mat = [NaN 0.0; 0.0 1.0]
        @test_throws ArgumentError inrangecount(tree, query_mat, 1.0)
    end
end

end # module
