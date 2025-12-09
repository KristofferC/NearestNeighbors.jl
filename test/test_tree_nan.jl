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
