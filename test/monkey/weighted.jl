module TestMonkeyWeighted
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "..", "TestSetup.jl"))
using ..Main.TestSetup
using NearestNeighbors
using Test
using Distances: WeightedEuclidean, WeightedCityblock, WeightedMinkowski

@testset "weighted metrics monkey" for TreeType in (KDTree, BallTree), T in (Float32, Float64)
    # Weights must match the data dimension, so the metric is constructed per
    # trial. Compares vs brute force.
    for i in 1:20
        dim_data = rand(1:4)
        size_data = rand(100:151)
        data = (rand(T, dim_data, size_data) .- T(0.2)) .* T(5)
        w = rand(T, dim_data) .+ T(0.5)
        for metric in (WeightedEuclidean(w), WeightedCityblock(w), WeightedMinkowski(w, 3.0))
            tree = TreeType(data, metric; leafsize = rand(1:15))
            btree = BruteTree(data, metric)
            p = (rand(T, dim_data) .- T(0.2)) .* T(5)
            k = rand(1:12)
            idx, dist = knn(tree, p, k, true)
            bidx, bdist = knn(btree, p, k, true)
            @test idx == bidx
            @test dist ≈ bdist
            r = T(2.0)
            @test inrange(tree, p, r, true) == inrange(btree, p, r, true)
        end
    end
end

end # module
