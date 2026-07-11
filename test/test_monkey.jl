module TestMonkey
# This contains a bunch of random tests that should hopefully detect if
# some edge case has been missed in the real tests
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "TestSetup.jl"))
using ..Main.TestSetup: fullmetrics, trees_with_brute
using NearestNeighbors
using Test
using Distances: MinkowskiMetric, Hamming, WeightedEuclidean, WeightedCityblock, WeightedMinkowski

@testset "metric" for metric in fullmetrics
    @testset "tree type" for TreeType in trees_with_brute
        @testset "type" for T in (Float32, Float64)
            @testset "knn monkey" begin
                # Checks that we find existing point in the tree
                # and that it is the closest
                if TreeType == KDTree && !isa(metric, MinkowskiMetric)
                    continue
                elseif TreeType == BallTree && isa(metric, Hamming)
                    continue
                end
                for i in 1:30
                    dim_data = rand(1:4)
                    size_data = rand(1000:1300)
                    data = rand(T, dim_data, size_data)
                    for j = 1:5
                        tree = TreeType(data, metric; leafsize = rand(1:15))
                        n = rand(1:size_data)
                        idx, dist = knn(tree, data[:,n], rand(1:30), true)
                        @test issorted(dist) == true
                        @test n == idx[1]
                    end
                end

                # Compares vs Brute Force
                for i in 1:30
                    dim_data = rand(1:5)
                    size_data = rand(100:151)
                    data = rand(T, dim_data, size_data)
                    leafsize = rand(1:15)
                    tree = TreeType(data, metric; leafsize)
                    btree = BruteTree(data, metric)
                    k = rand(1:12)
                    p = rand(dim_data)
                    idx, dist = knn(tree, p, k, true)
                    bidx, bdist = knn(btree, p, k, true)
                    @test idx == bidx
                    @test dist ≈ bdist
                end
            end

            @testset "inrange monkey" begin
                # Test against brute force
                for i in 1:30
                    dim_data = rand(1:6)
                    size_data = rand(20:250)
                    data = rand(T, dim_data, size_data)
                    tree = TreeType(data, metric; leafsize = rand(1:8))
                    btree = BruteTree(data, metric)
                    p = 0.5 * ones(dim_data)
                    r = 0.3

                    idxs = inrange(tree, p, r, true)
                    bidxs = inrange(btree, p, r, true)

                    @test idxs == bidxs
                end
            end

            @testset "coupled monkey" begin
                for i in 1:50
                    dim_data = rand(1:5)
                    size_data = rand(100:1000)
                    data = randn(T, dim_data, size_data)
                    tree = TreeType(data, metric; leafsize = rand(1:8))
                    point = randn(dim_data)
                    idxs_ball = Int[]
                    r = 0.1
                    while length(idxs_ball) < 10
                        r *= 2.0
                        idxs_ball = inrange(tree, point, r, true)
                    end
                    idxs_knn, dists = knn(tree, point, length(idxs_ball))

                    @test sort(idxs_knn) == sort(idxs_ball)
                end
            end
        end
    end
end

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
