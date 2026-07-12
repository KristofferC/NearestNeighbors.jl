module TestSetup

using NearestNeighbors
using NearestNeighbors: KDTree, BallTree, BruteTree
using StaticArrays
using Test
using LinearAlgebra
using Distances: Distances, Metric, MinkowskiMetric, Chebyshev, Euclidean, Minkowski, Hamming,
                 WeightedEuclidean, evaluate, PeriodicEuclidean

struct CustomMetric1 <: Metric end
Distances.evaluate(::CustomMetric1, a::AbstractVector, b::AbstractVector) = maximum(abs.(a .- b))
function NearestNeighbors.interpolate(::CustomMetric1,
                                      a::V,
                                      b::V,
                                      x,
                                      d) where {V <: AbstractVector}
    idx = (abs.(b .- a) .>= d - x)
    c = copy(Array(a))
    c[idx] = (1 - x / d) * a[idx] + (x / d) * b[idx]
    return c, true
end

struct CustomMetric2 <: Metric end
Distances.evaluate(::CustomMetric2, a::AbstractVector, b::AbstractVector) = norm(a - b) / (norm(a) + norm(b))

const metrics = [Chebyshev(), Euclidean(), Minkowski(3.5)]
const fullmetrics = [metrics; Hamming(); CustomMetric1(); CustomMetric2()]
const trees = [KDTree, BallTree]
const trees_with_brute = [BruteTree; trees]

# Shared driver for the randomized "monkey" tests (see test/monkey/): builds
# trees of the given type over random data and checks the results, mostly
# against brute force. The metrics are passed by the caller since not every
# tree type supports every metric.
function monkey_tests(TreeType, metrics_to_test)
    @testset "metric" for metric in metrics_to_test
        @testset "type" for T in (Float32, Float64)
            @testset "knn monkey" begin
                # Checks that we find existing point in the tree
                # and that it is the closest
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

export metrics, fullmetrics, trees, trees_with_brute, CustomMetric1, CustomMetric2, monkey_tests

end
