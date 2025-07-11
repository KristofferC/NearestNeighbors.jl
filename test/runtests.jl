using NearestNeighbors
using StaticArrays

using Test
using LinearAlgebra

using Distances: Distances, Metric, MinkowskiMetric, evaluate, PeriodicEuclidean
struct CustomMetric1 <: Metric end
Distances.evaluate(::CustomMetric1, a::AbstractVector, b::AbstractVector) = maximum(abs.(a .- b))
function NearestNeighbors.interpolate(::CustomMetric1,
                                      a::V,
                                      b::V,
                                      x,
                                      d,
                                      ab) where {V <: AbstractVector}
    idx = (abs.(b .- a) .>= d - x)
    c = copy(Array(a))
    c[idx] = (1 - x / d) * a[idx] + (x / d) * b[idx]
    return c, true
end
struct CustomMetric2 <: Metric end
Distances.evaluate(::CustomMetric2, a::AbstractVector, b::AbstractVector) = norm(a - b) / (norm(a) + norm(b))

# TODO: Cityblock()
const metrics = [Chebyshev(), Euclidean(), Minkowski(3.5)]
const fullmetrics = [metrics; Hamming(); CustomMetric1(); CustomMetric2()]
const trees = [KDTree, BallTree]
const trees_with_brute = [BruteTree; trees]

include("test_knn.jl")
include("test_inrange.jl")
include("test_monkey.jl")
include("test_datafreetree.jl")

@testset "views of SVector" begin
    x = [rand(SVector{3}) for i in 1:20]
    for T in (KDTree, BruteTree, BallTree)
        for reorder in (true, false)
            S = T(x; reorder)
            @test S isa T
        end
    end
end

@testset "periodic euclidean" begin
    pred = PeriodicEuclidean([Inf, 2.5])
    l = [0.0 0.0; 0.0 2.5]
    S = BallTree(l, pred)
    @test inrange(S,[0.0,0.0], 1e-2, true) == [1, 2]
end

using NearestNeighbors: HyperRectangle, get_min_distance_no_end, get_max_distance_no_end
@testset "hyperrectangle" begin
    ms = (Chebyshev(), Cityblock(), Minkowski(3.5), Euclidean())
    hr = HyperRectangle([-1.0, -2.0], [1.0, 2.0])

    # Point inside
    point = [-0.5, 0.3]
    closest_point = [-0.5, 0.3]
    furthest_point = [1.0, -2.0]
    for m in ms
        @test get_min_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(closest_point, point))
        @test get_max_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(furthest_point, point))
    end

    # Point outside both axis
    point = [1.5, 2.3]
    closest_point = [1.0, 2.0]
    furthest_point = [-1.0, -2.0]
    for m in ms
        @test get_min_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(closest_point, point))
        @test get_max_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(furthest_point, point))
    end

    # Point outside one axis
    point = [0.5, 2.3]
    closest_point = [0.5, 2.0]
    furthest_point = [-1.0, -2.0]
    for m in ms
        @test get_min_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(closest_point, point))
        @test get_max_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(furthest_point, point))
    end
end
