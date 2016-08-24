using NearestNeighbors

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

import Distances: Metric, evaluate
immutable CustomMetric1 <: Metric end
evaluate(::CustomMetric1, a::AbstractVector, b::AbstractVector) = maximum(abs(a - b))
function NearestNeighbors.interpolate{V <: AbstractVector}(::CustomMetric1,
                                                           a::V,
                                                           b::V,
                                                           x,
                                                           d,
                                                           ab)
    idx = (abs(b - a) .>= d - x)
    c = copy(a)
    c[idx] = (1 - x / d) .* a[idx] + (x / d) .* b[idx]
    return c, true
end
immutable CustomMetric2 <: Metric end
evaluate(::CustomMetric2, a::AbstractVector, b::AbstractVector) = norm(a - b) / (norm(a) + norm(b))

# TODO: Cityblock()
const metrics = [Chebyshev(), Euclidean(), Minkowski(3.5)]
const fullmetrics = [metrics; Hamming(); CustomMetric1(); CustomMetric2()]
const trees = [KDTree, BallTree]
const trees_with_brute = [BruteTree; trees]

include("test_knn.jl")
include("test_inrange.jl")
include("test_monkey.jl")
include("test_datafreetree.jl")
