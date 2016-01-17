using NearestNeighbors

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

import Distances: Metric, evaluate
immutable CustomMetric <: Metric end
evaluate(::CustomMetric, a::AbstractVector, b::AbstractVector) = norm(a - b)

# TODO: Cityblock()
const metrics = [Chebyshev(), Euclidean(), Minkowski(3.5)]
const fullmetrics = [metrics; Hamming(); CustomMetric()]
const trees = [KDTree, BallTree]
const trees_with_brute = [BruteTree; trees]

include("test_knn.jl")
include("test_inrange.jl")
include("test_monkey.jl")
include("datafreetree.jl")
