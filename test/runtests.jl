using NearestNeighbors

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

const metrics = [Chebyshev(), Euclidean(), Minkowski(3.5)]
const fullmetrics = [metrics; Hamming()]
const trees = [KDTree, BallTree]
const trees_with_brute = [BruteTree; trees]

include("test_knn.jl")
include("test_inrange.jl")
include("test_monkey.jl")
