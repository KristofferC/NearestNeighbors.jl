using FactCheck
using NearestNeighbors

const metrics = [Euclidean(), Minkowski(3.5), Cityblock()]

include("test_knn.jl")
include("test_inrange.jl")
include("test_monkey.jl")
