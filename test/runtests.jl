@info "Running thests with $(Threads.nthreads()) threads"

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
include("test_tree_tree_inrange.jl")
include("test_monkey.jl")
include("test_datafreetree.jl")
include("test_tree_data.jl")
include("test_periodic.jl")
include("test_tree_nan.jl")
include("test_unitful.jl")
include("test_tree_walk.jl")
include("test_misc.jl")
