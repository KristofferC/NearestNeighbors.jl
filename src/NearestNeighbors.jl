__precompile__()

module NearestNeighbors

using Distances
import Distances: Metric, result_type, eval_reduce, eval_end, eval_op, eval_start, evaluate

using StaticArrays

import Base.show
import Compat.view

export BruteTree, KDTree, BallTree, DataFreeTree
export knn, inrange # TODOs? , allpairs, distmat, npairs
export injectdata

export Euclidean,
       Cityblock,
       Minkowski,
       Chebyshev,
       Hamming

# Change this to enable debugging
const DEBUG = false

abstract NNTree{V <: AbstractVector, P <: Metric}

typealias DistanceType Float64
typealias MinkowskiMetric Union{Euclidean, Chebyshev, Cityblock, Minkowski}

function check_input{V1, V2 <: AbstractVector}(::NNTree{V1}, ::Vector{V2})
    if length(V1) != length(V2)
        throw(ArgumentError(
            "dimension of input points:$(length(V2)) and tree data:$(length(V1)) must agree"))
    end
end

function check_input{V1, V2 <: Number}(::NNTree{V1}, point::Vector{V2})
    if length(V1) != length(point)
        throw(ArgumentError(
            "dimension of input points:$(length(point)) and tree data:$(length(V1)) must agree"))
    end
end

include("debugging.jl")
include("evaluation.jl")
include("tree_data.jl")
include("datafreetree.jl")
include("knn.jl")
include("inrange.jl")
include("hyperspheres.jl")
include("hyperrectangles.jl")
include("utilities.jl")
include("brute_tree.jl")
include("kd_tree.jl")
include("ball_tree.jl")
include("tree_ops.jl")

end # module
