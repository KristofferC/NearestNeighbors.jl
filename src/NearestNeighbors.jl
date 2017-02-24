__precompile__()

module NearestNeighbors

using Distances
import Distances: Metric, result_type, eval_reduce, eval_end, eval_op, eval_start, evaluate

using StaticArrays
using Compat

import Base.show


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

@compat abstract type NNTree{V <: AbstractVector, P <: Metric} end

const MinkowskiMetric = Union{Euclidean, Chebyshev, Cityblock, Minkowski}

function check_input{V1, V2 <: AbstractVector}(::NNTree{V1}, ::AbstractVector{V2})
    if length(V1) != length(V2)
        throw(ArgumentError(
            "dimension of input points:$(length(V2)) and tree data:$(length(V1)) must agree"))
    end
end

function check_input{V1, T <: Number}(::NNTree{V1}, point::AbstractVector{T})
    if length(V1) != length(point)
        throw(ArgumentError(
            "dimension of input points:$(length(point)) and tree data:$(length(V1)) must agree"))
    end
end

get_T{T <: AbstractFloat}(::Type{T}) = T
get_T{T}(::T) = Float64

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
