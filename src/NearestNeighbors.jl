#__precompile__()

module NearestNeighbors

using Devectorize
using Distances

import Distances: Metric, result_type, eval_reduce, eval_end, eval_op, eval_start, evaluate

import Base.show

export BruteTree, KDTree, BallTree
export knn, inrange # TODOs? , allpairs, distmat, npairs

 export Euclidean,
        Cityblock,
        Minkowski,
        Chebyshev,
        Hamming


# Change this to enable debugging
const DEBUG = false

abstract NNTree{T <: AbstractFloat, P <: Metric}

function check_input(tree::NNTree, points::AbstractArray)
    ndim_points = size(points,1)
    ndim_tree = size(tree.data, 1)
    if ndim_points != ndim_tree
        throw(ArgumentError(
            "dimension of input points:$(ndim_points) and tree data:$(ndim_tree) must agree"))
    end
end

include("debugging.jl")
include("evaluation.jl")
include("tree_data.jl")
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
