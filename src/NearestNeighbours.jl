module NearestNeighbors

using Devectorize
using Distances
using Parameters

import Distances: Metric, result_type, eval_reduce, eval_end, eval_op, eval_start, evaluate

import Base.show

export BruteTree, KDTree, BallTree
export knn, inrange # TODOs? , allpairs, distmat, npairs

 export Euclidean,,
        Chebyshev,
        Cityblock,
        Minkowski
# TODO:
#        Hamming,
#        CosineDist,
#        CorrDist,
#        ChiSqDist,
#        KLDivergence,
#        JSDivergence,
#        SpanNormDist


include("debugging.jl")
include("nn_tree.jl")
include("evaluation.jl")
include("hyperspheres.jl")
include("hyperrectangles.jl")
include("utilities.jl")
include("brute_tree.jl")
include("kd_tree.jl")
include("ball_tree.jl")
include("tree_ops.jl")


end
