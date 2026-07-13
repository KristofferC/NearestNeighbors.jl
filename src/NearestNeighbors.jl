module NearestNeighbors

using Distances: Distances, PreMetric, Metric, UnionMinkowskiMetric, eval_reduce, eval_end, eval_op, eval_start, evaluate, parameters, Euclidean, Cityblock, Minkowski, Chebyshev, Hamming, Mahalanobis, WeightedEuclidean, WeightedCityblock, WeightedMinkowski

using StaticArrays: StaticArrays, MVector, SVector
using Base: setindex
using AbstractTrees: AbstractTrees

export NNTree, BruteTree, BruteTree!, KDTree, KDTree!, BallTree, BallTree!, DataFreeTree, PeriodicTree
export knn, knn!, nn, allnn, allknn, inrange, inrange!, inrangecount, inrange_pairs # TODOs?, npairs
export injectdata
export TreeNode, treeroot, leafpoints, leaf_point_indices, treeregion
export preorder, postorder, leaves
export HyperRectangle, HyperSphere

export Euclidean,
       Cityblock,
       Minkowski,
       Chebyshev,
       Hamming,
       WeightedEuclidean,
       WeightedCityblock,
       WeightedMinkowski

abstract type NNTree{V <: AbstractVector,P <: PreMetric} end

# Trees whose storage can be taken over by the mutating constructors
# (KDTree!, BallTree!, BruteTree!) override this to error when a consumed
# tree is used.
@inline check_valid(::NNTree) = nothing
@noinline throw_consumed(tree) = throw(ArgumentError(
    "the storage of this tree was taken over by `$(nameof(typeof(tree)))!`; the tree can no longer be used"))

const NonweightedMinkowskiMetric = Union{Euclidean,Chebyshev,Cityblock,Minkowski}
const WeightedMinkowskiMetric = Union{WeightedEuclidean,WeightedCityblock,WeightedMinkowski}
const MinkowskiMetric = UnionMinkowskiMetric

# `length` of a point type when the type itself encodes it (e.g. `SVector`),
# otherwise `nothing`
static_length(::Type{V}) where {V <: StaticArrays.StaticArray} = length(V)
static_length(::Type) = nothing

# Dimension of the points stored in the tree. Point types like `Vector` do not
# encode their length in the type, so it is read from the stored data;
# `nothing` if it cannot be determined (no data stored).
function tree_dimension(tree::NNTree{V}) where {V}
    dim = static_length(V)
    dim !== nothing && return dim
    data = get_tree(tree).data
    return isempty(data) ? nothing : length(first(data))
end

@noinline throw_dimension_mismatch(point_dim, tree_dim) = throw(ArgumentError(
    "dimension of input points:$(point_dim) and tree data:$(tree_dim) must agree"))

function check_input(tree::NNTree, points::AbstractVector{V2}) where {V2 <: AbstractVector}
    check_valid(tree)
    dim = tree_dimension(tree)
    dim === nothing && return
    for p in points
        length(p) == dim || throw_dimension_mismatch(length(p), dim)
    end
end

function check_input(tree::NNTree, point::AbstractVector{T}) where {T <: Number}
    check_valid(tree)
    dim = tree_dimension(tree)
    dim === nothing || length(point) == dim || throw_dimension_mismatch(length(point), dim)
end

function check_input(tree::NNTree, m::AbstractMatrix)
    check_valid(tree)
    dim = tree_dimension(tree)
    dim === nothing || size(m, 1) == dim || throw_dimension_mismatch(size(m, 1), dim)
end

get_T(::Type{T}) where {T} = typeof(float(zero(T)))

get_tree(tree::NNTree) = tree

include("hyperspheres.jl")
include("hyperrectangles.jl")
include("evaluation.jl")
include("utilities.jl")
include("tree_data.jl")
include("tree_ops.jl")
include("brute_tree.jl")
include("kd_tree.jl")
include("ball_tree.jl")
include("periodic_tree.jl")
include("datafreetree.jl")
include("tree_walk.jl")
include("knn.jl")
include("inrange.jl")

# Type for internal distance calculations (before eval_end)
dist_type_internal(tree::NNTree{V}) where V = get_T(eltype(V))
dist_type_internal(tree::KDTree{V}) where V = typeof(eval_pow(tree.metric, zero(get_T(eltype(V)))))

# Get the "infinity" value in the correct distance space for the tree
dist_typemax(tree::NNTree{V}) where V = typemax(dist_type_internal(tree))

for dim in (2, 3)
    for Tree in (KDTree, BallTree)
        tree = Tree(rand(dim, 10))
        knn(tree, rand(dim), 5)
        inrange(tree, rand(dim), 0.5)
        inrange_pairs(tree, 0.5)
    end
end

end # module
