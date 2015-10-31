abstract NNTree{T, P}
abstract HyperObject{T}

immutable TreeData
    last_node_size::Int
    leaf_size::Int
    n_leafs::Int
    n_internal_nodes::Int
    cross_node::Int
    first_leaf_row::Int
    offset::Int
end

function TreeData(data, leaf_size)
    n_dim, n_p = size(data)
    n_leafs =  ceil(Integer, n_p / leaf_size)
    n_internal_nodes = n_leafs - 1
    l = floor(Integer, log2(n_leafs))
    offset = 2(n_leafs - 2^l) - 1
    cross_node = 2^(l+1)
    last_node_size = n_p % leaf_size
    if last_node_size == 0
        last_node_size = leaf_size
    end

     # This only happens when n_p / leaf_size is a power of 2?
    if cross_node >= n_internal_nodes + n_leafs
        cross_node = div(cross_node, 2)
    end

    TreeData(last_node_size, leaf_size, n_leafs,
    n_internal_nodes, cross_node, l,
    offset)
end

function check_input(tree::NNTree, points::AbstractArray)
    ndim_points = size(points,1)
    ndim_tree = size(tree.data, 1)
    if ndim_points != ndim_tree
        throw(ArgumentError(
            "dimension of input points:$(ndim_points) and tree data:$(ndim_tree) must agree"))
    end
end

"""
    knn(tree::NNTree, points, k [, sortres=false]) -> indices, distances

Performs a lookup of the `k` nearest neigbours to the `points` from the data
in the `tree`.
"""
function knn{T <: AbstractFloat}(tree::NNTree{T}, points::AbstractArray{T}, k::Int, sortres=false)

    check_input(tree, points)

    if k > size(tree.data, 2) || k <= 0
        throw(ArgumentError("k > number of points in tree or ≦ 0"))
    end

    dists = Array(Vector{T}, size(points, 2))
    idxs = Array(Vector{Int}, size(points, 2))
    point = zeros(T, size(points, 1))
    for i in 1:size(points, 2)
        @devec point[:] = points[:, i]
        best_idxs, best_dists = _knn(tree, point, k)
        if sortres
            heap_sort_inplace!(best_dists, best_idxs)
        end
        dists[i] = best_dists
        if tree.reordered
            for j in 1:k
                @inbounds best_idxs[j] = tree.indices[best_idxs[j]]
            end
        end
        idxs[i] = best_idxs
    end
    return do_return(idxs, dists, points)
end

do_return(idxs, dists, ::AbstractVector) = idxs[1], dists[1]
do_return(idxs, dists, ::AbstractMatrix) = idxs, dists


# Conversions for knn if input data is not floating points
function knn{T <: AbstractFloat, P <: Real}(tree::NNTree{T}, points::AbstractArray{P}, k::Int)
  knn(tree, map(T, points), k)
end

"""
    inrange(tree::NNTree, points, radius [, sortres=false]) -> indices

Find all the points in the tree which is closer than `radius` to `points`.
"""
function inrange{T <: AbstractFloat}(tree::NNTree{T},
                                     points::AbstractArray{T},
                                     radius::T,
                                     sortres=false)
    check_input(tree, points)

    if radius < 0
        throw(ArgumentError("the query radius r must be ≧ 0"))
    end

    idxs = Array(Vector{Int}, size(points, 2))
    point = zeros(T, size(points, 1))

    for i in 1:size(points, 2)
        @devec point[:] = points[:, i]
        idx_in_ball = _inrange(tree, point, radius)
        if tree.reordered
            @inbounds for j in 1:length(idx_in_ball)
                idx_in_ball[j] = tree.indices[idx_in_ball[j]]
            end
        end
        if sortres
            sort!(idx_in_ball)
        end
        idxs[i] = idx_in_ball
    end
    return do_return_inrange(idxs, points)
end

do_return_inrange(idxs, ::AbstractVector) = idxs[1]
do_return_inrange(idxs, ::AbstractMatrix) = idxs

function inrange{T <: AbstractFloat, P <: Real}(tree::NNTree{T},
                                                points::AbstractArray{P},
                                                radius,
                                                sortres=false)
    inrange(tree, map(T, points), T(radius), sortres)
end
