immutable KDNode{T}
    lo::T
    hi::T
    split_val::T
    split_dim::Int
end

immutable KDTree{T <: AbstractFloat, P <: MinkowskiMetric} <: NNTree{T, P}
    data::Matrix{T} # dim x n_p array with floats
    hyper_rec::HyperRectangle{T}
    indices::Vector{Int}
    metric::P
    nodes::Vector{KDNode{T}}
    tree_data::TreeData
    reordered::Bool
end

"""
    KDTree(data [, metric = Euclidean(), leafsize = 30]) -> kdtree

Creates a `BallTree` from the data using the given `metric` and `leafsize`.
The `metric` must be a `MinkowskiMetric`.
"""
function KDTree{T <: AbstractFloat, P <: MinkowskiMetric}(data::Matrix{T},
                                                          metric::P = Euclidean();
                                                          leafsize::Int = 10,
                                                          reorder::Bool = true)

    tree_data = TreeData(data, leafsize)
    n_d = size(data, 1)
    n_p = size(data, 2)

    indices = collect(1:n_p)
    split_vals = Array(T, tree_data.n_internal_nodes)
    split_dims = Array(Int, tree_data.n_internal_nodes)
    nodes = Array(KDNode{T}, tree_data.n_internal_nodes)

    if reorder
       indices_reordered = Vector{Int}(n_p)
       data_reordered = Matrix{T}(n_d, n_p)
     else
       # Dummy variables
       indices_reordered = Vector{Int}(0)
       data_reordered = Matrix{T}(0, 0)
     end

    # Create first bounding hyper rectangle
    hyper_rec = compute_bbox(data)

    # Call the recursive KDTree builder
    build_KDTree(1, data, data_reordered, hyper_rec, nodes, indices, indices_reordered,
                 1, size(data,2), tree_data, reorder)
    if reorder
        data = data_reordered
        indices = indices_reordered
    end

    KDTree(data, hyper_rec, indices, metric, nodes, tree_data, reorder)
end


# Recursive function to build the tree.
# Calculates what dimension has the maximum spread,
# and how many points to send to each side.
# Splits the hyper cubes and calls recursively
# with the new cubes and node indices.
function build_KDTree{T <: AbstractFloat}(index::Int,
                                          data::Matrix{T},
                                          data_reordered::Matrix{T},
                                          hyper_rec::HyperRectangle{T},
                                          nodes::Vector{KDNode{T}},
                                          indices::Vector{Int},
                                          indices_reordered::Vector{Int},
                                          low::Int,
                                          high::Int,
                                          tree_data::TreeData,
                                          reorder::Bool)
    n_p = high - low + 1 # Points left
    if n_p <= tree_data.leaf_size
        if reorder
            reorder_data!(data_reordered, data, index, indices, indices_reordered, tree_data)
        end
        return
    end

    # The number of leafs left, ceil to count a partially filled bucket node as 1.
    mid_idx = find_split(low, tree_data.leaf_size, n_p)

    n_d = size(data, 1)
    split_dim = 1
    max_spread = zero(T)
    # Find dimension and and spread where the spread is maximal
    for d in 1:n_d
        spread = hyper_rec.maxes[d] - hyper_rec.mins[d]
        if spread > max_spread
            max_spread = spread
            split_dim = d
        end
    end

    select_spec!(indices, mid_idx, low, high, data, split_dim)

    split_val = data[split_dim, indices[mid_idx]]

    lo = hyper_rec.mins[split_dim]
    hi = hyper_rec.maxes[split_dim]

    nodes[index] = KDNode{T}(lo, hi, split_val, split_dim)

    hyper_rec.maxes[split_dim] = split_val
    build_KDTree(getleft(index), data, data_reordered, hyper_rec, nodes,
                  indices, indices_reordered, low, mid_idx - 1 , tree_data, reorder)
    hyper_rec.maxes[split_dim] = hi

    hyper_rec.mins[split_dim] = split_val
    build_KDTree(getright(index), data, data_reordered, hyper_rec, nodes,
                  indices, indices_reordered, mid_idx, high, tree_data, reorder)
    hyper_rec.mins[split_dim] = lo
end


####################################################################
# Query functions
####################################################################
function _knn{T <: AbstractFloat}(tree::KDTree{T},
                                  point::AbstractVector{T},
                                  k::Int)
    best_idxs = [-1 for _ in 1:k]
    best_dists = [typemax(T) for _ in 1:k]
    init_min = get_min_distance(tree.hyper_rec, point)
    knn_kernel!(tree, 1, point, best_idxs, best_dists, init_min)
    @simd for i in eachindex(best_dists)
        @inbounds best_dists[i] = eval_end(tree.metric, best_dists[i])
    end
    return best_idxs, best_dists
end

function knn_kernel!{T <: AbstractFloat}(tree::KDTree{T},
                                  index::Int,
                                  point::Vector{T},
                                  best_idxs ::Vector{Int},
                                  best_dists::Vector{T},
                                  min_dist::T)
    @NODE 1
    # At a leaf node. Go through all points in node and add those in range
    if isleaf(tree.tree_data.n_internal_nodes, index)
        add_points_knn!(best_dists, best_idxs, tree, index, point, false)
        return
    end

    node = tree.nodes[index]
    p_dim = point[node.split_dim]
    split_val = node.split_val
    lo = node.lo
    hi = node.hi
    split_diff = p_dim - split_val
    M = tree.metric

    # Point is to the right of the split value
    if split_diff > 0
        # Call closer one
        knn_kernel!(tree, getright(index), point, best_idxs, best_dists, min_dist)

        # Only call further one if it is close enough
        ddiff = inv_eval_end(M, p_dim - split_val)
        new_min_left = min_dist + ddiff - inv_eval_end(M, max(zero(T), p_dim - hi))
        if new_min_left < best_dists[1]
             knn_kernel!(tree, getleft(index), point, best_idxs, best_dists, new_min_left)
        end
    else # Point is to the the left of the split value
        # Call closer one
        knn_kernel!(tree, getleft(index), point, best_idxs, best_dists, min_dist)

        # Only call further one if it is close enough
        ddiff = inv_eval_end(M, split_val - p_dim)
        new_min_right = min_dist + ddiff - inv_eval_end(M, max(zero(T), lo - p_dim))
        if new_min_right < best_dists[1]
            knn_kernel!(tree, getright(index), point, best_idxs, best_dists, new_min_right)
        end
    end
    return
end

function _inrange{T, P <: MinkowskiMetric}(tree::KDTree{T, P},
                    point::AbstractVector{T},
                    radius::T)
    idx_in_ball = Int[]
    init_min = get_min_distance(tree.hyper_rec, point)
    inrange_kernel!(tree, 1, point, inv_eval_end(tree.metric, radius), idx_in_ball,
                   init_min)
    return idx_in_ball
end

# Explicitly check the distance between leaf node and point while traversing
function inrange_kernel!{T <: AbstractFloat}(tree::KDTree{T},
                                           index::Int,
                                           point::Vector{T},
                                           r::T,
                                           idx_in_ball::Vector{Int},
                                           min_dist::T)
    @NODE 1
    if min_dist > r # Point is outside hyper rectangle, skip the whole sub tree
        return
    end

   # Todo:
   # if max_dist < r # Point is inside bbox, add whole sub tree
   #     addall(tree, index, idx_in_ball)
   #     return
   # end

    # At a leaf node. Go through all points in node and add those in range
    if isleaf(tree.tree_data.n_internal_nodes, index)
        add_points_inrange!(idx_in_ball, tree, index, point, r, false)
        return
    end

    node = tree.nodes[index]
    p_dim = point[node.split_dim]
    split_val = node.split_val
    lo = node.lo
    hi = node.hi
    split_diff = p_dim - split_val
    M = tree.metric

    # Point is to the right of the split value
    if split_diff > 0
        # The closer rectangle has the same min distance but potentially smaller max distance
       # if hi - p_dim > split_diff
       #     new_max_right = max_dist
       # else
       #     new_max_right = max_dist - inv_eval_end(M, p_dim - lo) + inv_eval_end(M, p_dim - split_val)
       # end
        inrange_kernel!(tree, getright(index), point, r, idx_in_ball, min_dist)

        # The further away rectangle has the same max distance butpotentially  larger min distance:
        ddiff = inv_eval_end(M, p_dim - split_val)
        new_min_left = min_dist + ddiff - inv_eval_end(M, max(zero(T), p_dim - hi))
        inrange_kernel!(tree, getleft(index), point, r, idx_in_ball, new_min_left)
    else # Point is the the left of the split value
        # if p_dim - lo > -split_diff
        #    new_max_left = max_dist
        #else
        #    new_max_left = max_dist - inv_eval_end(M, hi - p_dim) + inv_eval_end(M, split_val - p_dim)
        #end
        # The closer rectangle has the same min distance but potentially smaller max distance
        inrange_kernel!(tree, getleft(index), point, r, idx_in_ball, min_dist)

        # The further away rectangle has the same max distance but potentially larger min distance:
        ddiff = inv_eval_end(M, split_val - p_dim)
        new_min_right = min_dist + ddiff - inv_eval_end(M, max(zero(T), lo - p_dim))
        inrange_kernel!(tree, getright(index), point, r, idx_in_ball, new_min_right)
    end
end
