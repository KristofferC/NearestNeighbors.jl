# A BallTree (also called Metric tree) is a tree that is created
# from successively splitting points into surrounding hyper spheres
# which radius are determined from the given metric.
# The tree uses the triangle inequality to prune the search space
# when finding the neighbors to a point,
immutable BallTree{T <: AbstractFloat, M <: Metric} <: NNTree{T, M}
    data::Matrix{T}                       # dim x n_points array with floats
    hyper_spheres::Vector{HyperSphere{T}} # Each hyper sphere bounds its children
    indices::Vector{Int}                  # Translates from tree index -> point index
    metric::M                             # Metric used for tree
    tree_data::TreeData                   # Some constants needed
    reordered::Bool                       # If the data has been reordered
end

# When we create the bounding spheres we need some temporary arrays.
# We create a type to hold them to not allocate these arrays at every
# function call and to reduce the number of parameters in the tree builder.
immutable ArrayBuffers{T <: AbstractFloat}
    left::Vector{T}
    right::Vector{T}
    v12::Vector{T}
    zerobuf::Vector{T}
end

function ArrayBuffers(T, ndim)
    ArrayBuffers{T}(zeros(T, ndim), zeros(T, ndim), zeros(T, ndim), zeros(T, ndim))
end

"""
    BallTree(data [, metric = Euclidean(), leafsize = 10]) -> balltree

Creates a `BallTree` from the data using the given `metric` and `leafsize`.
"""
function BallTree{T <: AbstractFloat, M<:Metric}(data::Matrix{T},
                                                 metric::M = Euclidean();
                                                 leafsize::Int = 10,
                                                 reorder::Bool = true,
                                                 storedata::Bool = true)

    reorder = storedata ? reorder : false

    tree_data = TreeData(data, leafsize)
    n_d = size(data, 1)
    n_p = size(data, 2)
    array_buffs = ArrayBuffers(T, size(data, 1))
    indices = collect(1:n_p)

    # Bottom up creation of hyper spheres so need spheres even for leafs
    hyper_spheres = Array(HyperSphere{T}, tree_data.n_internal_nodes + tree_data.n_leafs)

    if reorder
       indices_reordered = Vector{Int}(n_p)
       data_reordered = Matrix{T}(n_d, n_p)
     else
       # Dummy variables
       indices_reordered = Vector{Int}(0)
       data_reordered = Matrix{T}(0, 0)
     end

    # Call the recursive BallTree builder
    build_BallTree(1, data, data_reordered, hyper_spheres, metric, indices, indices_reordered,
                   1,  size(data,2), tree_data, array_buffs, reorder)

    if reorder
       data = data_reordered
       indices = indices_reordered
    end

    BallTree(storedata ? data : similar(data,0,0), hyper_spheres, indices, metric, tree_data, reorder)
end

# Recursive function to build the tree.
function build_BallTree{T <: AbstractFloat}(index::Int,
                                            data::Matrix{T},
                                            data_reordered::Matrix{T},
                                            hyper_spheres::Vector{HyperSphere{T}},
                                            metric::Metric,
                                            indices::Vector{Int},
                                            indices_reordered::Vector{Int},
                                            low::Int,
                                            high::Int,
                                            tree_data::TreeData,
                                            array_buffs::ArrayBuffers{T},
                                            reorder::Bool)

    n_points = high - low + 1 # Points left
    if n_points <= tree_data.leafsize
        if reorder
            reorder_data!(data_reordered, data, index, indices, indices_reordered, tree_data)
        end
        # Create bounding sphere of points in leaf node by brute force
        hyper_spheres[index] = create_bsphere(data, metric, indices, low, high)
        return
    end

    # Find split such that one of the sub trees has 2^p points
    # and the left sub tree has more points
    mid_idx = find_split(low, tree_data.leafsize, n_points)

    # Brute force to find the dimension with the largest spread
    split_dim = find_largest_spread(data, indices, low, high)

    # Sort the data at the mid_idx boundary using the split_dim
    # to compare
    select_spec!(indices, mid_idx, low, high, data, split_dim)

    build_BallTree(getleft(index), data, data_reordered, hyper_spheres, metric,
                   indices, indices_reordered, low, mid_idx - 1,
                   tree_data, array_buffs, reorder)

    build_BallTree(getright(index), data, data_reordered, hyper_spheres, metric,
                  indices, indices_reordered, mid_idx, high,
                  tree_data, array_buffs, reorder)

    # Finally create bounding hyper sphere from the two children's hyper spheres
    hyper_spheres[index]  =  create_bsphere(metric, hyper_spheres[getleft(index)],
                                            hyper_spheres[getright(index)],
                                            array_buffs)
end

function _knn{T}(tree::BallTree{T},
                 point::AbstractVector{T},
                 k::Int)
    best_idxs = [-1 for _ in 1:k]
    best_dists = [typemax(T) for _ in 1:k]
    knn_kernel!(tree, 1, point, best_idxs, best_dists)
    return best_idxs, best_dists
end

function knn_kernel!{T}(tree::BallTree{T},
                        index::Int,
                        point::AbstractArray{T},
                        best_idxs ::Vector{Int},
                        best_dists::Vector{T})
    @NODE 1
    if isleaf(tree.tree_data.n_internal_nodes, index)
        add_points_knn!(best_dists, best_idxs, tree, index, point, true)
        return
    end

    left_sphere = tree.hyper_spheres[getleft(index)]
    right_sphere = tree.hyper_spheres[getright(index)]

    left_dist = max(zero(T), evaluate(tree.metric, point, left_sphere.center) - left_sphere.r)
    right_dist = max(zero(T), evaluate(tree.metric, point, right_sphere.center) - right_sphere.r)

    if left_dist <= best_dists[1] || right_dist <= best_dists[1]
        if left_dist < right_dist
            knn_kernel!(tree, getleft(index), point, best_idxs, best_dists)
            if right_dist <=  best_dists[1]
                 knn_kernel!(tree, getright(index), point, best_idxs, best_dists)
             end
        else
            knn_kernel!(tree, getright(index), point, best_idxs, best_dists)
            if left_dist <=  best_dists[1]
                 knn_kernel!(tree, getleft(index), point, best_idxs, best_dists)
            end
        end
    end
    return
end

function _inrange{T}(tree::BallTree{T},
                     point::AbstractVector{T},
                     radius::Number)
    idx_in_ball = Int[]                                 # List to hold the indices in range
    ball = HyperSphere(point, radius)                   # The "query ball"
    inrange_kernel!(tree, 1, point, ball, idx_in_ball)  # Call the recursive range finder
    return idx_in_ball
end

function inrange_kernel!{T}(tree::BallTree{T},
                            index::Int,
                            point::Vector{T},
                            query_ball::HyperSphere{T},
                            idx_in_ball::Vector{Int})
    @NODE 1
    sphere = tree.hyper_spheres[index]

    # If the query ball in the bounding sphere for the current sub tree
    # do not intersect we can disrecard the whole subtree
    if !intersects(tree.metric, sphere, query_ball)
        return
    end

    # At a leaf node, check all points in the leaf node
    if isleaf(tree.tree_data.n_internal_nodes, index)
        add_points_inrange!(idx_in_ball, tree, index, point, query_ball.r, true)
        return
    end

    # The query ball encloses the sub tree bounding sphere. Add all points in the
    # sub tree without checking the distance function.
    if encloses(tree.metric, sphere, query_ball)
         addall(tree, index, idx_in_ball)
    else
        # Recursively call the left and right sub tree.
        inrange_kernel!(tree,  getleft(index), point, query_ball, idx_in_ball)
        inrange_kernel!(tree, getright(index), point, query_ball, idx_in_ball)
    end
end
