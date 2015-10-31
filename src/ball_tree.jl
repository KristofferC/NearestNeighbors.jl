
####################################################################
# Ball Tree
####################################################################
immutable BallTree{T <: AbstractFloat, P <: Metric} <: NNTree{T, P}
    data::Matrix{T} # dim x n_points array with floats
    hyper_spheres::Vector{HyperSphere{T}} # Each hyper rectangle bounds its children
    indices::Vector{Int} # Translates from a point index to the actual point in the data
    metric::P
    tree_data::TreeData
    reordered::Bool
end

@with_kw immutable ArrayBuffers{T <: AbstractFloat}
    left::Vector{T}
    right::Vector{T}
    v12::Vector{T}
    zerobuf::Vector{T}
end

function ArrayBuffers(T, ndim)
    ArrayBuffers{T}(zeros(T, ndim), zeros(T, ndim), zeros(T, ndim), zeros(T, ndim))
end

"""
    BallTree(data [, metric = Euclidean(), leafsize = 30]) -> balltree

Creates a `BallTree` from the data using the given `metric` and `leafsize`.
"""
function BallTree{T <: AbstractFloat, P<:Metric}(data::Matrix{T},
                                      metric::P = Euclidean();
                                      leafsize::Int = 30,
                                      reorder::Bool = true)

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

    BallTree(data, hyper_spheres, indices, metric, tree_data, reorder)
end


# Recursive function to build the tree.
# Calculates what dimension has the maximum spread,
# and how many points to send to each side.
# Recurses down to the leaf nodes, and creates the hyper shphere
# and then creates the rest of the spheres on the way back up the stack
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
    if n_points <= tree_data.leaf_size
        if reorder
            reorder_data!(data_reordered, data, index, indices, indices_reordered, tree_data)
        end
        # Create bounding sphere of points in leaf nodeby brute force
        hyper_spheres[index] = create_bsphere(data, metric, indices, low, high)
        return
    end

    mid_idx = find_split(low, tree_data.leaf_size, n_points)
    split_dim = find_largest_spread(data, indices, low, high)

    select_spec!(indices, mid_idx, low, high, data, split_dim)

    build_BallTree(getleft(index), data, data_reordered, hyper_spheres, metric,
                   indices, indices_reordered, low, mid_idx - 1,
                   tree_data, array_buffs, reorder)

    build_BallTree(getright(index), data, data_reordered, hyper_spheres, metric,
                  indices, indices_reordered, mid_idx, high,
                  tree_data, array_buffs, reorder)

    # Finally create hyper sphere from the two children
    hyper_spheres[index]  =  create_bsphere(metric, hyper_spheres[getleft(index)],
                                            hyper_spheres[getright(index)],
                                            array_buffs)
end




function _knn{T <: AbstractFloat}(tree::BallTree{T},
                                  point::AbstractVector{T},
                                  k::Int)
    best_idxs = [-1 for _ in 1:k]
    best_dists = [typemax(T) for _ in 1:k]
    knn_kernel!(tree, point, best_idxs, best_dists)
    return best_idxs, best_dists
end

function knn_kernel!{T <: AbstractFloat}(tree::BallTree{T},
                                  point::AbstractArray{T},
                                  best_idxs ::Vector{Int},
                                  best_dists::Vector{T},
                                  index::Int=1)
    if isleaf(tree.tree_data.n_internal_nodes, index)
        add_points_knn!(best_dists, best_idxs, tree, index, point, true)
        return
    end

    left_sphere = tree.hyper_spheres[getleft(index)]
    right_sphere = tree.hyper_spheres[getright(index)]

    left_dist = evaluate(tree.metric, left_sphere.center, point) - left_sphere.r
    right_dist = evaluate(tree.metric, right_sphere.center, point) - right_sphere.r

    if left_dist <= best_dists[1] || right_dist <= best_dists[1]
        if left_dist < right_dist
            knn_kernel!(tree, point, best_idxs, best_dists, getleft(index))
            if right_dist <=  best_dists[1]
                 knn_kernel!(tree, point, best_idxs, best_dists, getright(index))
             end
        else
            knn_kernel!(tree, point, best_idxs, best_dists, getright(index))
            if left_dist <=  best_dists[1]
                 knn_kernel!(tree, point, best_idxs, best_dists,  getleft(index))
            end
        end
    end
    return
end


function _inrange{T}(tree::BallTree{T},
                    point::AbstractVector{T},
                    radius::T)
    idx_in_ball = Int[]
    ball = HyperSphere(point, radius)
    inrange_kernel!(tree, 1, point, ball, idx_in_ball)
    return idx_in_ball
end

function inrange_kernel!{T <: AbstractFloat}(tree::BallTree{T},
                                            index::Int,
                                            point::Vector{T},
                                            query_ball::HyperSphere{T},
                                            idx_in_ball::Vector{Int})
    @NODE 1
    sphere = tree.hyper_spheres[index]

    if !intersects(tree.metric, sphere, query_ball)
        return
    end

    if isleaf(tree.tree_data.n_internal_nodes, index)
        add_points_inrange!(idx_in_ball, tree, index, point, query_ball.r, true)
        return
    end

    if encloses(tree.metric, sphere, query_ball)
         addall(tree, index, idx_in_ball)
    else
        inrange_kernel!(tree,  getleft(index), point, query_ball, idx_in_ball)
        inrange_kernel!(tree, getright(index), point, query_ball, idx_in_ball)
    end
end
