# A BallTree (also called Metric tree) is a tree that is created
# from successively splitting points into surrounding hyper spheres
# which radius are determined from the given metric.
# The tree uses the triangle inequality to prune the search space
# when finding the neighbors to a point,
struct BallTree{V <: AbstractVector,N,T,M <: Metric} <: NNTree{V,M}
    data::Vector{V}
    hyper_spheres::Vector{HyperSphere{N,T}} # Each hyper sphere bounds its children
    indices::Vector{Int}                  # Translates from tree index -> point index
    metric::M                             # Metric used for tree
    tree_data::TreeData                   # Some constants needed
    reordered::Bool                       # If the data has been reordered
end


"""
    BallTree(data [, metric = Euclidean(); leafsize = 25, reorder = true]) -> balltree

Creates a `BallTree` from the data using the given `metric` and `leafsize`.
"""
function BallTree(data::AbstractVector{V},
                  metric::Metric = Euclidean();
                  leafsize::Int = 25,
                  reorder::Bool = true,
                  storedata::Bool = true,
                  reorderbuffer::Vector{V} = Vector{V}(),
                  parallel::Bool = Threads.nthreads() > 1) where {V <: AbstractArray}
    reorder = !isempty(reorderbuffer) || (storedata ? reorder : false)

    tree_data = TreeData(data, leafsize)
    n_p = length(data)

    indices = collect(1:n_p)

    # Bottom up creation of hyper spheres so need spheres even for leafs)
    hyper_spheres = Vector{HyperSphere{length(V),eltype(V)}}(undef, tree_data.n_internal_nodes + tree_data.n_leafs)

    indices_reordered = Vector{Int}()
    data_reordered = Vector{V}()

    if reorder
        resize!(indices_reordered, n_p)
        if isempty(reorderbuffer)
            resize!(data_reordered, n_p)
        else
            data_reordered = reorderbuffer
        end
    end

    if metric isa Distances.UnionMetrics
        p = parameters(metric)
        if p !== nothing && length(p) != length(V)
            throw(ArgumentError(
                "dimension of input points:$(length(V)) and metric parameter:$(length(p)) must agree"))
        end
    end

    if n_p > 0
        # Call the recursive BallTree builder
        build_BallTree(1, data, data_reordered, hyper_spheres, metric, indices, indices_reordered,
                       1:length(data), tree_data, reorder, parallel)
    end

    if reorder
       data = data_reordered
       indices = indices_reordered
    end

    BallTree(storedata ? data : similar(data, 0), hyper_spheres, indices, metric, tree_data, reorder)
end

function BallTree(data::AbstractVecOrMat{T},
                  metric::Metric = Euclidean();
                  leafsize::Int = 25,
                  storedata::Bool = true,
                  reorder::Bool = true,
                  reorderbuffer::Matrix{T} = Matrix{T}(undef, 0, 0),
                  parallel::Bool = Threads.nthreads() > 1) where {T <: AbstractFloat}
    dim = size(data, 1)
    points = copy_svec(T, data, Val(dim))
    if isempty(reorderbuffer)
        reorderbuffer_points = Vector{SVector{dim,T}}()
    else
        reorderbuffer_points = copy_svec(T, reorderbuffer, Val(dim))
    end
    BallTree(points, metric; leafsize, storedata, reorder,
            reorderbuffer = reorderbuffer_points, parallel)
end

# Recursive function to build the tree.
function build_BallTree(index::Int,
                        data::AbstractVector{V},
                        data_reordered::Vector{V},
                        hyper_spheres::Vector{HyperSphere{N,T}},
                        metric::Metric,
                        indices::Vector{Int},
                        indices_reordered::Vector{Int},
                        range::UnitRange{Int},
                        tree_data::TreeData,
                        reorder::Bool,
                        parallel::Bool) where {V <: AbstractVector, N, T}

    n_points = length(range) # Points left
    if n_points <= tree_data.leafsize
        if reorder
            reorder_data!(data_reordered, data, index, indices, indices_reordered, tree_data)
        end
        # Create bounding sphere of points in leaf node by brute force
        hyper_spheres[index] = create_bsphere(data, metric, indices, range)
        return
    end

    # Find split such that one of the sub trees has 2^p points
    # and the left sub tree has more points
    mid_idx = find_split(first(range), tree_data.leafsize, n_points)

    # Brute force to find the dimension with the largest spread
    split_dim = find_largest_spread(data, indices, range)

    # Sort the data at the mid_idx boundary using the split_dim
    # to compare
    select_spec!(indices, mid_idx, first(range), last(range), data, split_dim)

    parallel_threshold = 10 * tree_data.leafsize

    if parallel && Threads.nthreads() > 1 && n_points > parallel_threshold
        left_task = Threads.@spawn build_BallTree(getleft(index), data, data_reordered, hyper_spheres, metric,
                                                  indices, indices_reordered, first(range):mid_idx - 1,
                                                  tree_data, reorder, parallel)

        build_BallTree(getright(index), data, data_reordered, hyper_spheres, metric,
                      indices, indices_reordered, mid_idx:last(range),
                      tree_data, reorder, parallel)

        wait(left_task)
    else
        build_BallTree(getleft(index), data, data_reordered, hyper_spheres, metric,
                       indices, indices_reordered, first(range):mid_idx - 1,
                       tree_data, reorder, parallel)

        build_BallTree(getright(index), data, data_reordered, hyper_spheres, metric,
                      indices, indices_reordered, mid_idx:last(range),
                      tree_data, reorder, parallel)
    end

    # Finally create bounding hyper sphere from the two children's hyper spheres
    hyper_spheres[index] = create_bsphere(metric, hyper_spheres[getleft(index)],
                                          hyper_spheres[getright(index)])
end

function _knn(tree::BallTree,
              point::AbstractVector,
              best_idxs::AbstractVector{<:Integer},
              best_dists::AbstractVector,
              skip::F) where {F}
    knn_kernel!(tree, 1, point, best_idxs, best_dists, skip)
    return
end


function knn_kernel!(tree::BallTree{V},
                     index::Int,
                     point::AbstractArray,
                     best_idxs::AbstractVector{<:Integer},
                     best_dists::AbstractVector,
                     skip::F) where {V, F}
    if isleaf(tree.tree_data.n_internal_nodes, index)
        add_points_knn!(best_dists, best_idxs, tree, index, point, true, skip)
        return
    end

    left_sphere = tree.hyper_spheres[getleft(index)]
    right_sphere = tree.hyper_spheres[getright(index)]

    left_dist = distance_to_sphere(tree.metric, point, left_sphere)
    right_dist = distance_to_sphere(tree.metric, point, right_sphere)

    if left_dist <= best_dists[1] || right_dist <= best_dists[1]
        if left_dist < right_dist
            knn_kernel!(tree, getleft(index), point, best_idxs, best_dists, skip)
            if right_dist <= best_dists[1]
                knn_kernel!(tree, getright(index), point, best_idxs, best_dists, skip)
            end
        else
            knn_kernel!(tree, getright(index), point, best_idxs, best_dists, skip)
            if left_dist <= best_dists[1]
                knn_kernel!(tree, getleft(index), point, best_idxs, best_dists, skip)
            end
        end
    end
    return
end

function _inrange(tree::BallTree{V},
                  point::AbstractVector,
                  radius::Number,
                  point_index::Int = 1,
                  callback::Union{Nothing, Function} = nothing) where {V}
    ball = HyperSphere(convert(V, point), convert(eltype(V), radius)) # The "query ball"
    return inrange_kernel!(tree, 1, point, ball, callback, point_index) # Call the recursive range finder
end

function inrange_kernel!(tree::BallTree,
                         index::Int,
                         point::AbstractVector,
                         query_ball::HyperSphere,
                         callback::Union{Nothing, Function},
                         point_index::Int)

    if index > length(tree.hyper_spheres)
        return 0
    end

    sphere = tree.hyper_spheres[index]

    # If the query ball in the bounding sphere for the current sub tree
    # do not intersect we can disrecard the whole subtree
    dist, do_intersect = intersects(tree.metric, sphere, query_ball)
    if !do_intersect
        return 0
    end

    # At a leaf node, check all points in the leaf node
    if isleaf(tree.tree_data.n_internal_nodes, index)
        r = tree.metric isa MinkowskiMetric ? eval_pow(tree.metric, query_ball.r) : query_ball.r
        return add_points_inrange!(tree, index, point, r, callback, point_index)
    end

    count = 0

    # The query ball encloses the sub tree bounding sphere. Add all points in the
    # sub tree without checking the distance function.
    if encloses_fast(dist, tree.metric, sphere, query_ball)
        count += addall(tree, index, callback, point_index)
    else
        # Recursively call the left and right sub tree.
        count += inrange_kernel!(tree,  getleft(index), point, query_ball, callback, point_index)
        count += inrange_kernel!(tree, getright(index), point, query_ball, callback, point_index)
    end
    return count
end
