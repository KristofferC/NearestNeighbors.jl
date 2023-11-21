# A KDNode stores the information needed in each non leaf node
# to make the needed distance computations
struct KDNode{T}
    split_val::T    # The value the hyper rectangle was split at
    split_dim::Int16  # The dimension the hyper rectangle was split at
end

struct KDTree{V <: AbstractVector,M <: MinkowskiMetric,T} <: NNTree{V,M}
    data::Vector{V}
    hyper_rec::HyperRectangle{T}
    # In the knn and inrange functions we need to mutate an input `HyperRectangle`
    # If we modify the original one, there is a risk that the user does a Ctrl-C and
    # the original `HyperRectangle` is left in a modified state.
    # Therefore, we keep a backup of the original `HyperRectangle` and use that one
    # in the knn and inrange functions.
    backup_hyper_rec::HyperRectangle{T}
    indices::Vector{Int}
    metric::M
    nodes::Vector{KDNode{T}}
    tree_data::TreeData
    reordered::Bool
end


"""
    KDTree(data [, metric = Euclidean(); leafsize = 10, reorder = true]) -> kdtree

Creates a `KDTree` from the data using the given `metric` and `leafsize`.
The `metric` must be a `MinkowskiMetric`.
"""
function KDTree(data::AbstractVector{V},
                metric::M = Euclidean();
                leafsize::Int = 10,
                storedata::Bool = true,
                reorder::Bool = true,
                reorderbuffer::Vector{V} = Vector{V}()) where {V <: AbstractArray, M <: MinkowskiMetric}
    reorder = !isempty(reorderbuffer) || (storedata ? reorder : false)

    tree_data = TreeData(data, leafsize)
    n_d = length(V)
    n_p = length(data)

    indices = collect(1:n_p)
    nodes = Vector{KDNode{eltype(V)}}(undef, tree_data.n_internal_nodes)

    if reorder
        indices_reordered = Vector{Int}(undef, n_p)
        if isempty(reorderbuffer)
            data_reordered = Vector{V}(undef, n_p)
        else
            data_reordered = reorderbuffer
        end
    else
        # Dummy variables
        indices_reordered = Vector{Int}()
        data_reordered = Vector{V}()
    end

    if metric isa Distances.UnionMetrics
        p = parameters(metric)
        if p !== nothing && length(p) != length(V)
            throw(ArgumentError(
                "dimension of input points:$(length(V)) and metric parameter:$(length(p)) must agree"))
        end
    end

    # Create first bounding hyper rectangle that bounds all the input points
    hyper_rec = compute_bbox(data)

    # Call the recursive KDTree builder
    build_KDTree(1, data, data_reordered, hyper_rec, nodes, indices, indices_reordered,
                 1, length(data), tree_data, reorder)
    if reorder
        data = data_reordered
        indices = indices_reordered
    end

    if metric isa Distances.UnionMetrics
        p = parameters(metric)
        if p !== nothing && length(p) != length(V)
            throw(ArgumentError(
                "dimension of input points:$(length(V)) and metric parameter:$(length(p)) must agree"))
        end
    end

    backup_rec = HyperRectangle(copy(hyper_rec.mins), copy(hyper_rec.maxes))
    KDTree(storedata ? data : similar(data, 0), hyper_rec, backup_rec, indices, metric, nodes, tree_data, reorder)
end

 function KDTree(data::AbstractVecOrMat{T},
                 metric::M = Euclidean();
                 leafsize::Int = 10,
                 storedata::Bool = true,
                 reorder::Bool = true,
                 reorderbuffer::Matrix{T} = Matrix{T}(undef, 0, 0)) where {T <: AbstractFloat, M <: MinkowskiMetric}
    dim = size(data, 1)
    npoints = size(data, 2)
    points = copy_svec(T, data, Val(dim))
    if isempty(reorderbuffer)
        reorderbuffer_points = Vector{SVector{dim,T}}()
    else
        reorderbuffer_points = copy_svec(T, reorderbuffer, Val(dim))
    end
    KDTree(points, metric, leafsize = leafsize, storedata = storedata, reorder = reorder,
           reorderbuffer = reorderbuffer_points)
end

function build_KDTree(index::Int,
                      data::AbstractVector{V},
                      data_reordered::Vector{V},
                      hyper_rec::HyperRectangle,
                      nodes::Vector{KDNode{T}},
                      indices::Vector{Int},
                      indices_reordered::Vector{Int},
                      low::Int,
                      high::Int,
                      tree_data::TreeData,
                      reorder::Bool) where {V <: AbstractVector, T}
    n_p = high - low + 1 # Points left
    if n_p <= tree_data.leafsize
        if reorder
            reorder_data!(data_reordered, data, index, indices, indices_reordered, tree_data)
        end
        return
    end

    mid_idx = find_split(low, tree_data.leafsize, n_p)

    split_dim = 1
    max_spread = zero(T)
    # Find dimension and spread where the spread is maximal
    for d in 1:length(V)
        spread = hyper_rec.maxes[d] - hyper_rec.mins[d]
        if spread > max_spread
            max_spread = spread
            split_dim = d
        end
    end

    select_spec!(indices, mid_idx, low, high, data, split_dim)

    split_val = data[indices[mid_idx]][split_dim]

    lo = hyper_rec.mins[split_dim]
    hi = hyper_rec.maxes[split_dim]
    nodes[index] = KDNode{T}(split_val, split_dim)

    # Call the left sub tree with an updated hyper rectangle
    hyper_rec.maxes[split_dim] = split_val
    build_KDTree(getleft(index), data, data_reordered, hyper_rec, nodes,
                  indices, indices_reordered, low, mid_idx - 1, tree_data, reorder)
    hyper_rec.maxes[split_dim] = hi # Restore the hyper rectangle

    # Call the right sub tree with an updated hyper rectangle
    hyper_rec.mins[split_dim] = split_val
    build_KDTree(getright(index), data, data_reordered, hyper_rec, nodes,
                  indices, indices_reordered, mid_idx, high, tree_data, reorder)
    # Restore the hyper rectangle
    hyper_rec.mins[split_dim] = lo
end


function _knn(tree::KDTree,
              point::AbstractVector,
              best_idxs::AbstractVector{Int},
              best_dists::AbstractVector,
              skip::F) where {F}
    init_min = get_min_distance(tree.hyper_rec, point)
    copy!(tree.backup_hyper_rec.maxes, tree.hyper_rec.maxes)
    copy!(tree.backup_hyper_rec.mins, tree.hyper_rec.mins)
    knn_kernel!(tree, 1, point, best_idxs, best_dists, init_min, tree.backup_hyper_rec, skip)
    @simd for i in eachindex(best_dists)
        @inbounds best_dists[i] = eval_end(tree.metric, best_dists[i])
    end
end

function knn_kernel!(tree::KDTree{V},
                        index::Int,
                        point::AbstractVector,
                        best_idxs::AbstractVector{Int},
                        best_dists::AbstractVector,
                        min_dist,
                        hyper_rec::HyperRectangle,
                        skip::F) where {V, F}
    # At a leaf node. Go through all points in node and add those in range
    if isleaf(tree.tree_data.n_internal_nodes, index)
        add_points_knn!(best_dists, best_idxs, tree, index, point, false, skip)
        return
    end

    node = tree.nodes[index]

    split_dim = node.split_dim
    p_dim = point[split_dim]
    split_val = node.split_val
    lo = hyper_rec.mins[split_dim]
    hi = hyper_rec.maxes[split_dim]
    split_diff = p_dim - split_val
    M = tree.metric
    # Point is to the right of the split value
    if split_diff > 0
        close = getright(index)
        far = getleft(index)
        x, y = hyper_rec.mins, hyper_rec.maxes
        vx, vy = lo, hi
        ddiff = max(zero(eltype(V)), p_dim - hi)
    else
        close = getleft(index)
        far = getright(index)
        x, y = hyper_rec.maxes, hyper_rec.mins
        vx, vy = hi, lo
        ddiff = max(zero(eltype(V)), lo - p_dim)
    end
    # Always call closer sub tree
    x[split_dim] = split_val
    knn_kernel!(tree, close, point, best_idxs, best_dists, min_dist, hyper_rec, skip)
    x[split_dim] = vx # Restore

    split_diff_pow = eval_pow(M, split_diff)
    ddiff_pow = eval_pow(M, ddiff)
    diff_tot = eval_diff(M, split_diff_pow, ddiff_pow)
    new_min = eval_reduce(M, min_dist, diff_tot)
    if new_min < best_dists[1]
        y[split_dim] = split_val
        knn_kernel!(tree, far, point, best_idxs, best_dists, new_min, hyper_rec, skip)
        y[split_dim] = vy
    end
    return
end

function _inrange(tree::KDTree,
                  point::AbstractVector,
                  radius::Number,
                  idx_in_ball::Union{Nothing, Vector{Int}} = Int[])
    init_min = get_min_distance(tree.hyper_rec, point)
    copy!(tree.backup_hyper_rec.maxes, tree.hyper_rec.maxes)
    copy!(tree.backup_hyper_rec.mins, tree.hyper_rec.mins)
    return inrange_kernel!(tree, 1, point, eval_op(tree.metric, radius, zero(init_min)), idx_in_ball,
            tree.backup_hyper_rec, init_min)
end

# Explicitly check the distance between leaf node and point while traversing
function inrange_kernel!(tree::KDTree,
                         index::Int,
                         point::AbstractVector,
                         r::Number,
                         idx_in_ball::Union{Nothing, Vector{Int}},
                         hyper_rec::HyperRectangle,
                         min_dist)
    # Point is outside hyper rectangle, skip the whole sub tree
    if min_dist > r
        return 0
    end

    # At a leaf node. Go through all points in node and add those in range
    if isleaf(tree.tree_data.n_internal_nodes, index)
        return add_points_inrange!(idx_in_ball, tree, index, point, r, false)
    end

    node = tree.nodes[index]
    split_val = node.split_val
    split_dim = node.split_dim
    lo = hyper_rec.mins[split_dim]
    hi = hyper_rec.maxes[split_dim]
    p_dim = point[split_dim]
    split_diff = p_dim - split_val
    M = tree.metric

    count = 0

    if split_diff > 0 # Point is to the right of the split value
        close = getright(index)
        far = getleft(index)
        x, y = hyper_rec.mins, hyper_rec.maxes
        vx, vy = lo, hi
        ddiff = max(zero(p_dim - hi), p_dim - hi)
    else # Point is to the left of the split value
        close = getleft(index)
        far = getright(index)
        x, y = hyper_rec.maxes, hyper_rec.mins
        vx, vy = hi, lo
        ddiff = max(zero(lo - p_dim), lo - p_dim)
    end
    # Call closer sub tree
    x[split_dim] = split_val
    count += inrange_kernel!(tree, close, point, r, idx_in_ball, hyper_rec, min_dist)
    x[split_dim] = vx # Restore max

    # TODO: We could potentially also keep track of the max distance
    # between the point and the hyper rectangle and add the whole sub tree
    # in case of the max distance being <= r similarly to the BallTree inrange method.
    # It would be interesting to benchmark this on some different data sets.

    # Call further sub tree with the new min distance
    split_diff_pow = eval_pow(M, split_diff)
    ddiff_pow = eval_pow(M, ddiff)
    diff_tot = eval_diff(M, split_diff_pow, ddiff_pow)
    new_min = eval_reduce(M, min_dist, diff_tot)
    y[split_dim] = split_val
    count += inrange_kernel!(tree, far, point, r, idx_in_ball, hyper_rec, new_min)
    y[split_dim] = vy # Restore min
    return count
end
