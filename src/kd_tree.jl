struct KDTree{V <: AbstractVector, M <: MinkowskiMetric, T, TH} <: NNTree{V,M}
    data::Vector{V}
    hyper_rec::HyperRectangle{TH}
    indices::Vector{Int}
    metric::M
    split_vals::Vector{T}
    split_dims::Vector{UInt16}
    tree_data::TreeData
    reordered::Bool
end


"""
    KDTree(data [, metric = Euclidean(); leafsize = 25, reorder = true]) -> kdtree

Creates a `KDTree` from the data using the given `metric` and `leafsize`.

# Arguments
- `data`: Point data as a matrix of size `nd × np` or vector of vectors
- `metric`: Distance metric to use (must be a `MinkowskiMetric` like `Euclidean`, `Chebyshev`, `Minkowski`, or `Cityblock`). Default: `Euclidean()`
- `leafsize`: Number of points at which to stop splitting the tree. Default: `25`
- `reorder`: If `true`, reorder data to improve cache locality. Default: `true`

# Returns
- `kdtree`: A `KDTree` instance

KDTree works best for low-dimensional data with axis-aligned metrics.
"""
function KDTree(data::AbstractVector{V},
                metric::M = Euclidean();
                leafsize::Int = 25,
                storedata::Bool = true,
                reorder::Bool = true,
                reorderbuffer::Vector{V} = Vector{V}()) where {V <: AbstractArray, M <: MinkowskiMetric}
    reorder = !isempty(reorderbuffer) || (storedata ? reorder : false)

    tree_data = TreeData(data, leafsize)
    n_p = length(data)

    indices = collect(1:n_p)
    split_vals = Vector{eltype(V)}(undef, tree_data.n_internal_nodes)
    split_dims = Vector{UInt16}(undef, tree_data.n_internal_nodes)

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
    build_KDTree(1, data, data_reordered, hyper_rec, split_vals, split_dims, indices, indices_reordered,
                 1:length(data), tree_data, reorder)
    if reorder
        data = data_reordered
        indices = indices_reordered
    end

    KDTree(storedata ? data : similar(data, 0), hyper_rec, indices, metric, split_vals, split_dims, tree_data, reorder)
end

function KDTree(data::AbstractVecOrMat{T},
                 metric::M = Euclidean();
                 leafsize::Int = 25,
                 storedata::Bool = true,
                 reorder::Bool = true,
                 reorderbuffer::Matrix{T} = Matrix{T}(undef, 0, 0)) where {T <: AbstractFloat, M <: MinkowskiMetric}
    dim = size(data, 1)
    points = copy_svec(T, data, Val(dim))
    if isempty(reorderbuffer)
        reorderbuffer_points = Vector{SVector{dim,T}}()
    else
        reorderbuffer_points = copy_svec(T, reorderbuffer, Val(dim))
    end
    KDTree(points, metric; leafsize, storedata, reorder,
           reorderbuffer = reorderbuffer_points)
end

function build_KDTree(index::Int,
                      data::AbstractVector{V},
                      data_reordered::Vector{V},
                      hyper_rec::HyperRectangle,
                      split_vals::Vector{T},
                      split_dims::Vector{UInt16},
                      indices::Vector{Int},
                      indices_reordered::Vector{Int},
                      range,
                      tree_data::TreeData,
                      reorder::Bool) where {V <: AbstractVector, T}
    n_p = length(range) # Points left
    if n_p <= tree_data.leafsize
        if reorder
            reorder_data!(data_reordered, data, index, indices, indices_reordered, tree_data)
        end
        return
    end

    mid_idx = find_split(first(range), tree_data.leafsize, n_p)

    split_dim = argmax(d -> hyper_rec.maxes[d] - hyper_rec.mins[d], 1:length(V))

    select_spec!(indices, mid_idx, first(range), last(range), data, split_dim)

    split_val = data[indices[mid_idx]][split_dim]

    split_vals[index] = split_val
    split_dims[index] = split_dim

    # Call the left sub tree with an updated hyper rectangle
    new_maxes = @inbounds setindex(hyper_rec.maxes, split_val, split_dim)
    hyper_rec_left = HyperRectangle(hyper_rec.mins, new_maxes)
    build_KDTree(getleft(index), data, data_reordered, hyper_rec_left, split_vals, split_dims,
                  indices, indices_reordered, first(range):mid_idx - 1, tree_data, reorder)

    # Call the right sub tree with an updated hyper rectangle
    new_mins = @inbounds setindex(hyper_rec.mins, split_val, split_dim)
    hyper_rec_right = HyperRectangle(new_mins, hyper_rec.maxes)
    build_KDTree(getright(index), data, data_reordered, hyper_rec_right, split_vals, split_dims,
                  indices, indices_reordered, mid_idx:last(range), tree_data, reorder)
end


function _knn(tree::KDTree,
              point::AbstractVector,
              best_idxs::AbstractVector{<:Integer},
              best_dists::AbstractVector,
              skip::F) where {F}
    init_min = get_min_distance_no_end(tree.metric, tree.hyper_rec, point)
    knn_kernel!(tree, 1, point, best_idxs, best_dists, init_min, tree.hyper_rec, skip)
    @simd for i in eachindex(best_dists)
        @inbounds best_dists[i] = eval_end(tree.metric, best_dists[i])
    end
end

function knn_kernel!(tree::KDTree{V},
                        index::Int,
                        point::AbstractVector,
                        best_idxs::AbstractVector{<:Integer},
                        best_dists::AbstractVector,
                        min_dist,
                        hyper_rec::HyperRectangle,
                        skip::F) where {V, F}
    # At a leaf node. Go through all points in node and add those in range
    if isleaf(tree.tree_data.n_internal_nodes, index)
        add_points_knn!(best_dists, best_idxs, tree, index, point, false, skip)
        return
    end

    split_dim = tree.split_dims[index]
    p_dim = point[split_dim]
    split_val = tree.split_vals[index]
    split_diff = p_dim - split_val
    M = tree.metric
    # Point is to the right of the split value
    if split_diff > 0
        close = getright(index)
        far = getleft(index)
        hyper_rec_far = HyperRectangle(hyper_rec.mins, @inbounds setindex(hyper_rec.maxes, split_val, split_dim))
        hyper_rec_close = HyperRectangle(@inbounds(setindex(hyper_rec.mins, split_val, split_dim)), hyper_rec.maxes)
    else
        close = getleft(index)
        far = getright(index)
        hyper_rec_far = HyperRectangle(@inbounds(setindex(hyper_rec.mins, split_val, split_dim)), hyper_rec.maxes)
        hyper_rec_close = HyperRectangle(hyper_rec.mins, @inbounds setindex(hyper_rec.maxes, split_val, split_dim))
    end
    # Always call closer sub tree
    knn_kernel!(tree, close, point, best_idxs, best_dists, min_dist, hyper_rec_close, skip)

    if M isa Chebyshev
        new_min = get_min_distance_no_end(M, hyper_rec_far, point)
    else
        new_min = update_new_min(M, min_dist, hyper_rec, point, split_dim, split_val)
    end

    if new_min < best_dists[1]
        knn_kernel!(tree, far, point, best_idxs, best_dists, new_min, hyper_rec_far, skip)
    end
    return
end

function _inrange(tree::KDTree,
                  point::AbstractVector,
                  radius::Number,
                  idx_in_ball::Union{Nothing, Vector{<:Integer}} = Int[])
    init_min = get_min_distance_no_end(tree.metric, tree.hyper_rec, point)
    return inrange_kernel!(tree, 1, point, eval_pow(tree.metric, radius), idx_in_ball,
            tree.hyper_rec, init_min)
end

# Explicitly check the distance between leaf node and point while traversing
function inrange_kernel!(tree::KDTree,
                         index::Int,
                         point::AbstractVector,
                         r::Number,
                         idx_in_ball::Union{Nothing, Vector{<:Integer}},
                         hyper_rec::HyperRectangle,
                         min_dist)
    # Point is outside hyper rectangle, skip the whole sub tree
    if min_dist > r
        return 0
    end

    # At a leaf node. Go through all points in node and add those in range
    if isleaf(tree.tree_data.n_internal_nodes, index)
        return add_points_inrange!(idx_in_ball, tree, index, point, r)
    end

    split_val = tree.split_vals[index]
    split_dim = tree.split_dims[index]
    p_dim = point[split_dim]
    split_diff = p_dim - split_val
    M = tree.metric

    count = 0

    if split_diff > 0 # Point is to the right of the split value
        close = getright(index)
        far = getleft(index)
        hyper_rec_far = HyperRectangle(hyper_rec.mins, @inbounds setindex(hyper_rec.maxes, split_val, split_dim))
        hyper_rec_close = HyperRectangle(@inbounds(setindex(hyper_rec.mins, split_val, split_dim)), hyper_rec.maxes)
    else # Point is to the left of the split value
        close = getleft(index)
        far = getright(index)
        hyper_rec_far = HyperRectangle(@inbounds(setindex(hyper_rec.mins, split_val, split_dim)), hyper_rec.maxes)
        hyper_rec_close = HyperRectangle(hyper_rec.mins, @inbounds setindex(hyper_rec.maxes, split_val, split_dim))
    end
    # Call closer sub tree
    count += inrange_kernel!(tree, close, point, r, idx_in_ball, hyper_rec_close, min_dist)

    # TODO: We could potentially also keep track of the max distance
    # between the point and the hyper rectangle and add the whole sub tree
    # in case of the max distance being <= r similarly to the BallTree inrange method.
    # It would be interesting to benchmark this on some different data sets.

    # Call further sub tree with the new min distance
    if M isa Chebyshev
        new_min = get_min_distance_no_end(M, hyper_rec_far, point)
    else
        new_min = update_new_min(M, min_dist, hyper_rec, point, split_dim, split_val)
    end

    count += inrange_kernel!(tree, far, point, r, idx_in_ball, hyper_rec_far, new_min)
    return count
end
