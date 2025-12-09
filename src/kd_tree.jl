struct KDTree{V <: AbstractVector, M <: MinkowskiMetric, T, TH} <: NNTree{V,M}
    data::Vector{V}
    hyper_rec::HyperRectangle{TH}
    indices::Vector{Int}
    metric::M
    split_vals::Vector{T}
    split_dims::Vector{UInt16}
    hyper_rects::Vector{HyperRectangle{TH}}
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
                reorderbuffer::Vector{V} = Vector{V}(),
                parallel::Bool = Threads.nthreads() > 1) where {V <: AbstractArray, M <: MinkowskiMetric}
    reorder = !isempty(reorderbuffer) || (storedata ? reorder : false)

    # Reject data containing NaNs early to avoid undefined behaviour later on.
    check_for_nan(data)

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
    hyper_rects = Vector{typeof(hyper_rec)}(undef, tree_data.n_internal_nodes)

    # Call the recursive KDTree builder
    build_KDTree(1, data, data_reordered, hyper_rec, split_vals, split_dims, hyper_rects, indices, indices_reordered,
                 1:length(data), tree_data, reorder, parallel)
    if reorder
        data = data_reordered
        indices = indices_reordered
    end

    KDTree(storedata ? data : similar(data, 0), hyper_rec, indices, metric, split_vals, split_dims, hyper_rects, tree_data, reorder)
end

function KDTree(data::AbstractVecOrMat{T},
                 metric::M = Euclidean();
                 leafsize::Int = 25,
                 storedata::Bool = true,
                 reorder::Bool = true,
                 reorderbuffer::Matrix{T} = Matrix{T}(undef, 0, 0),
                 parallel::Bool = Threads.nthreads() > 1) where {T <: AbstractFloat, M <: MinkowskiMetric}
    dim = size(data, 1)
    points = copy_svec(T, data, Val(dim))
    if isempty(reorderbuffer)
        reorderbuffer_points = Vector{SVector{dim,T}}()
    else
        reorderbuffer_points = copy_svec(T, reorderbuffer, Val(dim))
    end
    KDTree(points, metric; leafsize, storedata, reorder,
           reorderbuffer = reorderbuffer_points, parallel)
end

function build_KDTree(index::Int,
                      data::AbstractVector{V},
                      data_reordered::Vector{V},
                      hyper_rec::HyperRectangle,
                      split_vals::Vector{T},
                      split_dims::Vector{UInt16},
                      hyper_rects::Vector{<:HyperRectangle},
                      indices::Vector{Int},
                      indices_reordered::Vector{Int},
                      range,
                      tree_data::TreeData,
                      reorder::Bool,
                      parallel::Bool) where {V <: AbstractVector, T}
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
    hyper_rects[index] = hyper_rec

    hyper_rec_left, hyper_rec_right = split_hyperrectangle(hyper_rec, split_dim, split_val)

    parallel_threshold = 10 * tree_data.leafsize

    if parallel && Threads.nthreads() > 1 && n_p > parallel_threshold
        left_task = Threads.@spawn build_KDTree(getleft(index), data, data_reordered, hyper_rec_left, split_vals, split_dims,
                                                hyper_rects, indices, indices_reordered, first(range):mid_idx - 1, tree_data, reorder, parallel)

        build_KDTree(getright(index), data, data_reordered, hyper_rec_right, split_vals, split_dims,
                    hyper_rects, indices, indices_reordered, mid_idx:last(range), tree_data, reorder, parallel)

        fetch(left_task)
    else
        build_KDTree(getleft(index), data, data_reordered, hyper_rec_left, split_vals, split_dims,
                    hyper_rects, indices, indices_reordered, first(range):mid_idx - 1, tree_data, reorder, parallel)

        build_KDTree(getright(index), data, data_reordered, hyper_rec_right, split_vals, split_dims,
                    hyper_rects, indices, indices_reordered, mid_idx:last(range), tree_data, reorder, parallel)
    end
end


function _knn(tree::KDTree,
              point::AbstractVector,
              best_idxs::Union{Integer, AbstractVector{<:Integer}},
              best_dists::Union{Number, AbstractVector},
              best_dists_final::Union{Nothing, AbstractVector},
              skip::F,
              self_idx::Int) where {F}
    init_min = get_min_distance_no_end(tree.metric, tree.hyper_rec, point)
    best_idxs, best_dists = knn_kernel!(tree, 1, point, best_idxs, best_dists, init_min, tree.hyper_rec, skip, nothing, self_idx)
    best_dists isa Number && return best_idxs, eval_end(tree.metric, best_dists)
    @simd for i in eachindex(best_dists)
        @inbounds best_dists_final[i] = eval_end(tree.metric, best_dists[i])
    end
    return best_idxs, best_dists_final
end

function knn_kernel!(tree::KDTree{V},
                        index::Int,
                        point::AbstractVector,
                        best_idxs::Union{Integer, AbstractVector{<:Integer}},
                        best_dists::Union{Number, AbstractVector},
                        min_dist,
                        hyper_rec::HyperRectangle,
                        skip::F,
                        dedup::MaybeBitSet,
                        self_idx::Int) where {V, F}
    # At a leaf node. Go through all points in node and add those in range
    if isleaf(tree.tree_data.n_internal_nodes, index)
        return add_points_knn!(best_dists, best_idxs, tree, index, point, false, skip, dedup, self_idx)
    end

    split_dim = tree.split_dims[index]
    p_dim = point[split_dim]
    split_val = tree.split_vals[index]
    split_diff = p_dim - split_val
    M = tree.metric
    left_region, right_region = split_hyperrectangle(hyper_rec, split_dim, split_val)
    left_idx = getleft(index)
    right_idx = getright(index)
    # Point is to the right of the split value
    if split_diff > 0
        close = right_idx
        far = left_idx
        hyper_rec_far = left_region
        hyper_rec_close = right_region
    else
        close = left_idx
        far = right_idx
        hyper_rec_far = right_region
        hyper_rec_close = left_region
    end
    # Always call closer sub tree
    best_idxs, best_dists = knn_kernel!(tree, close, point, best_idxs, best_dists, min_dist, hyper_rec_close, skip, dedup, self_idx)

    if M isa Chebyshev
        new_min = get_min_distance_no_end(M, hyper_rec_far, point)
    else
        new_min = update_new_min(M, min_dist, hyper_rec, p_dim, split_dim, split_val)
    end

    best_dist_1 = first(best_dists)
    if new_min < best_dist_1
        best_idxs, best_dists = knn_kernel!(tree, far, point, best_idxs, best_dists, new_min, hyper_rec_far, skip, dedup, self_idx)
    end
    return best_idxs, best_dists
end

function _inrange(
        tree::KDTree,
        point::AbstractVector,
        radius::Number,
        idx_in_ball::Union{Nothing, Vector{<:Integer}},
        skip::F,
        self_idx::Int) where {F}
    init_min = get_min_distance_no_end(tree.metric, tree.hyper_rec, point)
    init_max_contribs = get_max_distance_contributions(tree.metric, tree.hyper_rec, point)
    init_max = tree.metric isa Chebyshev ? maximum(init_max_contribs) : sum(init_max_contribs)
    return inrange_kernel!(
        tree, 1, point, eval_pow(tree.metric, radius), idx_in_ball,
        tree.hyper_rec, init_min, init_max_contribs, init_max, skip, nothing, self_idx)
end


# Explicitly check the distance between leaf node and point while traversing
function inrange_kernel!(
        tree::KDTree,
        index::Int,
        point::AbstractVector,
        r::Number,
        idx_in_ball::Union{Nothing, Vector{<:Integer}},
        hyper_rec::HyperRectangle,
        min_dist,
        max_dist_contribs::SVector,
        max_dist,
        skip::F,
        dedup::MaybeBitSet,
        self_idx::Int) where {F}
    # Point is outside hyper rectangle, skip the whole sub tree
    if min_dist > r
        return 0
    end

    if max_dist < r
        return addall(tree, index, idx_in_ball, skip, dedup, self_idx)
    end

    # At a leaf node. Go through all points in node and add those in range
    if isleaf(tree.tree_data.n_internal_nodes, index)
        return add_points_inrange!(idx_in_ball, tree, index, point, r, skip, dedup, self_idx)
    end

    split_val = tree.split_vals[index]
    split_dim = tree.split_dims[index]
    p_dim = point[split_dim]
    split_diff = p_dim - split_val

    count = 0

    left_region, right_region = split_hyperrectangle(hyper_rec, split_dim, split_val)
    left_idx = getleft(index)
    right_idx = getright(index)
    if split_diff > 0 # Point is to the right of the split value
        close = right_idx
        far = left_idx
        hyper_rec_far = left_region
        hyper_rec_close = right_region
    else # Point is to the left of the split value
        close = left_idx
        far = right_idx
        hyper_rec_far = right_region
        hyper_rec_close = left_region
    end
    # Compute contributions for both close and far subtrees
    M = tree.metric
    old_contrib = max_dist_contribs[split_dim]
    if split_diff > 0
        # Point is to the right
        # Close subtree: split_val as new min, far subtree: split_val as new max
        new_contrib_close = get_max_distance_contribution_single(M, point[split_dim], split_val, hyper_rec.maxes[split_dim], split_dim)
        new_contrib_far = get_max_distance_contribution_single(M, point[split_dim], hyper_rec.mins[split_dim], split_val, split_dim)
    else
        # Point is to the left
        # Close subtree: split_val as new max, far subtree: split_val as new min
        new_contrib_close = get_max_distance_contribution_single(M, point[split_dim], hyper_rec.mins[split_dim], split_val, split_dim)
        new_contrib_far = get_max_distance_contribution_single(M, point[split_dim], split_val, hyper_rec.maxes[split_dim], split_dim)
    end

    # Update contributions and distances for close subtree
    new_max_contribs_close = setindex(max_dist_contribs, new_contrib_close, split_dim)
    new_max_dist_close = M isa Chebyshev ? maximum(new_max_contribs_close) : max_dist - old_contrib + new_contrib_close

    # Call closer sub tree
    count += inrange_kernel!(tree, close, point, r, idx_in_ball, hyper_rec_close, min_dist, new_max_contribs_close, new_max_dist_close, skip, dedup, self_idx)

    # Compute new min distance for far subtree
    new_min = M isa Chebyshev ? get_min_distance_no_end(M, hyper_rec_far, point) : update_new_min(M, min_dist, hyper_rec, p_dim, split_dim, split_val)

    # Update contributions and distances for far subtree
    new_max_contribs_far = setindex(max_dist_contribs, new_contrib_far, split_dim)
    new_max_dist_far = M isa Chebyshev ? maximum(new_max_contribs_far) : max_dist - old_contrib + new_contrib_far

    # Call further sub tree
    count += inrange_kernel!(tree, far, point, r, idx_in_ball, hyper_rec_far, new_min, new_max_contribs_far, new_max_dist_far, skip, dedup, self_idx)
    return count
end
# Self-query functions for finding all pairs within a tree


@inline function _order_pair(idx1::Int, rect1::HyperRectangle, idx2::Int, rect2::HyperRectangle)
    if idx1 <= idx2
        return idx1, rect1, idx2, rect2
    else
        return idx2, rect2, idx1, rect1
    end
end

function _inrange_kdtree_self!(results,
        tree::KDTree,
        idx::Int,
        rect::HyperRectangle,
        other_idx::Int,
        other_rect::HyperRectangle,
        r::Number,
        skip::F) where {F}
    idx, rect, other_idx, other_rect = _order_pair(idx, rect, other_idx, other_rect)

    min_dist, max_dist = get_min_max_distance(tree.metric, rect, other_rect)
    if min_dist > r
        return
    elseif max_dist < r
        _addall_kdtree_self!(results, tree, idx, other_idx, skip)
        return
    end

    leaf_here = isleaf(tree.tree_data.n_internal_nodes, idx)
    leaf_other = isleaf(tree.tree_data.n_internal_nodes, other_idx)
    if leaf_here
        if leaf_other
            _add_kdtree_self_leaf_pairs!(results, tree, idx, other_idx, r, skip)
        else
            left_other, right_other = split_hyperrectangle(other_rect, tree.split_dims[other_idx], tree.split_vals[other_idx])
            _inrange_kdtree_self!(results, tree, idx, rect, getleft(other_idx), left_other, r, skip)
            _inrange_kdtree_self!(results, tree, idx, rect, getright(other_idx), right_other, r, skip)
        end
    else
        left_rect, right_rect = split_hyperrectangle(rect, tree.split_dims[idx], tree.split_vals[idx])
        if leaf_other
            _inrange_kdtree_self!(results, tree, getleft(idx), left_rect, other_idx, other_rect, r, skip)
            _inrange_kdtree_self!(results, tree, getright(idx), right_rect, other_idx, other_rect, r, skip)
        else
            left_other, right_other = split_hyperrectangle(other_rect, tree.split_dims[other_idx], tree.split_vals[other_idx])
            if idx == other_idx
                _inrange_kdtree_self!(results, tree, getleft(idx), left_rect, getleft(other_idx), left_other, r, skip)
                _inrange_kdtree_self!(results, tree, getleft(idx), left_rect, getright(other_idx), right_other, r, skip)
                _inrange_kdtree_self!(results, tree, getright(idx), right_rect, getright(other_idx), right_other, r, skip)
            else
                _inrange_kdtree_self!(results, tree, getleft(idx), left_rect, getleft(other_idx), left_other, r, skip)
                _inrange_kdtree_self!(results, tree, getleft(idx), left_rect, getright(other_idx), right_other, r, skip)
                _inrange_kdtree_self!(results, tree, getright(idx), right_rect, getleft(other_idx), left_other, r, skip)
                _inrange_kdtree_self!(results, tree, getright(idx), right_rect, getright(other_idx), right_other, r, skip)
            end
        end
    end
    return
end

# Add all pairs between two leaves when their rectangles are fully enclosed by the search radius
function _addall_kdtree_self_leaf_pairs!(results, tree::KDTree, leaf_idx::Int, other_leaf_idx::Int, skip)
    point_range = get_leaf_range(tree.tree_data, leaf_idx)
    if leaf_idx == other_leaf_idx
        @inbounds for i in point_range
            idx_i = tree.indices[i]
            skip(idx_i) && continue
            for j in (i + 1):last(point_range)
                idx_j = tree.indices[j]
                if skip(idx_j)
                    continue
                end
                a, b = idx_i < idx_j ? (idx_i, idx_j) : (idx_j, idx_i)
                push!(results, (a, b))
            end
        end
    else
        query_range = get_leaf_range(tree.tree_data, other_leaf_idx)
        @inbounds for i in point_range
            idx_i = tree.indices[i]
            skip(idx_i) && continue
            for j in query_range
                idx_j = tree.indices[j]
                if skip(idx_j)
                    continue
                end
                a, b = idx_i < idx_j ? (idx_i, idx_j) : (idx_j, idx_i)
                push!(results, (a, b))
            end
        end
    end
    return
end

# Add only those pairs between two leaves that are within the radius when bounds intersect but aren’t fully enclosed
function _add_kdtree_self_leaf_pairs!(results, tree::KDTree, leaf_idx::Int, other_leaf_idx::Int, r::Number, skip)
    point_range = get_leaf_range(tree.tree_data, leaf_idx)
    if leaf_idx == other_leaf_idx
        @inbounds for i in point_range
            idx_i = tree.indices[i]
            skip(idx_i) && continue
            point_i = tree.data[tree.reordered ? i : tree.indices[i]]
            for j in (i + 1):last(point_range)
                idx_j = tree.indices[j]
                if skip(idx_j)
                    continue
                end
                if evaluate_maybe_end(tree.metric, point_i, tree.data[tree.reordered ? j : tree.indices[j]], false) <= r
                    a, b = idx_i < idx_j ? (idx_i, idx_j) : (idx_j, idx_i)
                    push!(results, (a, b))
                end
            end
        end
    else
        query_range = get_leaf_range(tree.tree_data, other_leaf_idx)
        @inbounds for i in point_range
            idx_i = tree.indices[i]
            skip(idx_i) && continue
            point_i = tree.data[tree.reordered ? i : tree.indices[i]]
            for j in query_range
                idx_j = tree.indices[j]
                if skip(idx_j)
                    continue
                end
                if evaluate_maybe_end(tree.metric, point_i, tree.data[tree.reordered ? j : tree.indices[j]], false) <= r
                    a, b = idx_i < idx_j ? (idx_i, idx_j) : (idx_j, idx_i)
                    push!(results, (a, b))
                end
            end
        end
    end
    return
end

# Add all pairs from two subtrees without distance checks once both rectangles are fully inside the radius
function _addall_kdtree_self!(results, tree::KDTree, idx::Int, other_idx::Int, skip::F) where {F}
    leaf_here = isleaf(tree.tree_data.n_internal_nodes, idx)
    leaf_other = isleaf(tree.tree_data.n_internal_nodes, other_idx)
    if leaf_here
        if leaf_other
            _addall_kdtree_self_leaf_pairs!(results, tree, idx, other_idx, skip)
        else
            _addall_kdtree_self!(results, tree, idx, getleft(other_idx), skip)
            _addall_kdtree_self!(results, tree, idx, getright(other_idx), skip)
        end
    else
        if leaf_other
            _addall_kdtree_self!(results, tree, getleft(idx), other_idx, skip)
            _addall_kdtree_self!(results, tree, getright(idx), other_idx, skip)
        else
            _addall_kdtree_self!(results, tree, getleft(idx), getleft(other_idx), skip)
            _addall_kdtree_self!(results, tree, getleft(idx), getright(other_idx), skip)
            _addall_kdtree_self!(results, tree, getright(idx), getleft(other_idx), skip)
            _addall_kdtree_self!(results, tree, getright(idx), getright(other_idx), skip)
        end
    end
    return
end

function _inrange_pairs(tree::KDTree{V}, radius::Number, sortres, skip::F) where {V, F}
    isempty(tree.data) && return NTuple{2,Int}[]
    pairs = NTuple{2,Int}[]
    root_rect = tree.hyper_rec
    _inrange_kdtree_self!(pairs, tree, 1, root_rect, 1, root_rect, eval_pow(tree.metric, radius), skip)
    sortres && sort!(pairs)
    return pairs
end
