check_radius(r) = r < 0 && throw(ArgumentError("the query radius r must be ≧ 0"))

"""
    inrange(tree::NNTree, points, radius) -> indices

Find all the points in the tree which are closer than `radius` to `points`.

# Arguments
- `tree`: The tree instance
- `points`: Query point(s) - can be a vector (single point), matrix (multiple points), or vector of vectors
- `radius`: Search radius

# Returns
- `indices`: Vector of indices of points within the radius

See also: `inrange!`, `inrangecount`.
"""
function inrange(tree::NNTree,
                 points::AbstractVector{T},
                 radius::Number,
                 sortres=false,
                 skip::F = Returns(false)) where {T <: AbstractVector, F}
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)

    idxs = [Vector{Int}() for _ in 1:length(points)]

    for i in 1:length(points)
        inrange_point!(tree, points[i], radius, sortres, idxs[i], skip)
    end
    return idxs
end

inrange_point!(tree, point, radius, sortres, idx, skip::F) where {F} = _inrange_point!(tree, point, radius, sortres, idx, skip)

function _inrange_point!(tree, point, radius, sortres, idx, skip::F) where {F}
    count = _inrange(tree, point, radius, idx, skip)
    if idx !== nothing
        inner_tree = get_tree(tree)
        if inner_tree.reordered
            @inbounds for j in 1:length(idx)
                idx[j] = inner_tree.indices[idx[j]]
            end
        end
        sortres && sort!(idx)
    end
    return count
end

"""
    inrange!(idxs, tree, point, radius)

Same functionality as `inrange` but stores the results in the input vector `idxs`.
Useful to avoid allocations or specify the element type of the output vector.

# Arguments
- `idxs`: Pre-allocated vector to store indices (must be empty)
- `tree`: The tree instance
- `point`: Query point
- `radius`: Search radius

See also: `inrange`, `inrangecount`.
"""
function inrange!(idxs::AbstractVector, tree::NNTree{V}, point::AbstractVector{T}, radius::Number, sortres=false, skip=Returns(false)) where {V, T <: Number}
    check_input(tree, point)
    check_for_nan_in_points(point)
    check_radius(radius)
    length(idxs) == 0 || throw(ArgumentError("idxs must be empty"))
    inrange_point!(tree, point, radius, sortres, idxs, skip)
    return idxs
end

function inrange(tree::NNTree{V}, point::AbstractVector{T}, radius::Number, sortres=false) where {V, T <: Number}
    return inrange!(Int[], tree, point, radius, sortres)
end

function inrange(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, sortres=false) where {V, T <: Number}
    dim = size(points, 1)
    inrange_matrix(tree, points, radius, Val(dim), sortres)
end

function inrange_matrix(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, ::Val{dim}, sortres, skip::F=Returns(false)) where {V, T <: Number, dim, F}
    # TODO: DRY with inrange for AbstractVector
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)
    n_points = size(points, 2)
    idxs = [Vector{Int}() for _ in 1:n_points]

    for i in 1:n_points
        point = SVector{dim,T}(ntuple(j -> points[j, i], Val(dim)))
        inrange_point!(tree, point, radius, sortres, idxs[i], skip)
    end
    return idxs
end

"""
    inrangecount(tree::NNTree, points, radius) -> count

Count all the points in the tree which are closer than `radius` to `points`.

# Arguments
- `tree`: The tree instance
- `points`: Query point(s) - can be a vector (single point), matrix (multiple points), or vector of vectors
- `radius`: Search radius

# Returns
- `count`: Number of points within the radius (integer for single point, vector for multiple points)
"""
function inrangecount(tree::NNTree{V}, point::AbstractVector{T}, radius::Number, skip::F=Returns(false)) where {V, T <: Number, F}
    check_input(tree, point)
    check_for_nan_in_points(point)
    check_radius(radius)
    return inrange_point!(tree, point, radius, false, nothing, skip)
end

function inrangecount(tree::NNTree,
        points::AbstractVector{T},
        radius::Number, skip::F=Returns(false)) where {T <: AbstractVector, F}
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)
    return inrange_point!.(Ref(tree), points, radius, false, nothing, skip)
end

function inrangecount(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, skip::F=Returns(false)) where {V, T <: Number, F}
    dim = size(points, 1)
    inrangecount_matrix(tree, points, radius, Val(dim), skip)
end

function inrangecount_matrix(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, ::Val{dim}, skip::F=Returns(false)) where {V, T <: Number, dim, F}
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)
    n_points = size(points, 2)
    counts = Vector{Int}(undef, n_points)

    for i in 1:n_points
        point = SVector{dim,T}(ntuple(j -> points[j, i], Val(dim)))
        counts[i] = inrange_point!(tree, point, radius, false, nothing, skip)
    end
    return counts
end

function inrange(tree::KDTree{V}, radius::Number, sortres=false, skip::F=Returns(false); method::Symbol=:tree) where {V, F}
    check_radius(radius)
    if method == :tree
        isempty(tree.data) && return NTuple{2,Int}[]
        pairs = NTuple{2,Int}[]
        root_rect = tree.hyper_rec
        _inrange_kdtree_self!(pairs, tree, 1, root_rect, 1, root_rect, eval_pow(tree.metric, radius), skip)
        sortres && sort!(pairs)
        return pairs
    elseif method == :point
        pairs = _inrange_kdtree_self_pointwise(tree, radius, skip)
        sortres && sort!(pairs)
        return pairs
    else
        throw(ArgumentError("Unknown method '$method'. Use :tree or :point."))
    end
end

function inrange(tree::BallTree{V}, radius::Number, sortres=false, skip::F=Returns(false); method::Symbol=:tree) where {V, F}
    check_radius(radius)
    if method == :tree
        isempty(tree.data) && return NTuple{2,Int}[]
        pairs = NTuple{2,Int}[]
        _inrange_balltree_self!(pairs, tree, 1, 1, radius, skip)
        sortres && sort!(pairs)
        return pairs
    elseif method == :point
        pairs = _inrange_balltree_self_pointwise(tree, radius, skip)
        sortres && sort!(pairs)
        return pairs
    else
        throw(ArgumentError("Unknown method '$method'. Use :tree or :point."))
    end
end

@inline function _kdtree_child_rectangles(tree::KDTree, index::Int, rect::HyperRectangle)
    split_dim = Int(tree.split_dims[index])
    split_val = tree.split_vals[index]
    left_rect = HyperRectangle(rect.mins, setindex(rect.maxes, split_val, split_dim))
    right_rect = HyperRectangle(setindex(rect.mins, split_val, split_dim), rect.maxes)
    return left_rect, right_rect
end

@inline function _kdtree_data_index(tree::KDTree, idx::Int)
    tree.reordered ? idx : tree.indices[idx]
end

@inline function _kdtree_query_slot(tree::KDTree, idx::Int)
    tree.indices[idx]
end

@inline function _kdtree_output_index(tree::KDTree, idx::Int)
    tree.indices[idx]
end

@inline function _balltree_data_index(tree::BallTree, idx::Int)
    tree.reordered ? idx : tree.indices[idx]
end

@inline function _balltree_output_index(tree::BallTree, idx::Int)
    tree.indices[idx]
end

@inline function _balltree_child_indices(index::Int)
    return getleft(index), getright(index)
end

@inline function _balltree_min_max(m::Metric, s1::HyperSphere, s2::HyperSphere)
    dist = evaluate(m, s1.center, s2.center)
    min_d = max(zero(dist), dist - (s1.r + s2.r))
    max_d = dist + s1.r + s2.r
    return min_d, max_d
end

function _addall_kdtree_leaf_pairs!(results, tree::KDTree, leaf_idx::Int, other::KDTree, other_leaf_idx::Int, skip)
    point_range = get_leaf_range(tree.tree_data, leaf_idx)
    query_range = get_leaf_range(other.tree_data, other_leaf_idx)
    @inbounds for q in query_range
        out_idx = _kdtree_query_slot(other, q)
        dest = results[out_idx]
        for p in point_range
            original_idx = tree.indices[p]
            skip(original_idx) && continue
            push!(dest, _kdtree_data_index(tree, p))
        end
    end
    return
end

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
            left_other, right_other = _kdtree_child_rectangles(tree, other_idx, other_rect)
            _inrange_kdtree_self!(results, tree, idx, rect, getleft(other_idx), left_other, r, skip)
            _inrange_kdtree_self!(results, tree, idx, rect, getright(other_idx), right_other, r, skip)
        end
    else
        left_rect, right_rect = _kdtree_child_rectangles(tree, idx, rect)
        if leaf_other
            _inrange_kdtree_self!(results, tree, getleft(idx), left_rect, other_idx, other_rect, r, skip)
            _inrange_kdtree_self!(results, tree, getright(idx), right_rect, other_idx, other_rect, r, skip)
        else
            left_other, right_other = _kdtree_child_rectangles(tree, other_idx, other_rect)
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

function _inrange_kdtree_self_pointwise(tree::KDTree, radius::Number, skip)
    pairs = NTuple{2,Int}[]
    tmp = Int[]
    for z in 1:length(tree.data)
        orig_i = _kdtree_output_index(tree, z)
        skip(orig_i) && continue
        empty!(tmp)
        inrange_point!(tree, tree.data[z], radius, false, tmp, skip)
        @inbounds for j in tmp
            j == orig_i && continue
            if orig_i < j
                push!(pairs, (orig_i, j))
            end
        end
    end
    return pairs
end

function _inrange_balltree_self_pointwise(tree::BallTree, radius::Number, skip)
    pairs = NTuple{2,Int}[]
    tmp = Int[]
    for z in 1:length(tree.data)
        orig_i = _balltree_output_index(tree, z)
        skip(orig_i) && continue
        empty!(tmp)
        local_skip = x -> (x == orig_i || skip(x))
        inrange_point!(tree, tree.data[_balltree_data_index(tree, z)], radius, false, tmp, local_skip)
        @inbounds for j in tmp
            j == orig_i && continue
            if orig_i < j
                push!(pairs, (orig_i, j))
            end
        end
    end
    return pairs
end

function _add_kdtree_leaf_pairs!(results, tree::KDTree, leaf_idx::Int, other::KDTree, other_leaf_idx::Int, r::Number, skip)
    point_range = get_leaf_range(tree.tree_data, leaf_idx)
    query_range = get_leaf_range(other.tree_data, other_leaf_idx)
    @inbounds for q in query_range
        q_out_idx = _kdtree_query_slot(other, q)
        q_point = other.data[_kdtree_data_index(other, q)]
        dest = results[q_out_idx]
        for p in point_range
            original_idx = tree.indices[p]
            skip(original_idx) && continue
            candidate_idx = _kdtree_data_index(tree, p)
            if evaluate_maybe_end(tree.metric, tree.data[candidate_idx], q_point, false) <= r
                push!(dest, candidate_idx)
            end
        end
    end
    return
end

function _addall_balltree_pairs!(results, tree::BallTree, idx::Int, other::BallTree, other_idx::Int, skip)
    leaf_here = isleaf(tree.tree_data.n_internal_nodes, idx)
    leaf_other = isleaf(other.tree_data.n_internal_nodes, other_idx)
    if leaf_here
        if leaf_other
            point_range = get_leaf_range(tree.tree_data, idx)
            query_range = get_leaf_range(other.tree_data, other_idx)
            @inbounds for q in query_range
                out_idx = _balltree_output_index(other, q)
                dest = results[out_idx]
                for p in point_range
                    cand = _balltree_data_index(tree, p)
                    skip(tree.indices[p]) && continue
                    push!(dest, cand)
                end
            end
        else
            _addall_balltree_pairs!(results, tree, idx, other, getleft(other_idx), skip)
            _addall_balltree_pairs!(results, tree, idx, other, getright(other_idx), skip)
        end
    else
        if leaf_other
            _addall_balltree_pairs!(results, tree, getleft(idx), other, other_idx, skip)
            _addall_balltree_pairs!(results, tree, getright(idx), other, other_idx, skip)
        else
            _addall_balltree_pairs!(results, tree, getleft(idx), other, getleft(other_idx), skip)
            _addall_balltree_pairs!(results, tree, getleft(idx), other, getright(other_idx), skip)
            _addall_balltree_pairs!(results, tree, getright(idx), other, getleft(other_idx), skip)
            _addall_balltree_pairs!(results, tree, getright(idx), other, getright(other_idx), skip)
        end
    end
    return
end

function _addall_kdtree_self_leaf_pairs!(results, tree::KDTree, leaf_idx::Int, other_leaf_idx::Int, skip)
    point_range = get_leaf_range(tree.tree_data, leaf_idx)
    query_range = get_leaf_range(tree.tree_data, other_leaf_idx)
    if leaf_idx == other_leaf_idx
        @inbounds for i in point_range
            idx_i = _kdtree_output_index(tree, i)
            skip(idx_i) && continue
            for j in (i + 1):last(point_range)
                idx_j = _kdtree_output_index(tree, j)
                if skip(idx_j)
                    continue
                end
                a, b = idx_i < idx_j ? (idx_i, idx_j) : (idx_j, idx_i)
                push!(results, (a, b))
            end
        end
    else
        @inbounds for i in point_range
            idx_i = _kdtree_output_index(tree, i)
            skip(idx_i) && continue
            for j in query_range
                idx_j = _kdtree_output_index(tree, j)
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

function _add_kdtree_self_leaf_pairs!(results, tree::KDTree, leaf_idx::Int, other_leaf_idx::Int, r::Number, skip)
    point_range = get_leaf_range(tree.tree_data, leaf_idx)
    query_range = get_leaf_range(tree.tree_data, other_leaf_idx)
    if leaf_idx == other_leaf_idx
        @inbounds for i in point_range
            idx_i = _kdtree_output_index(tree, i)
            skip(idx_i) && continue
            point_i = tree.data[_kdtree_data_index(tree, i)]
            for j in (i + 1):last(point_range)
                idx_j = _kdtree_output_index(tree, j)
                if skip(idx_j)
                    continue
                end
                if evaluate_maybe_end(tree.metric, point_i, tree.data[_kdtree_data_index(tree, j)], false) <= r
                    a, b = idx_i < idx_j ? (idx_i, idx_j) : (idx_j, idx_i)
                    push!(results, (a, b))
                end
            end
        end
    else
        @inbounds for i in point_range
            idx_i = _kdtree_output_index(tree, i)
            skip(idx_i) && continue
            point_i = tree.data[_kdtree_data_index(tree, i)]
            for j in query_range
                idx_j = _kdtree_output_index(tree, j)
                if skip(idx_j)
                    continue
                end
                if evaluate_maybe_end(tree.metric, point_i, tree.data[_kdtree_data_index(tree, j)], false) <= r
                    a, b = idx_i < idx_j ? (idx_i, idx_j) : (idx_j, idx_i)
                    push!(results, (a, b))
                end
            end
        end
    end
    return
end

function _addall_kdtree_pairs!(results, tree::KDTree, idx::Int, other::KDTree, other_idx::Int, skip)
    leaf_here = isleaf(tree.tree_data.n_internal_nodes, idx)
    leaf_other = isleaf(other.tree_data.n_internal_nodes, other_idx)
    if leaf_here
        if leaf_other
            _addall_kdtree_leaf_pairs!(results, tree, idx, other, other_idx, skip)
        else
            _addall_kdtree_pairs!(results, tree, idx, other, getleft(other_idx), skip)
            _addall_kdtree_pairs!(results, tree, idx, other, getright(other_idx), skip)
        end
    else
        if leaf_other
            _addall_kdtree_pairs!(results, tree, getleft(idx), other, other_idx, skip)
            _addall_kdtree_pairs!(results, tree, getright(idx), other, other_idx, skip)
        else
            _addall_kdtree_pairs!(results, tree, getleft(idx), other, getleft(other_idx), skip)
            _addall_kdtree_pairs!(results, tree, getleft(idx), other, getright(other_idx), skip)
            _addall_kdtree_pairs!(results, tree, getright(idx), other, getleft(other_idx), skip)
            _addall_kdtree_pairs!(results, tree, getright(idx), other, getright(other_idx), skip)
        end
    end
    return
end

function _addall_balltree_self!(results::Vector{NTuple{2,Int}}, tree::BallTree, idx::Int, other_idx::Int, skip)
    leaf_here = isleaf(tree.tree_data.n_internal_nodes, idx)
    leaf_other = isleaf(tree.tree_data.n_internal_nodes, other_idx)
    if leaf_here
        if leaf_other
            _addall_balltree_self_leaf_pairs!(results, tree, idx, other_idx, skip)
        else
            _addall_balltree_self!(results, tree, idx, getleft(other_idx), skip)
            _addall_balltree_self!(results, tree, idx, getright(other_idx), skip)
        end
    else
        if leaf_other
            _addall_balltree_self!(results, tree, getleft(idx), other_idx, skip)
            _addall_balltree_self!(results, tree, getright(idx), other_idx, skip)
        else
            _addall_balltree_self!(results, tree, getleft(idx), getleft(other_idx), skip)
            _addall_balltree_self!(results, tree, getleft(idx), getright(other_idx), skip)
            if idx == other_idx
                _addall_balltree_self!(results, tree, getright(idx), getright(other_idx), skip)
            else
                _addall_balltree_self!(results, tree, getright(idx), getleft(other_idx), skip)
                _addall_balltree_self!(results, tree, getright(idx), getright(other_idx), skip)
            end
        end
    end
    return
end

function _addall_balltree_self_leaf_pairs!(results::Vector{NTuple{2,Int}}, tree::BallTree, leaf_idx::Int, other_leaf_idx::Int, skip)
    point_range = get_leaf_range(tree.tree_data, leaf_idx)
    query_range = get_leaf_range(tree.tree_data, other_leaf_idx)
    if leaf_idx == other_leaf_idx
        @inbounds for i in point_range
            idx_i = _balltree_output_index(tree, i)
            skip(idx_i) && continue
            for j in (i + 1):last(point_range)
                idx_j = _balltree_output_index(tree, j)
                if skip(idx_j)
                    continue
                end
                a, b = idx_i < idx_j ? (idx_i, idx_j) : (idx_j, idx_i)
                push!(results, (a, b))
            end
        end
    else
        @inbounds for i in point_range
            idx_i = _balltree_output_index(tree, i)
            skip(idx_i) && continue
            for j in query_range
                idx_j = _balltree_output_index(tree, j)
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

function _add_balltree_self_leaf_pairs!(results::Vector{NTuple{2,Int}}, tree::BallTree, leaf_idx::Int, other_leaf_idx::Int, r::Number, skip)
    point_range = get_leaf_range(tree.tree_data, leaf_idx)
    query_range = get_leaf_range(tree.tree_data, other_leaf_idx)
    if leaf_idx == other_leaf_idx
        @inbounds for i in point_range
            idx_i = _balltree_output_index(tree, i)
            skip(idx_i) && continue
            point_i = tree.data[_balltree_data_index(tree, i)]
            for j in (i + 1):last(point_range)
                idx_j = _balltree_output_index(tree, j)
                if skip(idx_j)
                    continue
                end
                if evaluate(tree.metric, point_i, tree.data[_balltree_data_index(tree, j)]) <= r
                    a, b = idx_i < idx_j ? (idx_i, idx_j) : (idx_j, idx_i)
                    push!(results, (a, b))
                end
            end
        end
    else
        @inbounds for i in point_range
            idx_i = _balltree_output_index(tree, i)
            skip(idx_i) && continue
            point_i = tree.data[_balltree_data_index(tree, i)]
            for j in query_range
                idx_j = _balltree_output_index(tree, j)
                if skip(idx_j)
                    continue
                end
                if evaluate(tree.metric, point_i, tree.data[_balltree_data_index(tree, j)]) <= r
                    a, b = idx_i < idx_j ? (idx_i, idx_j) : (idx_j, idx_i)
                    push!(results, (a, b))
                end
            end
        end
    end
    return
end

function _addall_kdtree_self!(results, tree::KDTree, idx::Int, other_idx::Int, skip)
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

function _inrange_kdtree_tree!(results,
        tree::KDTree,
        idx::Int,
        rect::HyperRectangle,
        other::KDTree,
        other_idx::Int,
        other_rect::HyperRectangle,
        r::Number,
        skip::F) where {F}
    min_dist, max_dist = get_min_max_distance(tree.metric, rect, other_rect)
    if min_dist > r
        return
    elseif max_dist < r
        _addall_kdtree_pairs!(results, tree, idx, other, other_idx, skip)
        return
    end

    leaf_here = isleaf(tree.tree_data.n_internal_nodes, idx)
    leaf_other = isleaf(other.tree_data.n_internal_nodes, other_idx)
    if leaf_here
        if leaf_other
            _add_kdtree_leaf_pairs!(results, tree, idx, other, other_idx, r, skip)
        else
            left_other, right_other = _kdtree_child_rectangles(other, other_idx, other_rect)
            _inrange_kdtree_tree!(results, tree, idx, rect, other, getleft(other_idx), left_other, r, skip)
            _inrange_kdtree_tree!(results, tree, idx, rect, other, getright(other_idx), right_other, r, skip)
        end
    else
        left_rect, right_rect = _kdtree_child_rectangles(tree, idx, rect)
        if leaf_other
            _inrange_kdtree_tree!(results, tree, getleft(idx), left_rect, other, other_idx, other_rect, r, skip)
            _inrange_kdtree_tree!(results, tree, getright(idx), right_rect, other, other_idx, other_rect, r, skip)
        else
            left_other, right_other = _kdtree_child_rectangles(other, other_idx, other_rect)
            _inrange_kdtree_tree!(results, tree, getleft(idx), left_rect, other, getleft(other_idx), left_other, r, skip)
            _inrange_kdtree_tree!(results, tree, getleft(idx), left_rect, other, getright(other_idx), right_other, r, skip)
            _inrange_kdtree_tree!(results, tree, getright(idx), right_rect, other, getleft(other_idx), left_other, r, skip)
            _inrange_kdtree_tree!(results, tree, getright(idx), right_rect, other, getright(other_idx), right_other, r, skip)
        end
    end
    return
end

function _inrange_balltree_tree!(results,
        tree::BallTree,
        idx::Int,
        other::BallTree,
        other_idx::Int,
        r::Number,
        skip::F) where {F}
    if idx > length(tree.hyper_spheres) || other_idx > length(other.hyper_spheres)
        return
    end
    sphere = tree.hyper_spheres[idx]
    other_sphere = other.hyper_spheres[other_idx]
    min_d, max_d = _balltree_min_max(tree.metric, sphere, other_sphere)
    if min_d > r
        return
    elseif max_d < r
        _addall_balltree_pairs!(results, tree, idx, other, other_idx, skip)
        return
    end

    leaf_here = isleaf(tree.tree_data.n_internal_nodes, idx)
    leaf_other = isleaf(other.tree_data.n_internal_nodes, other_idx)
    if leaf_here
        if leaf_other
            point_range = get_leaf_range(tree.tree_data, idx)
            query_range = get_leaf_range(other.tree_data, other_idx)
            @inbounds for q in query_range
                out_idx = _balltree_output_index(other, q)
                dest = results[out_idx]
                for p in point_range
                    cand_orig = tree.indices[p]
                    skip(cand_orig) && continue
                    if evaluate(tree.metric, tree.data[_balltree_data_index(tree, p)], other.data[_balltree_data_index(other, q)]) <= r
                        push!(dest, _balltree_data_index(tree, p))
                    end
                end
            end
        else
            _inrange_balltree_tree!(results, tree, idx, other, getleft(other_idx), r, skip)
            _inrange_balltree_tree!(results, tree, idx, other, getright(other_idx), r, skip)
        end
    else
        if leaf_other
            _inrange_balltree_tree!(results, tree, getleft(idx), other, other_idx, r, skip)
            _inrange_balltree_tree!(results, tree, getright(idx), other, other_idx, r, skip)
        else
            _inrange_balltree_tree!(results, tree, getleft(idx), other, getleft(other_idx), r, skip)
            _inrange_balltree_tree!(results, tree, getleft(idx), other, getright(other_idx), r, skip)
            _inrange_balltree_tree!(results, tree, getright(idx), other, getleft(other_idx), r, skip)
            _inrange_balltree_tree!(results, tree, getright(idx), other, getright(other_idx), r, skip)
        end
    end
    return
end

function _inrange_balltree_self!(results::Vector{NTuple{2,Int}},
        tree::BallTree,
        idx::Int,
        other_idx::Int,
        r::Number,
        skip::F) where {F}
    if idx > length(tree.hyper_spheres) || other_idx > length(tree.hyper_spheres)
        return
    end

    if idx > other_idx
        idx, other_idx = other_idx, idx
    end

    sphere = tree.hyper_spheres[idx]
    other_sphere = tree.hyper_spheres[other_idx]
    min_d, max_d = _balltree_min_max(tree.metric, sphere, other_sphere)
    if min_d > r
        return
    elseif max_d < r
        _addall_balltree_self!(results, tree, idx, other_idx, skip)
        return
    end

    leaf_here = isleaf(tree.tree_data.n_internal_nodes, idx)
    leaf_other = isleaf(tree.tree_data.n_internal_nodes, other_idx)
    if leaf_here
        if leaf_other
            _add_balltree_self_leaf_pairs!(results, tree, idx, other_idx, r, skip)
        else
            _inrange_balltree_self!(results, tree, idx, getleft(other_idx), r, skip)
            _inrange_balltree_self!(results, tree, idx, getright(other_idx), r, skip)
        end
    else
        if leaf_other
            _inrange_balltree_self!(results, tree, getleft(idx), other_idx, r, skip)
            _inrange_balltree_self!(results, tree, getright(idx), other_idx, r, skip)
        else
            _inrange_balltree_self!(results, tree, getleft(idx), getleft(other_idx), r, skip)
            _inrange_balltree_self!(results, tree, getleft(idx), getright(other_idx), r, skip)
            if idx == other_idx
                _inrange_balltree_self!(results, tree, getright(idx), getright(other_idx), r, skip)
            else
                _inrange_balltree_self!(results, tree, getright(idx), getleft(other_idx), r, skip)
                _inrange_balltree_self!(results, tree, getright(idx), getright(other_idx), r, skip)
            end
        end
    end
    return
end
