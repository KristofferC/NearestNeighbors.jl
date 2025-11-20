const MaybeBitSet = Union{Nothing, BitSet}

# Helper functions to get node numbers and points
@inline getleft(i::Int) = 2i
@inline getright(i::Int) = 2i + 1
@inline getparent(i::Int) = div(i, 2)
@inline isleaf(n_internal_nodes::Int, idx::Int) = idx > n_internal_nodes

function Base.show(io::IO, tree::NNTree{V}) where {V}
    println(io, typeof(tree))
    println(io, "  Number of points: ", length(tree.data))
    println(io, "  Dimensions: ", length(V))
    println(io, "  Metric: ", tree.metric)
    print(io,   "  Reordered: ", tree.reordered)
end

"""
    find_split(low, leafsize, n_p)

Return the index of the point that becomes the root of the current subtree.
The split follows two deterministic rules:

1. One of the child subtrees must contain an exact power-of-two number of
   leaves (so its structure is a perfect binary tree).
2. The left subtree receives at least as many points as the right subtree.

Because every leaf except the last contains `leafsize` points, these rules can
be satisfied with integer arithmetic alone.  The helper below derives the
number of leaves that must stay on the left, converts that to a point count,
and compensates for the single partially filled leaf if the left subtree owns
it.  `low` is the first index in the active range, i.e. the call site splits
`low:(low + n_p - 1)` at the position returned here.

Example: `leafsize = 4`, `n_p = 10`.  Then `n_leafs = 3`, `pow = 2`,
`half_pow = 1`, `left_leaves = clamp(3 - 1, 1, 2) = 2`, so `left_points = 8`.
Since the right subtree hosts the perfect half-row and the final leaf is only
two points large, we subtract `leafsize - last_node_size = 2` and end up with
`left_points = 6`, i.e. the left subtree receives indices `low:low+5` as
expected.
"""
function find_split(low, leafsize, n_p)

    # Number of leaves that cover the current range of points.
    n_leafs = ceil(Int, n_p / leafsize)

    # Largest power-of-two subtree that fits in `n_leafs`.
    k = floor(Integer, log2(n_leafs))
    pow = 2^k
    half_pow = max(1, pow >>> 1)

    # Keep as many leaves on the left as possible while ensuring at least a
    # `half_pow` block stays on the right.  The clamp captures both scenarios:
    #   - When `rest > half_pow`, it saturates at `pow` (Rule 1 satisfied by the
    #     left subtree).
    #   - When `rest ≤ half_pow`, it returns `n_leafs - half_pow` so the right
    #     subtree receives the perfect `half_pow` block.
    left_leaves = clamp(n_leafs - half_pow, half_pow, pow)
    left_points = left_leaves * leafsize

    # If the data does not fill the last leaf completely, the deficit is
    # `leafsize - last_node_size`.  That deficit belongs to the left subtree
    # exactly when we fall in the `rest ∈ (0, half_pow]` scenario (i.e. the
    # right subtree hosts the perfect block).
    rest = n_leafs - pow
    last_node_size = n_p % leafsize
    if last_node_size == 0
        last_node_size = leafsize
    end
    if rest != 0 && rest <= half_pow && last_node_size != leafsize
        left_points -= leafsize - last_node_size
    end

    return low + left_points
end

# Gets number of points in a leaf node, this is equal to leafsize for every node
# except the last node.
@inline function n_ps(idx::Int, td::TreeData)
    if idx != td.last_full_node
        return td.leafsize
    else
        return td.last_node_size
    end
end

# Returns the index for the first point for a given leaf node.
@inline function point_index(idx::Int, td::TreeData)
    if idx >= td.cross_node
        return td.offset_cross + idx * td.leafsize
    else
        return td.offset + idx * td.leafsize
    end
end

# Returns a range over the points in a leaf node with a given index
@inline function get_leaf_range(td::TreeData, index)
    p_index = point_index(index, td)
    n_p = n_ps(index, td)
    return p_index:p_index + n_p - 1
end

# Store all the points in a leaf node continuously in memory in data_reordered to improve cache locality.
# Also stores the mapping to get the index into the original data from the reordered data.
function reorder_data!(data_reordered::Vector{V}, data::AbstractVector{V}, index::Int,
                         indices::Vector{Int}, indices_reordered::Vector{Int}, tree_data::TreeData) where {V}

    for i in get_leaf_range(tree_data, index)
        idx = indices[i]
        data_reordered[i] = data[idx]
        # Saves the inverse n
        indices_reordered[i] = idx
    end
end

@inline maybe_update_index(v::AbstractVector, i, val) = (v[i] = val; v)
@inline maybe_update_index(v::Number, i, val) = val

# Checks the distance function and add those points that are among the k best.
# Uses a heap for fast insertion.
@inline function add_points_knn!(best_dists::Union{Number, AbstractVector},
                                 best_idxs::Union{Integer, AbstractVector{<:Integer}},
                                 tree::NNTree, index::Int, point::AbstractVector,
                                 do_end::Bool, skip::F,
                                 dedup::MaybeBitSet) where {F}
    has_set = dedup !== nothing
    for z in get_leaf_range(tree.tree_data, index)
        if skip(tree.indices[z])
            continue
        end
        idx = tree.reordered ? z : tree.indices[z]
        dist_d = evaluate_maybe_end(tree.metric, tree.data[idx], point, do_end)
        update_existing_neighbor!(dedup, idx, dist_d, best_idxs, best_dists) && continue
        best_dist_1 = first(best_dists)
        if dist_d < best_dist_1
            has_set && push!(dedup, idx)
            best_dists = maybe_update_index(best_dists, 1, dist_d)
            best_idxs = maybe_update_index(best_idxs, 1, idx)
            best_dists isa AbstractVector && percolate_down!(best_dists, best_idxs, dist_d, idx)
        end
    end
    return best_idxs, best_dists
end

# Add those points in the leaf node that are within range.
# TODO: If we have a distance function that is incrementally increased
# as we sum over the dimensions (like the Minkowski norms) then we could
# stop computing the distance function as soon as we reach the desired radius.
# This will probably prevent SIMD and other optimizations so some care is needed
# to evaluate if it is worth it.
@inline function add_points_inrange!(idx_in_ball::Union{Nothing, AbstractVector{<:Integer}}, tree::NNTree,
                                     index::Int, point::AbstractVector, r::Number, skip::Function,
                                     dedup::MaybeBitSet)
    count = 0
    has_set = dedup !== nothing
    for z in get_leaf_range(tree.tree_data, index)
        if skip(tree.indices[z])
            continue
        end
        idx = tree.reordered ? z : tree.indices[z]
        if check_in_range(tree.metric, tree.data[idx], point, r)
            if has_set && idx in dedup
                continue
            end
            has_set && push!(dedup, idx)
            count += 1
            idx_in_ball !== nothing && push!(idx_in_ball, idx)
        end
    end
    return count
end

function check_in_range(metric::Metric, x1, x2, r)
    evaluate(metric, x1, x2) <= r
end

function check_in_range(metric::MinkowskiMetric, x1, x2, r)
    evaluate_maybe_end(metric, x1, x2, false) <= r
end


# Add all points in this subtree since we have determined
# they are all within the desired range
function addall(tree::NNTree, index::Int, idx_in_ball::Union{Nothing, Vector{<:Integer}}, skip::Function,
                dedup::MaybeBitSet)
    tree_data = tree.tree_data
    if isleaf(tree_data.n_internal_nodes, index)
        count = 0
        has_set = dedup !== nothing
        for z in get_leaf_range(tree_data, index)
            if skip(tree.indices[z])
                continue
            end
            idx = tree.reordered ? z : tree.indices[z]
            if has_set && idx in dedup
                continue
            end
            has_set && push!(dedup, idx)
            count += 1
            idx_in_ball !== nothing && push!(idx_in_ball, idx)
        end
        return count
    else
        return addall(tree, getleft(index), idx_in_ball, skip, dedup) +
               addall(tree, getright(index), idx_in_ball, skip, dedup)
    end
end

@inline function update_existing_neighbor!(dedup::MaybeBitSet, idx::Int, dist_d, best_idxs, best_dists)
    dedup === nothing && return false
    if idx in dedup
        pos = findfirst(==(idx), best_idxs)
        if pos === nothing
            delete!(dedup, idx)
            return false
        end
        if dist_d < best_dists[pos]
            best_dists[pos] = dist_d
            percolate_down!(best_dists, best_idxs, dist_d, idx, pos, length(best_dists))
        end
        return true
    end
    return false
end
