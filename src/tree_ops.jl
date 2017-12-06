# Helper functions to get node numbers and points
@inline getleft(i::Int) = 2i
@inline getright(i::Int) = 2i + 1
@inline getparent(i::Int) = div(i, 2)
@inline isleaf(n_internal_nodes::Int, idx::Int) = idx > n_internal_nodes

function show(io::IO, tree::NNTree{V}) where {V}
    println(io, typeof(tree))
    println(io, "  Number of points: ", length(tree.data))
    println(io, "  Dimensions: ", length(V))
    println(io, "  Metric: ", tree.metric)
    print(io,   "  Reordered: ", tree.reordered)
end

# We split the tree such that one of the sub trees has exactly 2^p points
# and such that the left sub tree always has more points.
# This means that we can deterministally (with just some comparisons)
# find if we are at a leaf node and how many
function find_split(low, leafsize, n_p)

    # The number of leafs node left in the tree,
    # use `ceil` to count a partially filled node as 1.
    n_leafs = ceil(Int, n_p / leafsize)

    # Number of leftover nodes needed
    k = floor(Integer, log2(n_leafs))
    rest = n_leafs - 2^k

    # The conditionals here fulfill the desired splitting procedure but
    # can probably be written in a nicer way

    # Can fill less than two nodes -> leafsize to left node.
    if n_p <= 2 * leafsize
        mid_idx = leafsize

    # The last leaf node will be in the right sub tree -> fill the left
    # sub tree with
    elseif rest > 2^(k - 1) # Last node over the "half line" in the row
        mid_idx = 2^k * leafsize

    # Perfectly filling both sub trees -> half to left and right sub tree
    elseif rest == 0
        mid_idx = 2^(k - 1) * leafsize

    # Else we fill the right sub tree -> send the rest to the left sub tree
    else
        mid_idx = n_p - 2^(k - 1) * leafsize
    end
    return mid_idx + low
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
    n_p =  n_ps(index, td)
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

# Checks the distance function and add those points that are among the k best.
# Uses a heap for fast insertion.
@inline function add_points_knn!(best_dists::Vector, best_idxs::Vector{Int},
                                 tree::NNTree, index::Int, point::AbstractVector,
                                 do_end::Bool, skip::F) where {F}
    for z in get_leaf_range(tree.tree_data, index)
        @POINT 1
        idx = tree.reordered ? z : tree.indices[z]
        dist_d = evaluate(tree.metric, tree.data[idx], point, do_end)
        if dist_d <= best_dists[1]
            if skip(tree.indices[z])
                continue
            end

            best_dists[1] = dist_d
            best_idxs[1] = idx
            percolate_down!(best_dists, best_idxs, dist_d, idx)
        end
    end
end

# Add those points in the leaf node that are within range.
# TODO: If we have a distance function that is incrementally increased
# as we sum over the dimensions (like the Minkowski norms) then we could
# stop computing the distance function as soon as we reach the desired radius.
# This will probably prevent SIMD and other optimizations so some care is needed
# to evaluate if it is worth it.
@inline function add_points_inrange!(idx_in_ball::Vector{Int}, tree::NNTree,
                                     index::Int, point::AbstractVector, r::Number, do_end::Bool)
    for z in get_leaf_range(tree.tree_data, index)
        @POINT 1
        idx = tree.reordered ? z : tree.indices[z]
        dist_d = evaluate(tree.metric, tree.data[idx], point, do_end)
        if dist_d <= r
            push!(idx_in_ball, idx)
        end
    end
end

# Add all points in this subtree since we have determined
# they are all within the desired range
function addall(tree::NNTree, index::Int, idx_in_ball::Vector{Int})
    tree_data = tree.tree_data
    @NODE 1
    if isleaf(tree.tree_data.n_internal_nodes, index)
        for z in get_leaf_range(tree.tree_data, index)
            @POINT_UNCHECKED 1
            idx = tree.reordered ? z : tree.indices[z]
            push!(idx_in_ball, idx)
        end
        return
    else
        addall(tree, getleft(index), idx_in_ball)
        addall(tree, getright(index), idx_in_ball)
    end
end
