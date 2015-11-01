# Helper functions to get node numbers and points
@inline getleft(i::Int) = 2i
@inline getright(i::Int) = 2i + 1
@inline getparent(i::Int) = div(i, 2)
@inline isleaf(n_internal_nodes::Int, idx::Int) = idx > n_internal_nodes

function show(io::IO, tree::NNTree)
    println(io, typeof(tree))
    println(io, "  Number of points: ", size(tree.data, 2))
    println(io, "  Dimensions: ", size(tree.data, 1))
    println(io, "  Metric: ", tree.metric)
    print(io, "  Reordered: ", tree.reordered)
end

# If the last leaf node will be on the right side of the tree we
# send points so that left tree will be perfectly filled,
# else we do the opposite.
function find_split(low, leaf_size, n_p)

    # The number of leafs left, ceil to count a partially
    # filled node as 1.
    n_leafs = ceil(Int, n_p / leaf_size)

    # Rows left in the sub tree
    k = floor(Int, log2(n_leafs))

    # Number of leftover nodes needed
    rest = n_leafs - 2^k

    if k == 0
        return low
    elseif n_p <= 2*leaf_size
        return leaf_size + low
    elseif  rest > 2^(k-1)
        return 2^k * leaf_size + low
    elseif rest == 0
        return 2^(k-1)* leaf_size + low
    else
        return n_p - 2^(k-1) * leaf_size + low
    end
end

# Gets number of points in a node, leaf_size for every node
# except the last one
@inline function n_ps(n_leafs::Int, n_internal_nodes::Int, leaf_size::Int,
                              last_node_size::Int, idx::Int)
    if idx != n_leafs + n_internal_nodes
        return leaf_size
    else
        return last_node_size
    end
end


# Returns the index for the first point for a given leaf node.
@inline function point_index(cross_node::Int, offset::Int, last_size:: Int,
                     leafsize::Int, n_internal::Int, idx::Int)
    if idx >= cross_node
        return (idx - cross_node) * leafsize + 1
    else
        return ((offset + idx - 1 - n_internal) * leafsize
                  + last_size + 1)
    end
end

@inline function get_leaf_range(tree_data, index)
    p_index = point_index(tree_data.cross_node, tree_data.offset, tree_data.last_node_size,
                              tree_data.leaf_size, tree_data.n_internal_nodes, index)
    n_p =  n_ps(tree_data.n_leafs, tree_data.n_internal_nodes, tree_data.leaf_size,
                    tree_data.last_node_size, index)
    return p_index:p_index + n_p - 1
end

function reorder_data!(data_reordered, data, index, indices, indices_reordered, tree_data)
    # Here we reorder the data points so that points contained
    # in nodes with an index close to each other are also themselves
    # close in memory.

    for i in get_leaf_range(tree_data, index)
        idx = indices[i]
        @devec data_reordered[:, i] = data[:, idx]

        # Saves the inverse n
        indices_reordered[i] = idx
    end
end


@inline function add_points_knn!{T}(best_dists::Vector{T}, best_idxs::Vector{Int},
                                   tree::NNTree{T}, index::Int, point::Vector{T},
                                   do_end::Bool)
    for z in get_leaf_range(tree.tree_data, index)
        @POINT 1
        idx = tree.reordered ? z : tree.indices[z]
        dist_d = evaluate(tree.metric, tree.data, point, idx, do_end)
        if dist_d <= best_dists[1]
            best_dists[1] = dist_d
            best_idxs[1] = idx
            percolate_down!(best_dists, best_idxs, dist_d, idx)
        end
    end
end

@inline function add_points_inrange!{T}(idx_in_ball::Vector{Int}, tree::NNTree{T},
                                       index::Int, point::Vector{T}, r::T, do_end::Bool)
    for z in get_leaf_range(tree.tree_data, index)
        @POINT 1
        idx = tree.reordered ? z : tree.indices[z]
        dist_d = evaluate(tree.metric, tree.data, point, idx, do_end)
        if dist_d <= r
            push!(idx_in_ball, idx)
        end
    end
end

# Adds everything in this subtree since we have determined
# that the hyper rectangle completely encloses the hyper sphere
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
