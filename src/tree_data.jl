struct TreeData
    last_node_size::Int    # Number of points in the last node
    leafsize::Int          # Number of points in each leaf node (except last)
    n_leafs::Int           # Number of leafs
    n_internal_nodes::Int  # Number of non leaf nodes
    cross_node::Int
    offset::Int
    offset_cross::Int
    last_full_node::Int
end


function TreeData(data::AbstractVector{V}, leafsize) where V
    n_dim, n_p = length(V), length(data)

    # If number of points is zero
    n_p == 0 && return TreeData(0, 0, 0, 0, 0, 0, 0, 0)

    n_leafs =  ceil(Integer, n_p / leafsize)
    n_internal_nodes = n_leafs - 1
    leafrow = floor(Integer, log2(n_leafs))
    cross_node = 2^(leafrow + 1)
    last_node_size = n_p % leafsize
    if last_node_size == 0
        last_node_size = leafsize
    end

    # This only happens when n_p / leafsize is a power of 2?
    if cross_node >= n_internal_nodes + n_leafs
        cross_node = div(cross_node, 2)
    end

    offset = 2(n_leafs - 2^leafrow) - 1
    k1 = (offset - n_internal_nodes - 1) * leafsize + last_node_size + 1
    k2 = -cross_node * leafsize + 1
    last_full_node = n_leafs + n_internal_nodes

    TreeData(last_node_size, leafsize, n_leafs,
    n_internal_nodes, cross_node, k1, k2, last_full_node)
end
