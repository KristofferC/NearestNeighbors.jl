immutable TreeData
    last_node_size::Int
    leaf_size::Int
    n_leafs::Int
    n_internal_nodes::Int
    cross_node::Int
    first_leaf_row::Int
    offset::Int
end

function TreeData(data, leaf_size)
    n_dim, n_p = size(data)
    n_leafs =  ceil(Integer, n_p / leaf_size)
    n_internal_nodes = n_leafs - 1
    l = floor(Integer, log2(n_leafs))
    offset = 2(n_leafs - 2^l) - 1
    cross_node = 2^(l+1)
    last_node_size = n_p % leaf_size
    if last_node_size == 0
        last_node_size = leaf_size
    end

     # This only happens when n_p / leaf_size is a power of 2?
    if cross_node >= n_internal_nodes + n_leafs
        cross_node = div(cross_node, 2)
    end

    TreeData(last_node_size, leaf_size, n_leafs,
    n_internal_nodes, cross_node, l,
    offset)
end
