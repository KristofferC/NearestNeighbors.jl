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

struct NNTreeNode{R}
    index::Int
    region::R
end 

# Show the info associated with the node. 
function show(io::IO, node::NNTreeNode)
    #println(io, typeof(tree(node)))
    print(io, "NNTreeNode: treeindex:",  (treeindex(node)), 
        " region: ", region(node))
    #println(io, "  region: ", region(node))
end 

"""
    treeindex(node) 

This returns the index of the given node. The indices of nodes are an 
implementation specific feature but are externally useful to 
associate metadata with nodes within the search tree. 
The range of indices is given by `eachindex(node)`. 
Nodes can be outside the range if they are leaf nodes. 

## Example
```julia
function walktree(tree, node) 
    println("Node index: ", treeindex(node), " and isleaf:", isleaf(tree, node) )
    if !isleaf(tree, node)
        walktree.(tree, children(node))
    end 
end 
using StableRNGs, GeometryBasics, NearestNeighbors 
T = KDTree(rand(StableRNG(1), Point2f, 25))
println("eachtreeindex: ", eachtreeindex(root(T)))
walktree(tree, root(T)) 
```

## See Also
[`eachindex`](@ref)
"""
@inline treeindex(node::NNTreeNode) = node.index 

"""
    eachtreeindex(node)

Get the full range of indices associated with the nodes of the search
tree, this only depends on the tree the node is associated with, so all
nodes of that tree will return the same thing. The index range only
corresponds to the internal nodes of the tree. 

## See Also
[`treeindex`](@ref)
"""
@inline eachtreeindex(tree::NNTree) = 1:tree.tree_data.n_internal_nodes

"""
    isleaf(tree, node)

Return true if the node is a leaf node of a tree. 
"""    
@inline isleaf(tree::NNTree, node::NNTreeNode) = isleaf(tree.tree_data.n_internal_nodes, treeindex(node))

"""
    isroot(tree, node)

Return true if the node is a root node of the tree. 
"""
@inline function isroot(_, node)
    return node.index == 1
end 

"""
    region(node)

Return the region of space associated with a node in the tree.
"""    
@inline region(node::NNTreeNode) = node.region 

"""
    children(tree, node)

Return the children of a given node in the tree. 
This throws an BoundsError if the node is a leaf. 
"""
@inline function children(tree::NNTree, node::NNTreeNode)
    if isleaf(tree, node)
        throw(ArgumentError("Cannot call children on leaf nodes"))
    end 
    i = treeindex(node) 
    r1, r2 = _split_regions(tree, region(node), i) 
    i1, i2 = getleft(i), getright(i)
    return (
        NNTreeNode(i1, r1),
        NNTreeNode(i2, r2) 
    )
end 

@inline function parent(tree::NNTree, node::NNTreeNode)
    i = treeindex(node) 
    p = getparent(i)
    if p == 0 
        throw(ArgumentError("Cannot call parent on the root node"))
    end 
    r = _parent_region(tree, region(node), i)
    return (
        NNTreeNode(p, r)
    )
end 

@inline function nextsibling(tree, node::NNTreeNode)
    if isroot(tree, node)
        return nothing
    else
        p = parent(tree, node)
        l, r = children(tree, p)
        if node == l 
            return r
        else
            return nothing
        end
    end
end 

@inline function prevsibling(tree, node::NNTreeNode)
    if isroot(tree, node)
        return nothing
    else
        p = parent(tree, node)
        l, r = children(tree, p)
        if node == r
            return l
        else
            return nothing
        end
    end
end 

"""
This function enables one to disable region computation by providing
a nothing type for that. it'll just omit the region computation
entirely. 

You can enable it with 
skip_regions(root) 
""" 
_split_regions(_::NNTree, r::Nothing, _) = nothing, nothing 
_parent_region(_::NNTree, r::Nothing, _) = nothing 

""" 
    skip_regions(node)

Sometimes all you need to navigate the nearest neighbor tree is 
the tree structure itself and not the regions associated with each 
node. In some cases, computing the regions can be expensive. So 
this call sets regions to `nothing` which propagates throughout 
the tree and simply elides the region computations. 

## Example
```julia 
using BenchmarkTools, StableRNGs, GeometryBasics
function count_points(tree, node)
    count = 0 
    if NearestNeighbors.isleaf(tree, node)
      count += length(NearestNeighbors.points_indices(tree, node))
    else 
      left, right = NearestNeighbors.children(tree, node)
      count += count_points(tree, left)
      count += count_points(tree, right)
    end 
    return count 
  end 
end   
pts = rand(StableRNG(1), Point2f, 1_000_000)
T = KDTree(pts)
@btime count_points(T, root(T))
@btime count_points(T, skip_regions(root(T))
```
"""    
@inline skip_regions(node::NNTreeNode) = NNTreeNode(treeindex(node), nothing)


"""
    root(T::NNTree)

Return the root node of the nearest neighbor search tree. 
"""    
function root(T::NNTree)
    return NNTreeNode(1, region(T))
end

function _points(tree_data, data, index, indices, reordered)
    if reordered 
        return (data[idx] for idx in get_leaf_range(tree_data, index))
    else
        return (data[indices[idx]] for idx in get_leaf_range(tree_data, index))
    end 
end 

"""
    points(tree, node)

Create an iterator for all the points contained within the
    node of the nearest neighbor tree. 
"""
function points(T::NNTree, N::NNTreeNode)
    return PointsIterator(T, skip_regions(N))
end 

struct PointsIterator{Tree <: NNTree, N <: NNTreeNode}
    tree::Tree
    start::N
end 

function _find_first_leaf(tree, node)
    while true
        if isleaf(tree, node)
            return node
        else
            node = children(tree, node)[1] # get the left child 
        end
    end
end

function _find_next_leaf(tree, node, stopindex)
    if node.index == stopindex 
        return nothing
    end 
    next = nextsibling(tree, node) 
    while next === nothing
        if isroot(tree, node) 
            return nothing # this is the end condition... 
        end 
        node = parent(tree, node)
        if node.index == stopindex 
            return nothing 
        end 

        next = nextsibling(tree, node) 
    end
    # now we need to find the leaf from this node...
    leaf = _find_first_leaf(tree, next)
    return leaf 
end 

function _next_leaf_and_leaf_iterate(tree, node, stopindex) 
    while true 
        node = _find_next_leaf(tree, node, stopindex)
        if node === nothing 
            return nothing 
        end 
        leafrange = get_leaf_range(tree.tree_data, treeindex(node))
        next = iterate(leafrange)
        if next !== nothing
            return next, leafrange, node 
        end
    end
end 

function _iterate(leafrangenext, leafrange, node, it)
    if leafrangenext === nothing 
        next = _next_leaf_and_leaf_iterate(it.tree, node, treeindex(it.start)) 
    
        if next === nothing 
            # if next is still nothing, then we are an empty tree
            return nothing
        end 

        leafrangenext, leafrange, node = next 
    end 

    idx, leafrangestate = leafrangenext 
    reordered = it.tree.reordered 
    if reordered
        pt = it.tree.data[idx]
    else
        pt = it.tree.data[it.tree.indices[idx]]
    end

    return pt, (leafrangestate, leafrange, node) 
end 

import Base.iterate, Base.IteratorSize, Base.eltype 

IteratorSize(_::PointsIterator) = Base.SizeUnknown()
eltype(it::PointsIterator) = eltype(it.tree.data)

function iterate(it::PointsIterator)
    node = _find_first_leaf(it.tree, it.start)
    leafrange = get_leaf_range(it.tree.tree_data, treeindex(node))
    leafrangenext = iterate(leafrange) 

    return _iterate(leafrangenext, leafrange, node, it)
end 
function iterate(it::PointsIterator, state)
    leafrangestate, leafrange, node = state
    leafrangenext = iterate(leafrange, leafrangestate)
    return _iterate(leafrangenext, leafrange, node, it)
end


function leafpoints(T::NNTree, node::NNTreeNode)
    # T = tree(node)
    # redirect to possibly specialize 
    return _points(T.tree_data, T.data, treeindex(node), T.indices, T.reordered)
end 

function leaf_points_indices(T::NNTree, node::NNTreeNode)
    # T = tree(node)
    tree_data = T.tree_data
    indices = T.indices
    return (indices[idx] for idx in get_leaf_range(tree_data, treeindex(node)))
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

# Checks the distance function and add those points that are among the k best.
# Uses a heap for fast insertion.
@inline function add_points_knn!(best_dists::AbstractVector, best_idxs::AbstractVector{<:Integer},
                                 tree::NNTree, index::Int, point::AbstractVector,
                                 do_end::Bool, skip::F) where {F}
    for z in get_leaf_range(tree.tree_data, index)
        idx = tree.reordered ? z : tree.indices[z]
        dist_d = evaluate_maybe_end(tree.metric, tree.data[idx], point, do_end)
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
@inline function add_points_inrange!(idx_in_ball::Union{Nothing, AbstractVector{<:Integer}}, tree::NNTree,
                                     index::Int, point::AbstractVector, r::Number, do_end::Bool)
    count = 0
    for z in get_leaf_range(tree.tree_data, index)
        idx = tree.reordered ? z : tree.indices[z]
        dist_d = evaluate_maybe_end(tree.metric, tree.data[idx], point, do_end)
        if dist_d <= r
            count += 1
            idx_in_ball !== nothing && push!(idx_in_ball, idx)
        end
    end
    return count
end

# Add all points in this subtree since we have determined
# they are all within the desired range
function addall(tree::NNTree, index::Int, idx_in_ball::Union{Nothing, Vector{<:Integer}})
    tree_data = tree.tree_data
    count = 0
    if isleaf(tree_data.n_internal_nodes, index)
        for z in get_leaf_range(tree_data, index)
            idx = tree.reordered ? z : tree.indices[z]
            count += 1
            idx_in_ball !== nothing && push!(idx_in_ball, idx)
        end
    else
        count += addall(tree, getleft(index), idx_in_ball)
        count += addall(tree, getright(index), idx_in_ball)
    end
    return count
end

# Add all points in this subtree since we have determined
# they are all within the desired range
function addall(tree::NNTree, node::NNTreeNode, idx_in_ball::Union{Nothing, Vector{<:Integer}})
    # tree = NearestNeighbors.tree(node)
    tree_data = tree.tree_data
    count = 0
    index = node.index 
    if isleaf(tree, node)
        for z in get_leaf_range(tree_data, index)
            idx = tree.reordered ? z : tree.indices[z]
            count += 1
            idx_in_ball !== nothing && push!(idx_in_ball, idx)
        end
    else
        left, right = children(tree, node) 
        count += addall(tree, left, idx_in_ball)
        count += addall(tree, right, idx_in_ball)
    end
    return count
end
