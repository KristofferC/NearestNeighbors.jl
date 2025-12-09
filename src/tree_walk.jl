# Tree walking with fat nodes (TreeNode bundles tree + index)
# Provides both AbstractTrees interface and custom performance-optimized walkers

const TREE_WALK_UNSUPPORTED = "tree walking is only supported for KDTree and BallTree"
const TREE_WALK_EMPTY = "tree does not contain any nodes"

@inline _tree_data(tree::BallTree) = tree.tree_data
@inline _tree_data(tree::KDTree) = tree.tree_data
@inline function _tree_data(tree::NNTree)
    throw(ArgumentError(TREE_WALK_UNSUPPORTED))
end

function _get_leaf_range(tree::NNTree)
    tree_data = _tree_data(tree)
    leaf_start = tree_data.n_internal_nodes + 1
    leaf_end = tree_data.n_internal_nodes + tree_data.n_leafs
    return leaf_start:leaf_end
end

function _walker_capacity(tree::NNTree)
    n_nodes = tree.tree_data.n_internal_nodes + tree.tree_data.n_leafs
    return max(64, ceil(Int, log2(n_nodes + 1)) * 2)
end

"""
    LeafPointView(tree, range)

Zero-copy view over the points stored in a leaf node.
"""
struct LeafPointView{TreeT<:NNTree}
    tree::TreeT
    range::UnitRange{Int}
end

Base.IndexStyle(::Type{<:LeafPointView}) = IndexLinear()
Base.length(view::LeafPointView) = length(view.range)
Base.size(view::LeafPointView) = (length(view),)
Base.axes(view::LeafPointView) = (Base.OneTo(length(view)),)
Base.eltype(view::LeafPointView{TreeT}) where {TreeT} = eltype(view.tree.data)

function Base.getindex(view::LeafPointView, i::Int)
    firstidx = first(view.range)
    lastidx = last(view.range)
    idx = firstidx + i - 1
    (idx < firstidx || idx > lastidx) && throw(BoundsError(view, i))
    storage_idx = view.tree.reordered ? idx : view.tree.indices[idx]
    return view.tree.data[storage_idx]
end

function Base.iterate(view::LeafPointView, state::Int=0)
    isempty(view.range) && return nothing
    idx = (state == 0) ? first(view.range) : state
    idx > last(view.range) && return nothing
    storage_idx = view.tree.reordered ? idx : view.tree.indices[idx]
    return (view.tree.data[storage_idx], idx + 1)
end

function _compute_treeregion(tree::Union{BallTree, KDTree}, idx::Int, is_leaf::Bool)
    if tree isa BallTree
        return tree.hyper_spheres[idx]
    elseif tree isa KDTree
        if is_leaf
            if idx == 1
                return tree.hyper_rec
            end
            parent_idx = getparent(idx)
            parent_rect = tree.hyper_rects[parent_idx]
            split_dim = tree.split_dims[parent_idx]
            split_val = tree.split_vals[parent_idx]
            left_rect, right_rect = split_hyperrectangle(parent_rect, split_dim, split_val)
            return idx == getleft(parent_idx) ? left_rect : right_rect
        else
            return tree.hyper_rects[idx]
        end
    else
        throw(ArgumentError("tree regions only defined for BallTree and KDTree"))
    end
end

# ============================================================================
# TreeNode: Fat node that bundles tree reference with node index
# ============================================================================

"""
    TreeNode{TreeT<:NNTree}

Lightweight handle that identifies a node inside `tree`. Bundles the tree
reference with the node index for convenient access. Obtain these handles
via `treeroot`, custom walkers, or `children`.
"""
struct TreeNode{TreeT<:NNTree}
    tree::TreeT
    index::Int
end

@inline _nodetree(node::TreeNode) = getfield(node, :tree)
@inline _treeindex(node::TreeNode) = getfield(node, :index)

function treeroot(tree::Union{KDTree, BallTree})
    _tree_data(tree).n_leafs == 0 && throw(ArgumentError(TREE_WALK_EMPTY))
    return TreeNode(tree, 1)
end

treeroot(tree::NNTree) = throw(ArgumentError(TREE_WALK_UNSUPPORTED))

function parent(node::TreeNode)
    idx = _treeindex(node)
    idx == 1 && return nothing
    return TreeNode(_nodetree(node), getparent(idx))
end

function children(node::TreeNode)
    _isleaf(node) && return ()
    tree = _nodetree(node)
    idx = _treeindex(node)
    left = TreeNode(tree, getleft(idx))
    right = TreeNode(tree, getright(idx))
    return (left, right)
end

_isleaf(node::TreeNode) = isleaf(_nodetree(node).tree_data.n_internal_nodes, _treeindex(node))

function LeafPointView(node::TreeNode)
    if !_isleaf(node)
        throw(ArgumentError("node $(_treeindex(node)) is not a leaf"))
    end
    tree = _nodetree(node)
    return LeafPointView(tree, get_leaf_range(tree.tree_data, _treeindex(node)))
end

"""
    leafpoints(node::TreeNode) -> LeafPointView

Return a view over the stored points in the leaf `node`. Throws an error if
`node` is not a leaf or if the tree was created with `storedata = false`.
"""
function leafpoints(node::TreeNode)
    tree = _nodetree(node)
    isempty(tree.data) &&
        throw(ArgumentError("tree was constructed without storing point data"))
    return LeafPointView(node)
end

"""
    leaf_point_indices(node::TreeNode)

Return the indices of the points owned by `node`.
"""
function leaf_point_indices(node::TreeNode)
    _isleaf(node) || throw(ArgumentError("node $(_treeindex(node)) is not a leaf"))
    tree = _nodetree(node)
    idx = _treeindex(node)
    range = get_leaf_range(tree.tree_data, idx)
    return view(tree.indices, range)
end

"""
    treeregion(node::TreeNode)

Return the geometric region attached to `node`. `BallTree` nodes expose
`HyperSphere`s while `KDTree` nodes yield `HyperRectangle`s.
"""
function treeregion(node::TreeNode)
    return _compute_treeregion(_nodetree(node), _treeindex(node), _isleaf(node))
end

# ============================================================================
# AbstractTrees.jl interface
# ============================================================================

import AbstractTrees

AbstractTrees.isroot(node::TreeNode) = _treeindex(node) == 1
AbstractTrees.children(node::TreeNode) = children(node)
AbstractTrees.parent(node::TreeNode) = parent(node)

AbstractTrees.NodeType(::Type{<:TreeNode}) = AbstractTrees.HasNodeType()
AbstractTrees.nodetype(::Type{<:TreeNode{T}}) where {T} = TreeNode{T}
AbstractTrees.ParentLinks(::Type{<:TreeNode}) = AbstractTrees.StoredParents()
AbstractTrees.SiblingLinks(::Type{<:TreeNode}) = AbstractTrees.ImplicitSiblings()
AbstractTrees.ChildIndexing(::Type{<:TreeNode}) = AbstractTrees.IndexedChildren()

Base.IteratorEltype(::Type{<:AbstractTrees.TreeIterator{<:TreeNode{T}}}) where {T} = Base.HasEltype()
Base.eltype(::Type{<:AbstractTrees.TreeIterator{<:TreeNode{T}}}) where {T} = TreeNode{T}

function AbstractTrees.printnode(io::IO, node::TreeNode)
    tree = _nodetree(node)
    idx = _treeindex(node)
    if _isleaf(node)
        n_points = length(get_leaf_range(tree.tree_data, idx))
        print(io, "Leaf(", n_points, " pts)")
    elseif tree isa BallTree
        sphere = treeregion(node)
        print(io, "Ball(r=", round(sphere.r, digits=3), ")")
    elseif tree isa KDTree
        dim = Int(tree.split_dims[idx])
        val = tree.split_vals[idx]
        print(io, "Split(dim=", dim, ", val=", round(val, digits=3), ")")
    end
end

function Base.show(io::IO, node::TreeNode)
    tree = _nodetree(node)
    tree_type = tree isa BallTree ? "BallTree" : tree isa KDTree ? "KDTree" : "Unknown"
    print(io, "TreeNode{", tree_type, "}(")
    if _isleaf(node)
        n_points = length(get_leaf_range(tree.tree_data, _treeindex(node)))
        print(io, "leaf, ", n_points, " points")
    else
        print(io, "internal")
    end
    print(io, ")")
end

# ============================================================================
# Custom walkers (performance-optimized, avoid AbstractTrees overhead)
# ============================================================================

struct PreOrderWalk{TreeT<:NNTree}
    root::TreeNode{TreeT}
end

"""
    preorder(tree) -> PreOrderWalk

Return an iterator that visits all nodes in pre-order (parent before children).
More efficient than `AbstractTrees.PreOrderDFS` for performance-critical code.
"""
preorder(tree::NNTree) = PreOrderWalk(treeroot(tree))

function _preorder_initial_state(walker::PreOrderWalk)
    tree = _nodetree(walker.root)
    capacity = _walker_capacity(tree)
    stack = Int[_treeindex(walker.root)]
    sizehint!(stack, capacity)
    return tree, stack
end

@inline function Base.iterate(walker::PreOrderWalk, state = _preorder_initial_state(walker))
    tree, stack = state
    isempty(stack) && return nothing
    idx = pop!(stack)

    if !isleaf(_tree_data(tree).n_internal_nodes, idx)
        push!(stack, getright(idx))
        push!(stack, getleft(idx))
    end

    return TreeNode(tree, idx), state
end

Base.IteratorSize(::Type{<:PreOrderWalk}) = Base.HasLength()
Base.length(walker::PreOrderWalk) =
    _nodetree(walker.root).tree_data.n_internal_nodes + _nodetree(walker.root).tree_data.n_leafs
Base.eltype(::Type{PreOrderWalk{TreeT}}) where TreeT = TreeNode{TreeT}


struct PostOrderWalk{TreeT<:NNTree}
    root::TreeNode{TreeT}
end

"""
    postorder(tree) -> PostOrderWalk

Return an iterator that visits all nodes in post-order (children before parent).
More efficient than `AbstractTrees.PostOrderDFS` for performance-critical code.
"""
postorder(tree::NNTree) = PostOrderWalk(treeroot(tree))

mutable struct PostOrderState{TreeT}
    tree::TreeT
    stack::Vector{Int}
    last_visited::Int
end

function _postorder_initial_state(walker::PostOrderWalk{TreeT}) where TreeT
    tree = _nodetree(walker.root)
    capacity = _walker_capacity(tree)
    stack = sizehint!(Int[_treeindex(walker.root)], capacity)
    return PostOrderState(tree, stack, 0)
end

function Base.iterate(walker::PostOrderWalk{TreeT}, state::PostOrderState{TreeT} = _postorder_initial_state(walker)) where TreeT
    while !isempty(state.stack)
        idx = state.stack[end]

        if isleaf(_tree_data(state.tree).n_internal_nodes, idx)
            pop!(state.stack)
            state.last_visited = idx
            return TreeNode(state.tree, idx), state
        end

        left_idx = getleft(idx)
        if state.last_visited != left_idx && state.last_visited != getright(idx)
            push!(state.stack, getright(idx))
            push!(state.stack, left_idx)
        else
            pop!(state.stack)
            state.last_visited = idx
            return TreeNode(state.tree, idx), state
        end
    end
    return nothing
end

Base.IteratorSize(::Type{<:PostOrderWalk}) = Base.HasLength()
Base.length(walker::PostOrderWalk) =
    _nodetree(walker.root).tree_data.n_internal_nodes + _nodetree(walker.root).tree_data.n_leafs
Base.eltype(::Type{PostOrderWalk{TreeT}}) where TreeT = TreeNode{TreeT}


struct LeafWalk{TreeT<:NNTree}
    tree::TreeT
    leaf_range::UnitRange{Int}
end

"""
    leaves(tree) -> LeafWalk

Return an iterator over all leaf nodes. This is the most efficient way to
iterate only over leaves.
"""
function leaves(tree::NNTree)
    return LeafWalk(tree, _get_leaf_range(tree))
end

@inline function Base.iterate(walker::LeafWalk)
    return iterate(walker, first(walker.leaf_range))
end

@inline function Base.iterate(walker::LeafWalk, idx::Int)
    idx > last(walker.leaf_range) && return nothing
    return TreeNode(walker.tree, idx), idx + 1
end

Base.IteratorSize(::Type{<:LeafWalk}) = Base.HasLength()
Base.length(walker::LeafWalk) = length(walker.leaf_range)
Base.eltype(::Type{LeafWalk{TreeT}}) where TreeT = TreeNode{TreeT}
