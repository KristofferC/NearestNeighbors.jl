"""
    TreeNode{TreeT<:NNTree}
Lightweight handle that identifies a node inside `tree`. Callers obtain these
handles via helpers such as `treeroot`, `TreeWalk`, or `children`
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


AbstractTrees.isroot(node::TreeNode) = _treeindex(node) == 1

# TreeNode-specific constructor for LeafPointView
function LeafPointView(node::TreeNode)
    if !_isleaf(node)
        throw(ArgumentError("node $(_treeindex(node)) is not a leaf"))
    end
    tree = _nodetree(node)
    return LeafPointView(tree, get_leaf_range(tree.tree_data, _treeindex(node)))
end

"""
    leafpoints(node::TreeNode) -> LeafPointView

Return a view over the stored points in the leaf `node`.  Throws an error if
`node` is not a leaf or if the tree was created with `storedata = false`.
"""
function leafpoints(node::TreeNode)
    tree = _nodetree(node)
    isempty(tree.data) &&
        throw(ArgumentError("tree was constructed without storing point data"))
    return LeafPointView(node)
end

"""
    leaf_point_indices(node::TreeNode; original::Bool = true)

Return the indices of the points owned by `node`.  When `original == true` the
result references the original, unreordered dataset; otherwise it returns the
contiguous block from the tree's internal ordering.
"""
function leaf_point_indices(node::TreeNode; original::Bool = true)
    _isleaf(node) || throw(ArgumentError("node $(_treeindex(node)) is not a leaf"))
    tree = _nodetree(node)
    idx = _treeindex(node)
    range = get_leaf_range(tree.tree_data, idx)
    return original ? view(tree.indices, range) : range
end

"""
    treeregion(node::TreeNode)

Return the geometric region attached to `node`. `BallTree` nodes expose
`HyperSphere`s while `KDTree` nodes yield `HyperRectangle`s.
For KDTree leaf nodes, the rectangle is reconstructed from the parent's split.
"""
function treeregion(node::TreeNode)
    return _compute_treeregion(_nodetree(node), _treeindex(node), _isleaf(node))
end


# AbstractTrees.jl interface
import AbstractTrees

AbstractTrees.children(node::TreeNode) = children(node)
AbstractTrees.parent(node::TreeNode) = parent(node)

# Define traits for type-stable iteration
AbstractTrees.NodeType(::Type{<:TreeNode}) = AbstractTrees.HasNodeType()
AbstractTrees.nodetype(::Type{<:TreeNode{T}}) where {T} = TreeNode{T}
AbstractTrees.ParentLinks(::Type{<:TreeNode}) = AbstractTrees.StoredParents()
AbstractTrees.SiblingLinks(::Type{<:TreeNode}) = AbstractTrees.ImplicitSiblings()
AbstractTrees.ChildIndexing(::Type{<:TreeNode}) = AbstractTrees.IndexedChildren()

# Type-stable iteration hints for better performance
Base.IteratorEltype(::Type{<:AbstractTrees.TreeIterator{<:TreeNode{T}}}) where {T} = Base.HasEltype()
Base.eltype(::Type{<:AbstractTrees.TreeIterator{<:TreeNode{T}}}) where {T} = TreeNode{T}

# Custom tree printing for better visualization
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

# Pretty printing for TreeNode in REPL
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

# Custom tree walkers (optimized implementations)
# These avoid AbstractTrees overhead and use specialized stack-based iteration

struct PreOrderWalkerCustom{TreeT<:NNTree}
    root::TreeNode{TreeT}
end

preorder_custom(tree::NNTree) = PreOrderWalkerCustom(treeroot(tree))

function _preorder_custom_initial_state(walker::PreOrderWalkerCustom)
    tree = _nodetree(walker.root)
    capacity = _walker_capacity(tree)
    stack = Int[_treeindex(walker.root)]
    sizehint!(stack, capacity)
    return tree, stack
end

@inline function Base.iterate(walker::PreOrderWalkerCustom, state = _preorder_custom_initial_state(walker))
    tree, stack = state
    isempty(stack) && return nothing
    idx = pop!(stack)

    if !isleaf(_tree_data(tree).n_internal_nodes, idx)
        push!(stack, getright(idx))
        push!(stack, getleft(idx))
    end

    return TreeNode(tree, idx), state
end

Base.IteratorSize(::Type{<:PreOrderWalkerCustom}) = Base.HasLength()
Base.length(walker::PreOrderWalkerCustom) =
    _nodetree(walker.root).tree_data.n_internal_nodes + _nodetree(walker.root).tree_data.n_leafs
Base.eltype(::Type{PreOrderWalkerCustom{TreeT}}) where TreeT = TreeNode{TreeT}


struct LeafWalkerCustom{TreeT<:NNTree}
    tree::TreeT
    leaf_range::UnitRange{Int}
end

function leaves_custom(tree::NNTree)
    return LeafWalkerCustom(tree, _get_leaf_range(tree))
end

@inline function Base.iterate(walker::LeafWalkerCustom)
    return iterate(walker, first(walker.leaf_range))
end

@inline function Base.iterate(walker::LeafWalkerCustom, idx::Int)
    idx > last(walker.leaf_range) && return nothing
    return TreeNode(walker.tree, idx), idx + 1
end

Base.IteratorSize(::Type{<:LeafWalkerCustom}) = Base.HasLength()
Base.length(walker::LeafWalkerCustom) = length(walker.leaf_range)
Base.eltype(::Type{LeafWalkerCustom{TreeT}}) where TreeT = TreeNode{TreeT}


struct PostOrderWalkerCustom{TreeT<:NNTree}
    root::TreeNode{TreeT}
end

postorder_custom(tree::NNTree) = PostOrderWalkerCustom(treeroot(tree))

mutable struct PostOrderStateCustom{TreeT}
    tree::TreeT
    stack::Vector{Int}
    last_visited::Int
end

function _postorder_custom_initial_state(walker::PostOrderWalkerCustom{TreeT}) where TreeT
    tree = _nodetree(walker.root)
    capacity = _walker_capacity(tree)
    stack = sizehint!(Int[_treeindex(walker.root)], capacity)
    return PostOrderStateCustom(tree, stack, 0)
end

function Base.iterate(walker::PostOrderWalkerCustom{TreeT}, state::PostOrderStateCustom{TreeT} = _postorder_custom_initial_state(walker)) where TreeT
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

Base.IteratorSize(::Type{<:PostOrderWalkerCustom}) = Base.HasLength()
Base.length(walker::PostOrderWalkerCustom) =
    _nodetree(walker.root).tree_data.n_internal_nodes + _nodetree(walker.root).tree_data.n_leafs
Base.eltype(::Type{PostOrderWalkerCustom{TreeT}}) where TreeT = TreeNode{TreeT}
