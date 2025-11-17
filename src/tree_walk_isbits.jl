# Tree walking with isbits nodes (index-only)
# Two approaches: plain isbits struct + custom walkers, and AbstractTrees IndexNode

# ============================================================================
# Approach 1: Plain isbits node with custom walkers
# ============================================================================

struct TreeNodeIsbits
    index::Int
end

function treeroot_isbits(tree::Union{KDTree, BallTree})
    _tree_data(tree).n_leafs == 0 && throw(ArgumentError(TREE_WALK_EMPTY))
    return TreeNodeIsbits(1)
end

treeroot_isbits(tree::NNTree) = throw(ArgumentError(TREE_WALK_UNSUPPORTED))

function children_isbits(tree::Union{BallTree, KDTree}, node::TreeNodeIsbits)
    _isleaf_isbits(tree, node) && return ()
    idx = node.index
    return (TreeNodeIsbits(getleft(idx)), TreeNodeIsbits(getright(idx)))
end

function parent_isbits(tree::Union{BallTree, KDTree}, node::TreeNodeIsbits)
    idx = node.index
    idx == 1 && return nothing
    return TreeNodeIsbits(getparent(idx))
end

_isleaf_isbits(tree::NNTree, node::TreeNodeIsbits) =
    isleaf(_tree_data(tree).n_internal_nodes, node.index)

function treeregion_isbits(tree::Union{BallTree, KDTree}, node::TreeNodeIsbits)
    return _compute_treeregion(tree, node.index, _isleaf_isbits(tree, node))
end

function leafpoints_isbits(tree::NNTree, node::TreeNodeIsbits)
    if !_isleaf_isbits(tree, node)
        throw(ArgumentError("node $(node.index) is not a leaf"))
    end
    isempty(tree.data) &&
        throw(ArgumentError("tree was constructed without storing point data"))
    return LeafPointView(tree, get_leaf_range(_tree_data(tree), node.index))
end

function leaf_point_indices_isbits(tree::NNTree, node::TreeNodeIsbits; original::Bool = true)
    _isleaf_isbits(tree, node) || throw(ArgumentError("node $(node.index) is not a leaf"))
    idx = node.index
    range = get_leaf_range(_tree_data(tree), idx)
    return original ? view(tree.indices, range) : range
end

# Custom walkers for isbits nodes

struct PreOrderWalkerIsbits{TreeT<:NNTree}
    tree::TreeT
    root::TreeNodeIsbits
end

preorder_isbits(tree::NNTree) = PreOrderWalkerIsbits(tree, treeroot_isbits(tree))

function _preorder_isbits_initial_state(walker::PreOrderWalkerIsbits)
    capacity = _walker_capacity(walker.tree)
    stack = TreeNodeIsbits[walker.root]
    sizehint!(stack, capacity)
    return stack
end

@inline function Base.iterate(walker::PreOrderWalkerIsbits, stack::Vector{TreeNodeIsbits} = _preorder_isbits_initial_state(walker))
    isempty(stack) && return nothing
    node = pop!(stack)

    if !_isleaf_isbits(walker.tree, node)
        idx = node.index
        push!(stack, TreeNodeIsbits(getright(idx)))
        push!(stack, TreeNodeIsbits(getleft(idx)))
    end

    return node, stack
end

Base.IteratorSize(::Type{<:PreOrderWalkerIsbits}) = Base.HasLength()
Base.length(walker::PreOrderWalkerIsbits) =
    walker.tree.tree_data.n_internal_nodes + walker.tree.tree_data.n_leafs
Base.eltype(::Type{<:PreOrderWalkerIsbits}) = TreeNodeIsbits


struct LeafWalkerIsbits{TreeT<:NNTree}
    tree::TreeT
    leaf_range::UnitRange{Int}
end

function leaves_isbits(tree::NNTree)
    return LeafWalkerIsbits(tree, _get_leaf_range(tree))
end

@inline function Base.iterate(walker::LeafWalkerIsbits)
    return iterate(walker, first(walker.leaf_range))
end

@inline function Base.iterate(walker::LeafWalkerIsbits, idx::Int)
    idx > last(walker.leaf_range) && return nothing
    return TreeNodeIsbits(idx), idx + 1
end

Base.IteratorSize(::Type{<:LeafWalkerIsbits}) = Base.HasLength()
Base.length(walker::LeafWalkerIsbits) = length(walker.leaf_range)
Base.eltype(::Type{<:LeafWalkerIsbits}) = TreeNodeIsbits


struct PostOrderWalkerIsbits{TreeT<:NNTree}
    tree::TreeT
    root::TreeNodeIsbits
end

postorder_isbits(tree::NNTree) = PostOrderWalkerIsbits(tree, treeroot_isbits(tree))

mutable struct PostOrderStateIsbits
    stack::Vector{TreeNodeIsbits}
    last_visited::Int
end

function _postorder_isbits_initial_state(walker::PostOrderWalkerIsbits)
    capacity = _walker_capacity(walker.tree)
    stack = sizehint!(TreeNodeIsbits[walker.root], capacity)
    return PostOrderStateIsbits(stack, 0)
end

function Base.iterate(walker::PostOrderWalkerIsbits, state::PostOrderStateIsbits = _postorder_isbits_initial_state(walker))
    while !isempty(state.stack)
        node = state.stack[end]
        idx = node.index

        if _isleaf_isbits(walker.tree, node)
            pop!(state.stack)
            state.last_visited = idx
            return node, state
        end

        left_idx = getleft(idx)
        if state.last_visited != left_idx && state.last_visited != getright(idx)
            push!(state.stack, TreeNodeIsbits(getright(idx)))
            push!(state.stack, TreeNodeIsbits(left_idx))
        else
            pop!(state.stack)
            state.last_visited = idx
            return node, state
        end
    end
    return nothing
end

Base.IteratorSize(::Type{<:PostOrderWalkerIsbits}) = Base.HasLength()
Base.length(walker::PostOrderWalkerIsbits) =
    walker.tree.tree_data.n_internal_nodes + walker.tree.tree_data.n_leafs
Base.eltype(::Type{<:PostOrderWalkerIsbits}) = TreeNodeIsbits

Base.show(io::IO, node::TreeNodeIsbits) = print(io, "TreeNodeIsbits(", node.index, ")")


# ============================================================================
# Approach 2: AbstractTrees IndexNode interface
# ============================================================================

using AbstractTrees: IndexNode

# Indexed tree interface implementation
function AbstractTrees.childindices(tree::Union{BallTree, KDTree}, idx::Int)
    isleaf(_tree_data(tree).n_internal_nodes, idx) && return ()
    return (getleft(idx), getright(idx))
end

function AbstractTrees.parentindex(tree::Union{BallTree, KDTree}, idx::Int)
    idx == 1 && return nothing
    return getparent(idx)
end

AbstractTrees.nodevalue(tree::Union{BallTree, KDTree}, idx::Int) = idx

function AbstractTrees.rootindex(tree::Union{BallTree, KDTree})
    _tree_data(tree).n_leafs == 0 && throw(ArgumentError(TREE_WALK_EMPTY))
    return 1
end

# Convenience constructor
treeroot_indexnode(tree::Union{BallTree, KDTree}) = IndexNode(tree, AbstractTrees.rootindex(tree))
treeroot_indexnode(tree::NNTree) = throw(ArgumentError(TREE_WALK_UNSUPPORTED))

# Define traits for AbstractTrees integration
AbstractTrees.ParentLinks(::Type{<:Union{BallTree, KDTree}}) = AbstractTrees.StoredParents()
AbstractTrees.SiblingLinks(::Type{<:Union{BallTree, KDTree}}) = AbstractTrees.ImplicitSiblings()
AbstractTrees.ChildIndexing(::Type{<:Union{BallTree, KDTree}}) = AbstractTrees.IndexedChildren()
AbstractTrees.NodeType(::Type{IndexNode{T,Int}}) where {T<:Union{BallTree,KDTree}} = AbstractTrees.HasNodeType()
AbstractTrees.nodetype(::Type{IndexNode{T,Int}}) where {T<:Union{BallTree,KDTree}} = IndexNode{T,Int}

# Helper accessors
_isleaf_indexnode(node::IndexNode{<:NNTree,Int}) =
    isleaf(_tree_data(node.tree).n_internal_nodes, node.index)

_treeindex_indexnode(node::IndexNode) = node.index
_nodetree_indexnode(node::IndexNode) = node.tree

children_indexnode(node::IndexNode) = AbstractTrees.children(node)

function treeregion_indexnode(node::IndexNode{<:Union{BallTree,KDTree},Int})
    return _compute_treeregion(node.tree, node.index, _isleaf_indexnode(node))
end

function leafpoints_indexnode(node::IndexNode{<:NNTree,Int})
    if !_isleaf_indexnode(node)
        throw(ArgumentError("node $(node.index) is not a leaf"))
    end
    tree = node.tree
    isempty(tree.data) &&
        throw(ArgumentError("tree was constructed without storing point data"))
    return LeafPointView(tree, get_leaf_range(_tree_data(tree), node.index))
end

function leaf_point_indices_indexnode(node::IndexNode{<:NNTree,Int}; original::Bool = true)
    _isleaf_indexnode(node) || throw(ArgumentError("node $(node.index) is not a leaf"))
    tree = node.tree
    idx = node.index
    range = get_leaf_range(_tree_data(tree), idx)
    return original ? view(tree.indices, range) : range
end

# Iteration helpers using AbstractTrees
preorder_indexnode(tree::Union{BallTree, KDTree}) = AbstractTrees.PreOrderDFS(treeroot_indexnode(tree))
postorder_indexnode(tree::Union{BallTree, KDTree}) = AbstractTrees.PostOrderDFS(treeroot_indexnode(tree))

# Leaf iterator
struct LeafWalkerIndexNode{TreeT<:NNTree}
    tree::TreeT
    leaf_range::UnitRange{Int}
end

function leaves_indexnode(tree::Union{BallTree, KDTree})
    return LeafWalkerIndexNode(tree, _get_leaf_range(tree))
end

@inline function Base.iterate(walker::LeafWalkerIndexNode)
    return iterate(walker, first(walker.leaf_range))
end

@inline function Base.iterate(walker::LeafWalkerIndexNode, idx::Int)
    idx > last(walker.leaf_range) && return nothing
    return IndexNode(walker.tree, idx), idx + 1
end

Base.IteratorSize(::Type{<:LeafWalkerIndexNode}) = Base.HasLength()
Base.length(walker::LeafWalkerIndexNode) = length(walker.leaf_range)
Base.eltype(::Type{LeafWalkerIndexNode{TreeT}}) where TreeT = IndexNode{TreeT,Int}

# Pretty printing
function AbstractTrees.printnode(io::IO, node::IndexNode{T,Int}) where {T<:Union{BallTree,KDTree}}
    tree = node.tree
    idx = node.index
    if _isleaf_indexnode(node)
        n_points = length(get_leaf_range(_tree_data(tree), idx))
        print(io, "Leaf(", n_points, " pts)")
    elseif tree isa BallTree
        sphere = treeregion_indexnode(node)
        print(io, "Ball(r=", round(sphere.r, digits=3), ")")
    elseif tree isa KDTree
        dim = Int(tree.split_dims[idx])
        val = tree.split_vals[idx]
        print(io, "Split(dim=", dim, ", val=", round(val, digits=3), ")")
    end
end
