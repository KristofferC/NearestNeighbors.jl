"""
    TreeNode{TreeT<:NNTree}
Lightweight handle that identifies a node inside `tree`. Callers obtain these
handles via helpers such as `treeroot`, `TreeWalk`, or `children`
"""
struct TreeNode{TreeT<:NNTree}
    tree::TreeT
    index::Int
end

const TREE_WALK_UNSUPPORTED = "tree walking is only supported for KDTree and BallTree"
const TREE_WALK_EMPTY = "tree does not contain any nodes"

@inline _tree_data(tree::BallTree) = tree.tree_data
@inline _tree_data(tree::KDTree) = tree.tree_data
@inline function _tree_data(tree::NNTree)
    throw(ArgumentError(TREE_WALK_UNSUPPORTED))
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

"""
    LeafPointView(node)

Zero-copy view over the points stored in a leaf node.  Values are materialised
on demand.
"""
struct LeafPointView{TreeT<:NNTree}
    tree::TreeT
    range::UnitRange{Int}
end

function LeafPointView(node::TreeNode)
    if !_isleaf(node)
        throw(ArgumentError("node $(_treeindex(node)) is not a leaf"))
    end
    tree = _nodetree(node)
    return LeafPointView(tree, get_leaf_range(tree.tree_data, _treeindex(node)))
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
    tree = _nodetree(node)
    idx = _treeindex(node)

    if tree isa BallTree
        return tree.hyper_spheres[idx]
    elseif tree isa KDTree
        if _isleaf(node)
            # Reconstruct leaf bounds from parent
            if idx == 1
                # Root is both leaf and internal (degenerate case)
                return tree.hyper_rec
            end
            parent_idx = getparent(idx)
            parent_rect = tree.hyper_rects[parent_idx]
            split_dim = tree.split_dims[parent_idx]
            split_val = tree.split_vals[parent_idx]
            left_rect, right_rect = split_hyperrectangle(parent_rect, split_dim, split_val)
            # Determine if we're left or right child
            return idx == getleft(parent_idx) ? left_rect : right_rect
        else
            return tree.hyper_rects[idx]
        end
    else
        throw(ArgumentError("tree regions are only defined for BallTree and KDTree nodes currently"))
    end
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
