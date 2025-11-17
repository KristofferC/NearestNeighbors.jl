# Shared utilities for tree walking implementations

const TREE_WALK_UNSUPPORTED = "tree walking is only supported for KDTree and BallTree"
const TREE_WALK_EMPTY = "tree does not contain any nodes"

@inline _tree_data(tree::BallTree) = tree.tree_data
@inline _tree_data(tree::KDTree) = tree.tree_data
@inline function _tree_data(tree::NNTree)
    throw(ArgumentError(TREE_WALK_UNSUPPORTED))
end

# Common calculation for leaf range
function _get_leaf_range(tree::NNTree)
    tree_data = _tree_data(tree)
    leaf_start = tree_data.n_internal_nodes + 1
    leaf_end = tree_data.n_internal_nodes + tree_data.n_leafs
    return leaf_start:leaf_end
end

# Common calculation for walker stack capacity
function _walker_capacity(tree::NNTree)
    n_nodes = tree.tree_data.n_internal_nodes + tree.tree_data.n_leafs
    return max(64, ceil(Int, log2(n_nodes + 1)) * 2)
end

"""
    LeafPointView(tree, range)

Zero-copy view over the points stored in a leaf node. Values are materialised on demand.
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

# Common treeregion logic
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
