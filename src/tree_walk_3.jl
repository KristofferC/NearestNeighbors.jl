# Alternative tree walking API with nodes wrapping the tree
# Nodes contain both tree reference and index for ergonomic API

struct TreeNode3{T<:NNTree}
    tree::T
    index::Int
end


function treeroot3(tree::Union{KDTree, BallTree})
    _tree_data(tree).n_leafs == 0 && throw(ArgumentError(TREE_WALK_EMPTY))
    return TreeNode3(tree, 1)
end

treeroot3(tree::NNTree) = throw(ArgumentError(TREE_WALK_UNSUPPORTED))

function children3(node::TreeNode3)
    _isleaf3(node) && return ()
    idx = node.index
    return (TreeNode3(node.tree, getleft(idx)), TreeNode3(node.tree, getright(idx)))
end

function parent3(node::TreeNode3)
    idx = node.index
    idx == 1 && return nothing
    return TreeNode3(node.tree, getparent(idx))
end

_isleaf3(node::TreeNode3) =
    isleaf(_tree_data(node.tree).n_internal_nodes, node.index)


function treeregion3(node::TreeNode3{<:Union{BallTree, KDTree}})
    tree = node.tree
    idx = node.index

    if tree isa BallTree
        return tree.hyper_spheres[idx]
    elseif tree isa KDTree
        if _isleaf3(node)
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
        error("unknown tree type")
    end
end


function leafpoints3(node::TreeNode3)
    if !_isleaf3(node)
        throw(ArgumentError("node $(node.index) is not a leaf"))
    end
    isempty(node.tree.data) &&
        throw(ArgumentError("tree was constructed without storing point data"))
    return LeafPointView(node.tree, get_leaf_range(_tree_data(node.tree), node.index))
end

function leaf_point_indices3(node::TreeNode3; original::Bool = true)
    _isleaf3(node) || throw(ArgumentError("node $(node.index) is not a leaf"))
    idx = node.index
    range = get_leaf_range(_tree_data(node.tree), idx)
    return original ? view(node.tree.indices, range) : range
end

# Custom tree walkers

struct PreOrderWalker3{TreeT<:NNTree}
    root::TreeNode3{TreeT}
end


preorder3(tree::NNTree) = PreOrderWalker3(treeroot3(tree))

function _preorder_initial_state(walker::PreOrderWalker3)
    tree = walker.root.tree
    n_nodes = tree.tree_data.n_internal_nodes + tree.tree_data.n_leafs
    capacity = max(64, ceil(Int, log2(n_nodes + 1)) * 2)
    return (tree, sizehint!(Int[walker.root.index], capacity))
end

@inline function Base.iterate(walker::PreOrderWalker3, state = _preorder_initial_state(walker))
    tree, stack = state
    isempty(stack) && return nothing
    idx = pop!(stack)

    if !isleaf(_tree_data(tree).n_internal_nodes, idx)
        push!(stack, getright(idx))
        push!(stack, getleft(idx))
    end

    return TreeNode3(tree, idx), state
end

Base.IteratorSize(::Type{<:PreOrderWalker3}) = Base.HasLength()
Base.length(walker::PreOrderWalker3) =
    walker.root.tree.tree_data.n_internal_nodes + walker.root.tree.tree_data.n_leafs
Base.eltype(::Type{PreOrderWalker3{TreeT}}) where TreeT = TreeNode3{TreeT}


struct LeafWalker3{TreeT<:NNTree}
    tree::TreeT
    leaf_range::UnitRange{Int}
end

function leaves3(tree::NNTree)
    tree_data = _tree_data(tree)
    leaf_start = tree_data.n_internal_nodes + 1
    leaf_end = tree_data.n_internal_nodes + tree_data.n_leafs
    return LeafWalker3(tree, leaf_start:leaf_end)
end

@inline function Base.iterate(walker::LeafWalker3)
    return iterate(walker, first(walker.leaf_range))
end

@inline function Base.iterate(walker::LeafWalker3, idx::Int)
    idx > last(walker.leaf_range) && return nothing
    return TreeNode3(walker.tree, idx), idx + 1
end

Base.IteratorSize(::Type{<:LeafWalker3}) = Base.HasLength()
Base.length(walker::LeafWalker3) = length(walker.leaf_range)
Base.eltype(::Type{LeafWalker3{TreeT}}) where TreeT = TreeNode3{TreeT}


struct PostOrderWalker3{TreeT<:NNTree}
    root::TreeNode3{TreeT}
end


postorder3(tree::NNTree) = PostOrderWalker3(treeroot3(tree))

mutable struct PostOrderState3{TreeT}
    tree::TreeT
    stack::Vector{Int}
    last_visited::Int
end

function _postorder_initial_state(walker::PostOrderWalker3{TreeT}) where TreeT
    tree = walker.root.tree
    n_nodes = tree.tree_data.n_internal_nodes + tree.tree_data.n_leafs
    capacity = max(64, ceil(Int, log2(n_nodes + 1)) * 2)
    stack = sizehint!(Int[walker.root.index], capacity)
    return PostOrderState3(tree, stack, 0)
end

function Base.iterate(walker::PostOrderWalker3{TreeT}, state::PostOrderState3{TreeT} = _postorder_initial_state(walker)) where TreeT
    while !isempty(state.stack)
        idx = state.stack[end]

        if isleaf(_tree_data(state.tree).n_internal_nodes, idx)
            # Leaf node - visit it
            pop!(state.stack)
            state.last_visited = idx
            return TreeNode3(state.tree, idx), state
        end

        # Internal node - check if children were just visited
        left_idx = getleft(idx)
        if state.last_visited != left_idx && state.last_visited != getright(idx)
            # Children not visited yet, push them
            push!(state.stack, getright(idx))
            push!(state.stack, left_idx)
        else
            # Children visited, now visit this node
            pop!(state.stack)
            state.last_visited = idx
            return TreeNode3(state.tree, idx), state
        end
    end
    return nothing
end

Base.IteratorSize(::Type{<:PostOrderWalker3}) = Base.HasLength()
Base.length(walker::PostOrderWalker3) =
    walker.root.tree.tree_data.n_internal_nodes + walker.root.tree.tree_data.n_leafs
Base.eltype(::Type{PostOrderWalker3{TreeT}}) where TreeT = TreeNode3{TreeT}

# Pretty printing
Base.show(io::IO, node::TreeNode3) = print(io, "TreeNode3(", node.index, ")")


function printnode3(io::IO, node::TreeNode3{<:Union{BallTree, KDTree}}; show_region=false)
    tree = node.tree
    idx = node.index
    if _isleaf3(node)
        n_points = length(get_leaf_range(_tree_data(tree), idx))
        print(io, "Leaf(", n_points, " pts)")
    elseif tree isa BallTree
        sphere = tree.hyper_spheres[idx]
        print(io, "Ball(r=", round(sphere.r, digits=3), ")")
    elseif tree isa KDTree
        dim = Int(tree.split_dims[idx])
        val = tree.split_vals[idx]
        print(io, "Split(dim=", dim, ", val=", round(val, digits=3), ")")
    end

    if show_region
        region = treeregion3(node)
        if tree isa BallTree
            print(io, " center=[")
            for d in eachindex(region.center)
                d > 1 && print(io, ", ")
                print(io, round(region.center[d], digits=2))
            end
            print(io, "]")
        elseif tree isa KDTree
            print(io, " rect=[")
            for d in eachindex(region.mins)
                d > 1 && print(io, ", ")
                print(io, round(region.mins[d], digits=2), "→", round(region.maxes[d], digits=2))
            end
            print(io, "]")
        end
    end
end

function printtree3(io::IO, node::TreeNode3{<:Union{KDTree, BallTree}}, continuation::String, depth::Int, maxdepth::Int, show_region::Bool)
    depth > maxdepth && return

    printnode3(io, node; show_region)
    println(io)

    if !_isleaf3(node)
        if depth < maxdepth
            kids = children3(node)
            for (i, child) in enumerate(kids)
                is_last = i == length(kids)
                print(io, continuation, is_last ? "└─ " : "├─ ")
                new_continuation = continuation * (is_last ? "   " : "│  ")
                printtree3(io, child, new_continuation, depth + 1, maxdepth, show_region)
            end
        else
            # At maxdepth but node has children - indicate there's more
            println(io, continuation, "└─ ⋯")
        end
    end
end

function printtree3(tree::Union{KDTree, BallTree}; maxdepth=5, show_region=false, io::IO=stdout)
    printtree3(io, treeroot3(tree), "", 0, maxdepth, show_region)
end
