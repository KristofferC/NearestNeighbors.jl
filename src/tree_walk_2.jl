# Alternative tree walking API with isbits nodes
# Nodes are plain data types, tree is passed as parameter

struct TreeNode2
    index::Int
end


function treeroot2(tree::Union{KDTree, BallTree})
    _tree_data(tree).n_leafs == 0 && throw(ArgumentError(TREE_WALK_EMPTY))
    return TreeNode2(1)
end

treeroot2(tree::NNTree) = throw(ArgumentError(TREE_WALK_UNSUPPORTED))

function children2(tree::Union{BallTree, KDTree}, node::TreeNode2)
    _isleaf2(tree, node) && return ()
    idx = node.index
    return (TreeNode2(getleft(idx)), TreeNode2(getright(idx)))
end

function parent2(tree::Union{BallTree, KDTree}, node::TreeNode2)
    idx = node.index
    idx == 1 && return nothing
    return TreeNode2(getparent(idx))
end

_isleaf2(tree::NNTree, node::TreeNode2) =
    isleaf(_tree_data(tree).n_internal_nodes, node.index)


function treeregion2(tree::Union{BallTree, KDTree}, node::TreeNode2)
    idx = node.index

    if tree isa BallTree
        return tree.hyper_spheres[idx]
    elseif tree isa KDTree
        if _isleaf2(tree, node)
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


function leafpoints2(tree::NNTree, node::TreeNode2)
    if !_isleaf2(tree, node)
        throw(ArgumentError("node $(node.index) is not a leaf"))
    end
    isempty(tree.data) &&
        throw(ArgumentError("tree was constructed without storing point data"))
    return LeafPointView(tree, get_leaf_range(_tree_data(tree), node.index))
end

function leaf_point_indices2(tree::NNTree, node::TreeNode2; original::Bool = true)
    _isleaf2(tree, node) || throw(ArgumentError("node $(node.index) is not a leaf"))
    idx = node.index
    range = get_leaf_range(_tree_data(tree), idx)
    return original ? view(tree.indices, range) : range
end

# Custom tree walkers

struct PreOrderWalker2{TreeT<:NNTree}
    tree::TreeT
    root::TreeNode2
end


preorder2(tree::NNTree) = PreOrderWalker2(tree, treeroot2(tree))

function _preorder_initial_state(walker::PreOrderWalker2)
    n_nodes = walker.tree.tree_data.n_internal_nodes + walker.tree.tree_data.n_leafs
    capacity = max(64, ceil(Int, log2(n_nodes + 1)) * 2)
    return sizehint!(TreeNode2[walker.root], capacity)
end

@inline function Base.iterate(walker::PreOrderWalker2, stack::Vector{TreeNode2} = _preorder_initial_state(walker))
    isempty(stack) && return nothing
    node = pop!(stack)

    if !_isleaf2(walker.tree, node)
        idx = node.index
        push!(stack, TreeNode2(getright(idx)))
        push!(stack, TreeNode2(getleft(idx)))
    end

    return node, stack
end

Base.IteratorSize(::Type{<:PreOrderWalker2}) = Base.HasLength()
Base.length(walker::PreOrderWalker2) =
    walker.tree.tree_data.n_internal_nodes + walker.tree.tree_data.n_leafs
Base.eltype(::Type{<:PreOrderWalker2}) = TreeNode2


struct LeafWalker2{TreeT<:NNTree}
    tree::TreeT
    leaf_range::UnitRange{Int}
end

function leaves2(tree::NNTree)
    tree_data = _tree_data(tree)
    leaf_start = tree_data.n_internal_nodes + 1
    leaf_end = tree_data.n_internal_nodes + tree_data.n_leafs
    return LeafWalker2(tree, leaf_start:leaf_end)
end

function Base.iterate(walker::LeafWalker2)
    return iterate(walker, first(walker.leaf_range))
end

function Base.iterate(walker::LeafWalker2, idx::Int)
    idx > last(walker.leaf_range) && return nothing
    return TreeNode2(idx), idx + 1
end

Base.IteratorSize(::Type{<:LeafWalker2}) = Base.HasLength()
Base.length(walker::LeafWalker2) = length(walker.leaf_range)
Base.eltype(::Type{<:LeafWalker2}) = TreeNode2


struct PostOrderWalker2{TreeT<:NNTree}
    tree::TreeT
    root::TreeNode2
end


postorder2(tree::NNTree) = PostOrderWalker2(tree, treeroot2(tree))

mutable struct PostOrderState
    stack::Vector{TreeNode2}
    last_visited::Int
end

function _postorder_initial_state(walker::PostOrderWalker2)
    n_nodes = walker.tree.tree_data.n_internal_nodes + walker.tree.tree_data.n_leafs
    capacity = max(64, ceil(Int, log2(n_nodes + 1)) * 2)
    stack = sizehint!(TreeNode2[walker.root], capacity)
    return PostOrderState(stack, 0)
end

function Base.iterate(walker::PostOrderWalker2, state::PostOrderState = _postorder_initial_state(walker))
    while !isempty(state.stack)
        node = state.stack[end]
        idx = node.index

        if _isleaf2(walker.tree, node)
            # Leaf node - visit it
            pop!(state.stack)
            state.last_visited = idx
            return node, state
        end

        # Internal node - check if children were just visited
        left_idx = getleft(idx)
        if state.last_visited != left_idx && state.last_visited != getright(idx)
            # Children not visited yet, push them
            push!(state.stack, TreeNode2(getright(idx)))
            push!(state.stack, TreeNode2(left_idx))
        else
            # Children visited, now visit this node
            pop!(state.stack)
            state.last_visited = idx
            return node, state
        end
    end
    return nothing
end

Base.IteratorSize(::Type{<:PostOrderWalker2}) = Base.HasLength()
Base.length(walker::PostOrderWalker2) =
    walker.tree.tree_data.n_internal_nodes + walker.tree.tree_data.n_leafs
Base.eltype(::Type{<:PostOrderWalker2}) = TreeNode2

# Pretty printing
Base.show(io::IO, node::TreeNode2) = print(io, "TreeNode2(", node.index, ")")


function printnode2(io::IO, tree::Union{BallTree, KDTree}, node::TreeNode2; show_region=false)
    idx = node.index
    if _isleaf2(tree, node)
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
        region = treeregion2(tree, node)
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

function printtree2(io::IO, tree::Union{KDTree, BallTree}, node::TreeNode2, continuation::String, depth::Int, maxdepth::Int, show_region::Bool)
    depth > maxdepth && return

    printnode2(io, tree, node; show_region)
    println(io)

    if !_isleaf2(tree, node)
        if depth < maxdepth
            kids = children2(tree, node)
            for (i, child) in enumerate(kids)
                is_last = i == length(kids)
                print(io, continuation, is_last ? "└─ " : "├─ ")
                new_continuation = continuation * (is_last ? "   " : "│  ")
                printtree2(io, tree, child, new_continuation, depth + 1, maxdepth, show_region)
            end
        else
            # At maxdepth but node has children - indicate there's more
            println(io, continuation, "└─ ⋯")
        end
    end
end

function printtree2(tree::Union{KDTree, BallTree}; maxdepth=5, show_region=false, io::IO=stdout)
    printtree2(io, tree, treeroot2(tree), "", 0, maxdepth, show_region)
end
