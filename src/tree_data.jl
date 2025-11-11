"""
    TreeData

Cache of pre-computed constants that lets us answer structural questions about
the implicit full binary tree (number of points in a leaf, index range of a
bucket, etc.) using only integer arithmetic.  KDTree and BallTree use these
values to reorder the input data once, after which they never have to store the
explicit tree shape.

# Fields
- `last_node_size`: Number of points stored in the final (possibly partial) leaf.
- `leafsize`: Target number of points for every other leaf.
- `n_leafs`: Total number of leaves after splitting.
- `n_internal_nodes`: Total non-leaf nodes (`n_leafs - 1` for any binary tree).
- `cross_node`: Index in the implicit heap layout where the partially filled
  bottom row begins; leaves with index ≥ `cross_node` are packed from the start
  of the reordered array, while leaves with index < `cross_node` belong to the
  complete rows and therefore live after the dense prefix.
- `offset` / `offset_cross`: Affine terms that map a leaf index to the start of
  its contiguous storage block depending on which side of `cross_node` it lives.
- `last_full_node`: Heap index of the final leaf (used to detect the only
  bucket that may have fewer than `leafsize` points).
"""
struct TreeData
    last_node_size::Int
    leafsize::Int
    n_leafs::Int
    n_internal_nodes::Int
    cross_node::Int
    offset::Int
    offset_cross::Int
    last_full_node::Int
end


function TreeData(data::AbstractVector{V}, leafsize) where V
    n_p = length(data)

    # Trivial input: no points means no tree.  Early-out to avoid log/div-by-zero.
    n_p == 0 && return TreeData(0, 0, 0, 0, 0, 0, 0, 0)

    # The tree is stored implicitly in an array using the usual “heap” indexing.
    # Once we know how many leaves we need, the rest of the metadata follows
    # deterministically.
    n_leafs = ceil(Integer, n_p / leafsize)
    n_internal_nodes = n_leafs - 1

    # `leafrow` is the exponent of the largest full row (power of two leaves)
    # that fits inside `n_leafs`.  Everything above that row is complete.
    leafrow = floor(Integer, log2(n_leafs))
    cross_node = 2^(leafrow + 1) # first leaf index in the next (possibly sparse) row
    last_node_size = n_p % leafsize
    if last_node_size == 0
        last_node_size = leafsize
    end

    # When the number of leaves is already an exact power of two the next row
    # does not exist, so halve `cross_node` to keep the comparison meaningful.
    # Example: `leafsize = 8`, `n_p = 32` -> `n_leafs = 4`, so the first leaf
    # index on the next row would be 8 even though the heap only contains
    # indices `1:7`.
    if cross_node >= n_internal_nodes + n_leafs
        cross_node = div(cross_node, 2)
    end

    # `row_adjust` encodes how many “missing” leaves there are in the last row.
    # Let `n_leafs = 2^leafrow + extra` where `extra ∈ [0, 2^leafrow)`.
    # Leaves with indices `< cross_node` live after the densely packed prefix
    # because their parent row is still full; multiplying the leaf index and
    # adding `offset` jumps over the prefix that belongs to the sparse row.
    row_adjust = 2(n_leafs - 2^leafrow) - 1

    # The first block that uses the after-prefix mapping belongs to leaf
    # `n_internal_nodes + 1`, so the affine term must rewind to the correct
    # insertion point.  For power-of-two trees this shrinks to
    # `offset = -leafsize + last_node_size + 1`, i.e. all leaves live in the
    # `offset_cross` regime (see below).
    offset = (row_adjust - n_internal_nodes - 1) * leafsize + last_node_size + 1

    # Leaves on/after `cross_node` are laid out from the start of the reordered
    # storage and simply march forward in `leafsize` sized blocks.
    offset_cross = -cross_node * leafsize + 1

    # The last leaf index equals the total number of nodes.  Only this leaf may
    # contain `last_node_size` points; every other leaf holds `leafsize`.
    last_full_node = n_leafs + n_internal_nodes

    # Example walkthrough:
    #   n_p = 10, leafsize = 4 -> n_leafs = 3, n_internal_nodes = 2,
    #   cross_node = 4, offset = 1.
    #   Leaves 4 and 5 (the sparse bottom row) receive ranges 1:4 and 5:6 via
    #   `offset_cross`, while leaf 3 receives 7:10 via `offset`.
    # This lets `get_leaf_range` reconstruct the contiguous block for any leaf
    # without storing explicit pointers.
    TreeData(last_node_size, leafsize, n_leafs,
    n_internal_nodes, cross_node, offset, offset_cross, last_full_node)
end
