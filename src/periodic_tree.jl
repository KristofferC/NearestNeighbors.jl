"""
    PeriodicTree(tree::NNTree, bounds_min, bounds_max) -> PeriodicTree

Creates a periodic wrapper around an existing nearest neighbor tree (KDTree, BallTree, or BruteTree)
that handles periodic boundary conditions.

# Arguments
- `tree::NNTree`: The underlying tree structure (KDTree, BallTree, or BruteTree)
- `bounds_min`: Vector of minimum bounds for each dimension
- `bounds_max`: Vector of maximum bounds for each dimension

# Requirements
- All data points in the tree must be within the specified periodic box bounds
- Box dimensions must be positive and finite (except for non-periodic dimensions)

# Returns
- `PeriodicTree`: A tree that performs nearest neighbor searches with periodic boundary conditions

# Algorithm
The periodic tree works by creating "mirror images" of the query point by shifting it by multiples
of the box dimensions. For each periodic image, it searches the underlying tree and combines results
while ensuring no duplicates are returned.

# Performance Notes
- For best performance, ensure search radii are ≤ half the smallest box dimension
- Larger radii will still work correctly but may perform redundant searches
- Dimensions with infinite bounds are treated as non-periodic

# Examples
```julia
using NearestNeighbors, StaticArrays

# Create some 2D data
data = [SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(7.0, 8.0)]
bounds_min = [0.0, 0.0]
bounds_max = [10.0, 10.0]

# Create periodic tree
kdtree = KDTree(data)
ptree = PeriodicTree(kdtree, bounds_min, bounds_max)

# Search near boundary - finds points through periodic wrapping
query_point = [9.0, 1.0]
idxs, dists = knn(ptree, query_point, 2)
```
"""
struct PeriodicTree{V<:AbstractVector, M, Tree <: NNTree{V, M}, D} <: NNTree{V,M}
    tree::Tree
    bbox::HyperRectangle{V}
    combos::Vector{SVector{D, Int}}
    box_widths::SVector{D, eltype(V)}

    function PeriodicTree(tree::NNTree{V,M}, bounds_min, bounds_max) where {V,M}
        dim = length(V)
        if length(bounds_min) != dim || length(bounds_max) != dim
            throw(ArgumentError("Bounding box dimensions do not match data dimensions"))
        end

        # Store finite box widths, use 0.0 for non-periodic dimensions to avoid Inf * 0 = NaN
        box_widths = SVector{dim}(
            isfinite(bounds_max[i] - bounds_min[i]) ? bounds_max[i] - bounds_min[i] : 0.0
            for i in 1:dim
        )

        # Check for valid box dimensions (finite dimensions must be positive)
        for i in 1:dim
            actual_width = bounds_max[i] - bounds_min[i]
            if isfinite(actual_width) && actual_width <= 0
                throw(ArgumentError("Box width in dimension $i must be positive, got $actual_width"))
            end
        end

        # Validate that all data points are within the periodic box bounds
        # This is important for correct periodic behavior
        for (idx, point) in enumerate(tree.data)
            for i in 1:dim
                if point[i] < bounds_min[i] || point[i] > bounds_max[i]
                    throw(ArgumentError("Data point $idx has coordinate $(point[i]) in dimension $i, which is outside the periodic box bounds [$(bounds_min[i]), $(bounds_max[i])]"))
                end
            end
        end

        # Find periodic dimensions (those with non-zero box widths)
        periodic_dims = findall(>(0), box_widths)
        n_periodic = length(periodic_dims)

        # Generate combinations only for periodic dimensions
        if n_periodic == 0
            # No periodic dimensions - only search original box
            combos_reordered = [zero(SVector{dim, Int})]
        else
            # Generate all combinations of [-1, 0, 1] for periodic dimensions only
            periodic_ranges = ntuple(i -> -1:1, Val(n_periodic))
            periodic_combos = collect(Iterators.product(periodic_ranges...))

            # Convert to full-dimension combo vectors
            combos = Vector{SVector{dim, Int}}()
            for combo_vals in periodic_combos
                full_combo = zeros(Int, dim)
                for (i, dim_idx) in enumerate(periodic_dims)
                    full_combo[dim_idx] = combo_vals[i]
                end
                push!(combos, SVector{dim, Int}(full_combo))
            end

            # Put the (0, 0, 0, ...) combo first to search the original box first
            # This is important for performance as the original box often contains the closest points
            zero_combo = zero(SVector{dim, Int})
            filtered_combos = filter(x -> x != zero_combo, combos)
            combos_reordered = pushfirst!(filtered_combos, zero_combo)
        end

        return new{V, M, typeof(tree), dim}(
            tree,
            HyperRectangle(SVector{dim}(bounds_min), SVector{dim}(bounds_max)),
            combos_reordered,
            box_widths
        )
    end
end

get_tree(tree::PeriodicTree) = tree.tree


function Base.show(io::IO, tree::PeriodicTree{V}) where {V}
    println(io, "Periodic Tree: $(typeof(tree.tree))")

    # Show periodic and non-periodic dimensions clearly
    periodic_dims = findall(>(0), tree.box_widths)
    non_periodic_dims = findall(==(0), tree.box_widths)

    println(io, "  Dimensions: ", length(V))
    if !isempty(periodic_dims)
        println(io, "    Periodic ($(length(periodic_dims))): ", periodic_dims)
        for dim in periodic_dims
            println(io, "      Dim $dim: [$(tree.bbox.mins[dim]), $(tree.bbox.maxes[dim])] (width: $(tree.box_widths[dim]))")
        end
    end
    if !isempty(non_periodic_dims)
        println(io, "    Non-periodic ($(length(non_periodic_dims))): ", non_periodic_dims)
    end

    println(io, "  Number of points: ", length(tree.tree.data))
    println(io, "  Metric: ", tree.tree.metric)
    print(io,   "  Reordered: ", tree.tree.reordered)
end

function _knn(tree::PeriodicTree{V,M},
    point::AbstractVector,
    best_idxs::AbstractVector{<:Integer},
    best_dists::AbstractVector,
    skip::F) where {V, M, F}

    # Search all periodic mirror boxes
    # Each combo represents a different "image" of the periodic box
    # e.g., (0,0) = original, (1,0) = shifted right by box_width, (-1,1) = shifted left and up
    for combo in tree.combos
        # Create the shift vector: multiply box dimensions by the combo coefficients
        shift_vector = tree.box_widths .* combo
        # Create a "mirror image" of the query point in this periodic box
        point_shifted = point + shift_vector

        # Calculate minimum distance from shifted point to the original bounding box
        min_dist_to_canonical = get_min_distance_no_end(tree.tree.metric, tree.bbox, point_shifted)
        
        # Optimization: Skip mirror boxes that can't improve current results
        # If minimum possible distance is >= current k-th nearest distance, skip this mirror box
        if eval_end(tree.tree.metric, min_dist_to_canonical) >= best_dists[1]
            continue
        end

        # Search the underlying tree with the shifted query point
        # The 'true' parameter enables uniqueness checking to prevent duplicate results
        if tree.tree isa KDTree
            knn_kernel!(tree.tree, 1, point_shifted, best_idxs, best_dists, min_dist_to_canonical, tree.tree.hyper_rec, skip, true)
        elseif tree.tree isa BallTree
            knn_kernel!(tree.tree, 1, point_shifted, best_idxs, best_dists, skip, true)
        else
            @assert tree.tree isa BruteTree
            knn_kernel!(tree.tree, point_shifted, best_idxs, best_dists, skip, true)
        end
    end

    # For KDTree, we need to finalize the distance calculations
    # This is because KDTree uses squared distances internally for efficiency
    if tree.tree isa KDTree
        @simd for i in eachindex(best_dists)
            @inbounds best_dists[i] = eval_end(tree.tree.metric, best_dists[i])
        end
    end

    # Verify no duplicates were returned (should be guaranteed by unique=true above)
    @assert allunique(best_idxs)
    return
end

function _inrange(tree::PeriodicTree{V},
    point::AbstractVector,
    radius::Number,
    idx_in_ball::Union{Nothing, Vector{<:Integer}},
    skip::F) where {V, F}

    # Search all periodic mirror boxes for points within the given radius
    for combo in tree.combos
        # Create the shift vector for this mirror box
        shift_vector = tree.box_widths .* combo
        # Create a "mirror image" of the query point
        point_shifted = point + shift_vector

        # Performance optimization: skip mirror boxes that are too far away
        # If the closest possible point in the original box is farther than radius,
        # then no points in this mirror box can be within radius
        min_dist_to_bbox = get_min_distance_no_end(tree.tree.metric, tree.bbox, point_shifted)
        if eval_end(tree.tree.metric, min_dist_to_bbox) > radius
            continue
        end

        # Search the underlying tree with the shifted query point
        # The 'true' parameter enables uniqueness checking to prevent duplicate results
        if tree.tree isa KDTree
            # KDTree requires additional distance computation parameters
            max_dist_contribs = get_max_distance_contributions(tree.tree.metric, tree.bbox, point_shifted)
            max_dist = tree.tree.metric isa Chebyshev ? maximum(max_dist_contribs) : sum(max_dist_contribs)
            inrange_kernel!(tree.tree, 1, point_shifted, eval_op(tree.tree.metric, radius, zero(min_dist_to_bbox)), idx_in_ball,
                          tree.tree.hyper_rec, min_dist_to_bbox, max_dist_contribs, max_dist, skip, true)
        elseif tree.tree isa BallTree
            # BallTree uses a hypersphere for range queries
            ball = HyperSphere(convert(V, point_shifted), convert(eltype(V), radius))
            inrange_kernel!(tree.tree, 1, point_shifted, ball, idx_in_ball, skip, true)
        else
            @assert tree.tree isa BruteTree
            # BruteTree has the simplest interface
            inrange_kernel!(tree.tree, point_shifted, radius, idx_in_ball, skip, true)
        end
    end

    # Verify no duplicates were returned (should be guaranteed by unique=true above)
    @assert allunique(idx_in_ball)
    return
end
