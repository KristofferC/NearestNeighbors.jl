check_radius(r) = r < zero(r) && throw(ArgumentError("the query radius r must be ≧ 0"))

"""
    inrange(tree::NNTree, points, radius) -> indices

Find all the points in the tree which are closer than `radius` to `points`.

# Arguments
- `tree`: The tree instance
- `points`: Query point(s) - can be a vector (single point), matrix (multiple points), or vector of vectors
- `radius`: Search radius
- `skip` (optional): Predicate to skip certain points.
- `skipself` (optional, batched queries only): When querying the same dataset, skip the point whose index matches the query index.

# Returns
- `indices`: Vector of indices of points within the radius

See also: `inrange!`, `inrangecount`.
"""
function inrange(tree::NNTree,
                 points::AbstractVector{T},
                 radius::Number,
                 sortres=false,
                 skip::F = Returns(false);
                 skipself::Bool=false) where {T <: AbstractVector, F}
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)

    idxs = [Vector{Int}() for _ in 1:length(points)]

    for i in 1:length(points)
        query_skip = _skip_with_self(skip, skipself ? i : nothing, skipself)
        inrange_point!(tree, points[i], radius, sortres, idxs[i], query_skip)
    end
    return idxs
end

inrange_point!(tree, point, radius, sortres, idx, skip::F) where {F} = _inrange_point!(tree, point, radius, sortres, idx, skip)

function _inrange_point!(tree, point, radius, sortres, idx, skip::F) where {F}
    count = _inrange(tree, point, radius, idx, skip)
    if idx !== nothing
        inner_tree = get_tree(tree)
        if inner_tree.reordered
            @inbounds for j in 1:length(idx)
                idx[j] = inner_tree.indices[idx[j]]
            end
        end
        sortres && sort!(idx)
    end
    return count
end

"""
    inrange!(idxs, tree, point, radius)

Same functionality as `inrange` but stores the results in the input vector `idxs`.
Useful to avoid allocations or specify the element type of the output vector.

# Arguments
- `idxs`: Pre-allocated vector to store indices (must be empty)
- `tree`: The tree instance
- `point`: Query point
- `radius`: Search radius

See also: `inrange`, `inrangecount`.
"""
function inrange!(idxs::AbstractVector, tree::NNTree{V}, point::AbstractVector{T}, radius::Number, sortres=false, skip=Returns(false); skipself::Bool=false) where {V, T <: Number}
    skipself && throw(ArgumentError("skipself is only supported for batched queries; pass a skip predicate instead for single points"))
    check_input(tree, point)
    check_for_nan_in_points(point)
    check_radius(radius)
    length(idxs) == 0 || throw(ArgumentError("idxs must be empty"))
    inrange_point!(tree, point, radius, sortres, idxs, skip)
    return idxs
end

function inrange(tree::NNTree{V}, point::AbstractVector{T}, radius::Number, sortres=false; skipself::Bool=false) where {V, T <: Number}
    skipself && throw(ArgumentError("skipself is only supported for batched queries; pass a skip predicate instead for single points"))
    return inrange!(Int[], tree, point, radius, sortres)
end

function inrange(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, sortres=false; skipself::Bool=false) where {V, T <: Number}
    dim = size(points, 1)
    inrange_matrix(tree, points, radius, Val(dim), sortres; skipself)
end

function inrange_matrix(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, ::Val{dim}, sortres, skip::F=Returns(false); skipself::Bool=false) where {V, T <: Number, dim, F}
    # TODO: DRY with inrange for AbstractVector
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)
    n_points = size(points, 2)
    idxs = [Vector{Int}() for _ in 1:n_points]

    for i in 1:n_points
        point = SVector{dim,T}(ntuple(j -> points[j, i], Val(dim)))
        query_skip = _skip_with_self(skip, skipself ? i : nothing, skipself)
        inrange_point!(tree, point, radius, sortres, idxs[i], query_skip)
    end
    return idxs
end

"""
    inrangecount(tree::NNTree, points, radius) -> count

Count all the points in the tree which are closer than `radius` to `points`.

# Arguments
- `tree`: The tree instance
- `points`: Query point(s) - can be a vector (single point), matrix (multiple points), or vector of vectors
- `radius`: Search radius
- `skip` (optional): Predicate to skip certain points.
- `skipself` (optional, batched queries only): When querying the same dataset, skip the point whose index matches the query index.

# Returns
- `count`: Number of points within the radius (integer for single point, vector for multiple points)
"""
function inrangecount(tree::NNTree{V}, point::AbstractVector{T}, radius::Number, skip::F=Returns(false); skipself::Bool=false) where {V, T <: Number, F}
    skipself && throw(ArgumentError("skipself is only supported for batched queries; pass a skip predicate instead for single points"))
    check_input(tree, point)
    check_for_nan_in_points(point)
    check_radius(radius)
    return inrange_point!(tree, point, radius, false, nothing, skip)
end

function inrangecount(tree::NNTree,
        points::AbstractVector{T},
        radius::Number, skip::F=Returns(false); skipself::Bool=false) where {T <: AbstractVector, F}
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)
    counts = Vector{Int}(undef, length(points))
    for i in 1:length(points)
        query_skip = _skip_with_self(skip, skipself ? i : nothing, skipself)
        counts[i] = inrange_point!(tree, points[i], radius, false, nothing, query_skip)
    end
    return counts
end

function inrangecount(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, skip::F=Returns(false); skipself::Bool=false) where {V, T <: Number, F}
    dim = size(points, 1)
    inrangecount_matrix(tree, points, radius, Val(dim), skip; skipself)
end

function inrangecount_matrix(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, ::Val{dim}, skip::F=Returns(false); skipself::Bool=false) where {V, T <: Number, dim, F}
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)
    n_points = size(points, 2)
    counts = Vector{Int}(undef, n_points)

    for i in 1:n_points
        point = SVector{dim,T}(ntuple(j -> points[j, i], Val(dim)))
        query_skip = _skip_with_self(skip, skipself ? i : nothing, skipself)
        counts[i] = inrange_point!(tree, point, radius, false, nothing, query_skip)
    end
    return counts
end

function inrange_pairs(tree::NNTree, radius::Number, sortres=false, skip::F=Returns(false)) where {F}
    check_radius(radius)
    return _inrange_pairs(tree, radius, sortres, skip)
end
