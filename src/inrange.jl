check_radius(r) = r < 0 && throw(ArgumentError("the query radius r must be ≧ 0"))

mutable struct ReservoirSampler{T<:Integer,RNG<:AbstractRNG,V<:AbstractVector{T}} <: AbstractVector{T}
    storage::V
    capacity::Int
    len::Int
    seen::Int
    rng::RNG
end

function ReservoirSampler(storage::AbstractVector{T}, capacity::Integer, rng::AbstractRNG) where {T<:Integer}
    capacity < 0 && throw(ArgumentError("k must be ≥ 0"))
    capacity <= length(storage) || throw(ArgumentError("storage length must be ≥ k"))
    return ReservoirSampler{T, typeof(rng), typeof(storage)}(storage, capacity, 0, 0, rng)
end

Base.IndexStyle(::Type{<:ReservoirSampler}) = IndexLinear()
Base.size(rs::ReservoirSampler) = (rs.len,)
Base.length(rs::ReservoirSampler) = rs.len
Base.getindex(rs::ReservoirSampler, i::Int) = rs.storage[i]
Base.setindex!(rs::ReservoirSampler, value, i::Int) = setindex!(rs.storage, value, i)

function Base.push!(rs::ReservoirSampler{T}, value) where {T}
    rs.seen += 1
    rs.capacity == 0 && return rs
    if rs.seen <= rs.capacity
        rs.len = rs.seen
        rs.storage[rs.len] = value
    else
        j = rand(rs.rng, 1:rs.seen)
        if j <= rs.capacity
            rs.storage[j] = value
        end
    end
    return rs
end

function Base.sort!(rs::ReservoirSampler; kwargs...)
    rs.len <= 1 && return rs
    sort!(view(rs.storage, 1:rs.len); kwargs...)
    return rs
end

"""
    inrange(tree::NNTree, points, radius) -> indices

Find all the points in the tree which are closer than `radius` to `points`.

# Arguments
- `tree`: The tree instance
- `points`: Query point(s) - can be a vector (single point), matrix (multiple points), or vector of vectors
- `radius`: Search radius

# Returns
- `indices`: Vector of indices of points within the radius

See also: `inrange!`, `inrangecount`.
"""
function inrange(tree::NNTree,
                 points::AbstractVector{T},
                 radius::Number,
                 sortres=false,
                 skip::F = Returns(false)) where {T <: AbstractVector, F}
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)

    idxs = [Vector{Int}() for _ in 1:length(points)]

    for i in 1:length(points)
        inrange_point!(tree, points[i], radius, sortres, idxs[i], skip)
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
function inrange!(idxs::AbstractVector, tree::NNTree{V}, point::AbstractVector{T}, radius::Number, sortres=false, skip=Returns(false)) where {V, T <: Number}
    check_input(tree, point)
    check_for_nan_in_points(point)
    check_radius(radius)
    length(idxs) == 0 || throw(ArgumentError("idxs must be empty"))
    inrange_point!(tree, point, radius, sortres, idxs, skip)
    return idxs
end

function inrange(tree::NNTree{V}, point::AbstractVector{T}, radius::Number, sortres=false) where {V, T <: Number}
    return inrange!(Int[], tree, point, radius, sortres)
end

function inrange(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, sortres=false) where {V, T <: Number}
    dim = size(points, 1)
    inrange_matrix(tree, points, radius, Val(dim), sortres)
end

function inrange_matrix(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, ::Val{dim}, sortres, skip::F=Returns(false)) where {V, T <: Number, dim, F}
    # TODO: DRY with inrange for AbstractVector
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)
    n_points = size(points, 2)
    idxs = [Vector{Int}() for _ in 1:n_points]

    for i in 1:n_points
        point = SVector{dim,T}(ntuple(j -> points[j, i], Val(dim)))
        inrange_point!(tree, point, radius, sortres, idxs[i], skip)
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

# Returns
- `count`: Number of points within the radius (integer for single point, vector for multiple points)
"""
function inrangecount(tree::NNTree{V}, point::AbstractVector{T}, radius::Number, skip::F=Returns(false)) where {V, T <: Number, F}
    check_input(tree, point)
    check_for_nan_in_points(point)
    check_radius(radius)
    return inrange_point!(tree, point, radius, false, nothing, skip)
end

function inrangecount(tree::NNTree,
        points::AbstractVector{T},
        radius::Number, skip::F=Returns(false)) where {T <: AbstractVector, F}
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)
    return inrange_point!.(Ref(tree), points, radius, false, nothing, skip)
end

function inrangecount(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, skip::F=Returns(false)) where {V, T <: Number, F}
    dim = size(points, 1)
    inrangecount_matrix(tree, points, radius, Val(dim), skip)
end

function inrangecount_matrix(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, ::Val{dim}, skip::F=Returns(false)) where {V, T <: Number, dim, F}
    check_input(tree, points)
    check_for_nan_in_points(points)
    check_radius(radius)
    n_points = size(points, 2)
    counts = Vector{Int}(undef, n_points)

    for i in 1:n_points
        point = SVector{dim,T}(ntuple(j -> points[j, i], Val(dim)))
        counts[i] = inrange_point!(tree, point, radius, false, nothing, skip)
    end
    return counts
end

function inrange_pairs(tree::NNTree, radius::Number, sortres=false, skip::F=Returns(false)) where {F}
    check_radius(radius)
    return _inrange_pairs(tree, radius, sortres, skip)
end

"""
    knninrange(tree::NNTree, point, radius, k; rng=Random.default_rng(), sortres=false, skip=Returns(false))

Return up to `k` indices drawn uniformly at random (without replacement) from the points
that lie within `radius` of `point`.

This behaves similarly to `inrange`, but it avoids returning more than `k` neighbors.

See also: `knninrange!`.
"""
function knninrange(tree::NNTree{V}, point::AbstractVector{T}, radius::Number, k::Integer;
                    rng::AbstractRNG=Random.default_rng(), sortres=false, skip::F=Returns(false)) where {V, T <: Number, F}
    check_input(tree, point)
    check_radius(radius)
    k < 0 && throw(ArgumentError("k must be ≥ 0"))
    k == 0 && return Int[]
    buf = Vector{Int}(undef, k)
    nsampled = knninrange!(buf, tree, point, radius, k; rng=rng, sortres=sortres, skip=skip)
    resize!(buf, nsampled)
    return buf
end

function knninrange(tree::NNTree{V}, points::AbstractVector{T}, radius::Number, k::Integer;
                    rng::AbstractRNG=Random.default_rng(), sortres=false, skip::F=Returns(false)) where {V, T <: AbstractVector, F}
    check_input(tree, points)
    check_radius(radius)
    return [knninrange(tree, points[i], radius, k; rng=rng, sortres=sortres, skip=skip) for i in 1:length(points)]
end

function knninrange(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, k::Integer;
                    rng::AbstractRNG=Random.default_rng(), sortres=false, skip::F=Returns(false)) where {V, T <: Number, F}
    check_input(tree, points)
    check_radius(radius)
    dim = size(points, 1)
    n_points = size(points, 2)
    idxs = Vector{Vector{Int}}(undef, n_points)
    for i in 1:n_points
        point = SVector{dim,T}(ntuple(j -> points[j, i], Val(dim)))
        idxs[i] = knninrange(tree, point, radius, k; rng=rng, sortres=sortres, skip=skip)
    end
    return idxs
end

"""
    knninrange!(idxs, tree, point, radius, k; rng=Random.default_rng(), sortres=false, skip=Returns(false))

Mutating version of `knninrange`. The first `k` entries of `idxs` are used as storage for the
reservoir sampler and will contain the sampled indices after the call returns. The function returns
the number of valid samples that were written (i.e. `min(k, number_in_range)`).

The length of `idxs` must be at least `k`. The contents beyond the returned sample length are left
untouched.
"""
function knninrange!(idxs::AbstractVector{<:Integer}, tree::NNTree{V}, point::AbstractVector{T},
                     radius::Number, k::Integer=length(idxs); rng::AbstractRNG=Random.default_rng(),
                     sortres=false, skip::F=Returns(false)) where {V, T <: Number, F}
    check_input(tree, point)
    check_radius(radius)
    k < 0 && throw(ArgumentError("k must be ≥ 0"))
    k == 0 && return 0
    k <= length(idxs) || throw(ArgumentError("idxs must have length ≥ k"))
    sampler = ReservoirSampler(idxs, k, rng)
    _inrange_point!(tree, point, radius, sortres, sampler, skip)
    return length(sampler)
end
