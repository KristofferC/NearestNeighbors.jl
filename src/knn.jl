function check_k(tree, k)
    if k > length(tree.data) || k < 0
        throw(ArgumentError("k > number of points in tree or < 0"))
    end
end

"""
    knn(tree::NNTree, points, k [, skip=always_false]) -> indices, distances

Performs a lookup of the `k` nearest neighbors to the `points` from the data
in the `tree`.

# Arguments
- `tree`: The tree instance
- `points`: Query point(s) - can be a vector (single point), matrix (multiple points), or vector of vectors
- `k`: Number of nearest neighbors to find
- `skip`: Optional predicate function to skip points based on their index (default: `always_false`)

# Returns
- `indices`: Indices of the k nearest neighbors
- `distances`: Distances to the k nearest neighbors

See also: `knn!`, `nn`.
"""
function knn(tree::NNTree{V}, points::AbstractVector{T}, k::Int, sortres=false, skip::F=always_false) where {V, T <: AbstractVector, F<:Function}
    check_input(tree, points)
    check_k(tree, k)
    n_points = length(points)
    dists = [Vector{get_T(eltype(V))}(undef, k) for _ in 1:n_points]
    idxs = [Vector{Int}(undef, k) for _ in 1:n_points]
    for i in 1:n_points
        knn_point!(tree, points[i], sortres, dists[i], idxs[i], skip)
    end
    return idxs, dists
end

function knn_point!(tree::NNTree{V}, point::AbstractVector{T}, sortres, dist, idx, skip::F) where {V, T <: Number, F}
    fill!(idx, -1)
    fill!(dist, typemax(get_T(eltype(V))))
    _knn(tree, point, idx, dist, skip)
    if skip !== always_false
        skipped_idxs = findall(==(-1), idx)
        deleteat!(idx, skipped_idxs)
        deleteat!(dist, skipped_idxs)
    end
    sortres && heap_sort_inplace!(dist, idx)
    if tree.reordered
        for j in eachindex(idx)
            @inbounds idx[j] = tree.indices[idx[j]]
        end
    end
    return
end

"""
    knn!(idxs, dists, tree, point, k [, skip=always_false])

Same functionality as `knn` but stores the results in the input vectors `idxs` and `dists`.
Useful to avoid allocations or specify the element type of the output vectors.

# Arguments
- `idxs`: Pre-allocated vector to store indices (must be of length `k`)
- `dists`: Pre-allocated vector to store distances (must be of length `k`)
- `tree`: The tree instance
- `point`: Query point
- `k`: Number of nearest neighbors to find
- `skip`: Optional predicate function to skip points based on their index (default: `always_false`)

See also: `knn`, `nn`.
"""
function knn!(idxs::AbstractVector{<:Integer}, dists::AbstractVector, tree::NNTree{V}, point::AbstractVector{T}, k::Int, sortres=false, skip::F=always_false) where {V, T <: Number, F<:Function}
    check_k(tree, k)
    length(idxs) == k || throw(ArgumentError("idxs must be of length k"))
    length(dists) == k || throw(ArgumentError("dists must be of length k"))
    knn_point!(tree, point, sortres, dists, idxs, skip)
    return idxs, dists
end

function knn(tree::NNTree{V}, point::AbstractVector{T}, k::Int, sortres=false, skip::F=always_false) where {V, T <: Number, F<:Function}
    idx = Vector{Int}(undef, k)
    dist = Vector{get_T(eltype(V))}(undef, k)
    return knn!(idx, dist, tree, point, k, sortres, skip)
end

function knn(tree::NNTree{V}, points::AbstractMatrix{T}, k::Int, sortres=false, skip::F=always_false) where {V, T <: Number, F<:Function}
    dim = size(points, 1)
    knn_matrix(tree, points, k, Val(dim), sortres, skip)
end

# Function barrier
function knn_matrix(tree::NNTree{V}, points::AbstractMatrix{T}, k::Int, ::Val{dim}, sortres=false, skip::F=always_false) where {V, T <: Number, F<:Function, dim}
    # TODO: DRY with knn for AbstractVector
    check_input(tree, points)
    check_k(tree, k)
    n_points = size(points, 2)
    dists = [Vector{get_T(eltype(V))}(undef, k) for _ in 1:n_points]
    idxs = [Vector{Int}(undef, k) for _ in 1:n_points]

    for i in 1:n_points
        point = SVector{dim,T}(ntuple(j -> points[j, i], Val(dim)))
        knn_point!(tree, point, sortres, dists[i], idxs[i], skip)
    end
    return idxs, dists
end

"""
    nn(tree::NNTree, point [, skip]) -> index, distance
    nn(tree::NNTree, points [, skip]) -> indices, distances

Performs a lookup of the single nearest neighbor to the `point(s)` from the data.

# Arguments
- `tree`: The tree instance
- `point(s)`: Query point(s) - can be a vector (single point), matrix (multiple points), or vector of vectors
- `skip`: Optional predicate function to skip points based on their index (default: `always_false`)

# Returns
- For single point: `index` and `distance` of the nearest neighbor
- For multiple points: vectors of `indices` and `distances` of the nearest neighbors

See also: `knn`.
"""
nn(tree::NNTree{V}, points::AbstractVector{T}, skip::F=always_false) where {V, T <: Number,         F <: Function} = _nn(tree, points, skip) .|> only
nn(tree::NNTree{V}, points::AbstractVector{T}, skip::F=always_false) where {V, T <: AbstractVector, F <: Function} = _nn(tree, points, skip)  |> _onlyeach
nn(tree::NNTree{V}, points::AbstractMatrix{T}, skip::F=always_false) where {V, T <: Number,         F <: Function} = _nn(tree, points, skip)  |> _onlyeach

_nn(tree, points, skip) = knn(tree, points, 1, false, skip)

_onlyeach(v::Tuple) = only.(first(v)), only.(last(v))
