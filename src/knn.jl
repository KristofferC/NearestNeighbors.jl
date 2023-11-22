function check_k(tree, k)
    if k > length(tree.data) || k < 0
        throw(ArgumentError("k > number of points in tree or < 0"))
    end
end

"""
    knn(tree::NNTree, points, k [, sortres=false]) -> indices, distances
    nn(tree:NNTree, points) -> indices, distances

Performs a lookup of the `k` nearest neigbours to the `points` from the data
in the `tree`. If `sortres = true` the result is sorted such that the results are
in the order of increasing distance to the point. `skip` is an optional predicate
to determine if a point that would be returned should be skipped based on its
index.
"""
function knn(tree::NNTree{V}, points::Vector{T}, k::Int, sortres=false, skip::F=always_false) where {V, T <: AbstractVector, F<:Function}
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

function knn(tree::NNTree{V}, point::AbstractVector{T}, k::Int, sortres=false, skip::F=always_false) where {V, T <: Number, F<:Function}
    check_k(tree, k)
    idx = Vector{Int}(undef, k)
    dist = Vector{get_T(eltype(V))}(undef, k)
    knn_point!(tree, point, sortres, dist, idx, skip)
    return idx, dist
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


nn(tree::NNTree{V}, points::AbstractVector{T}, skip::F=always_false) where {V, T <: Number,         F <: Function} = _nn(tree, points, skip) .|> first
nn(tree::NNTree{V}, points::AbstractVector{T}, skip::F=always_false) where {V, T <: AbstractVector, F <: Function} = _nn(tree, points, skip)  |> _firsteach
nn(tree::NNTree{V}, points::AbstractMatrix{T}, skip::F=always_false) where {V, T <: Number,         F <: Function} = _nn(tree, points, skip)  |> _firsteach

_nn(tree, points, skip) = knn(tree, points, 1, false, skip)

_firsteach(v::Tuple) = first.(first(v)), first.(last(v))
