check_k(tree, k) = (k > length(tree.data)|| k <= 0) && throw(ArgumentError("k > number of points in tree or ≦ 0"))

"""
    knn(tree::NNTree, points, k [, sortres=false]) -> indices, distances

Performs a lookup of the `k` nearest neigbours to the `points` from the data
in the `tree`. If `sortres = true` the result is sorted such that the results are
in the order of increasing distance to the point. `skip` is an optional predicate
to determine if a point that would be returned should be skipped.
"""
function knn{V, T <: AbstractVector}(tree::NNTree{V}, points::Vector{T}, k::Int, sortres=false, skip::Function=always_false)
    check_input(tree, points)
    check_k(tree, k)

    n_points = length(points)
    dists = [Vector{DistanceType}(k) for _ in 1:n_points]
    idxs = [Vector{Int}(k) for _ in 1:n_points]
    for i in 1:n_points
        knn_point!(tree, points[i], k, sortres, dists[i], idxs[i], skip)
    end
    return idxs, dists
end

function knn_point!{V, T <: Number}(tree::NNTree{V}, point::AbstractVector{T}, k::Int, sortres, skip, dist, idx)
    fill!(idx, -1)
    fill!(dist, typemax(DistanceType))
    _knn(tree, point, k, skip, idx, dist)
    sortres && heap_sort_inplace!(dist, idx)
    if tree.reordered
        for j in 1:k
            @inbounds idx[j] = tree.indices[idx[j]]
        end
    end
end

function knn{V, T <: Number}(tree::NNTree{V}, point::AbstractVector{T}, k::Int, sortres=false, skip::Function=always_false)
    if k > length(tree.data)|| k <= 0
        throw(ArgumentError("k > number of points in tree or ≦ 0"))
    end

    idx = Vector{Int}(k)
    dist = Vector{DistanceType}(k)
    knn_point!(tree, point, k, sortres, skip, dist, idx)
    return idx, dist
end

function knn{V, T <: Number}(tree::NNTree{V}, point::Matrix{T}, k::Int, sortres=false, skip::Function=always_false)
    dim = size(point, 1)
    npoints = size(point, 2)
    if isbits(T)
        new_data = reinterpret(SVector{dim, T}, point, (length(point) ÷ dim, ))
    else
        new_data = SVector{dim, T}[SVector{dim, T}(point[:, i]) for i in 1:npoints]
    end
    knn(tree, new_data, k, sortres, skip)
end
