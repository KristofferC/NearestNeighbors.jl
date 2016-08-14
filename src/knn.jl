"""
    knn(tree::NNTree, points, k [, sortres=false]) -> indices, distances

Performs a lookup of the `k` nearest neigbours to the `points` from the data
in the `tree`. If `sortres = true` the result is sorted such that the results are
in the order of increasing distance to the point. `skip` is an optional predicate
to determine if a point that would be returned should be skipped.
"""
function knn{V, T <: AbstractVector}(tree::NNTree{V}, points::Vector{T}, k::Int, sortres=false, skip::Function=always_false)
    check_input(tree, points)
    n_points = length(points)
    n_dim = length(V)

    if k > length(tree.data)|| k <= 0
        throw(ArgumentError("k > number of points in tree or ≦ 0"))
    end

    dists = Array(Vector{DistanceType}, n_points)
    idxs = Array(Vector{Int}, n_points)
    for i in 1:n_points
        point = points[i]
        best_idxs, best_dists = _knn(tree, point, k, skip)
        if sortres
            heap_sort_inplace!(best_dists, best_idxs)
        end
        dists[i] = best_dists
        if tree.reordered
            for j in 1:k
                @inbounds best_idxs[j] = tree.indices[best_idxs[j]]
            end
        end
        idxs[i] = best_idxs
    end
    return idxs, dists
end

function knn{V, T <: Number}(tree::NNTree{V}, point::AbstractVector{T}, k::Int, sortres=false, skip::Function=always_false)
    idxs, dists = knn(tree, Vector{T}[point], k, sortres, skip)
    return idxs[1], dists[1]
end

function knn{V, T <: Number}(tree::NNTree{V}, point::Vector{T}, k::Int, sortres=false, skip::Function=always_false)
    idxs, dists = knn(tree, [convert(SVector{length(point), T}, point)], k, sortres, skip)
    return idxs[1], dists[1]
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
