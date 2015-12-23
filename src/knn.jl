"""
    knn(tree::NNTree, points, k [, sortres=false]) -> indices, distances

Performs a lookup of the `k` nearest neigbours to the `points` from the data
in the `tree`. If `sortres = true` the result is sorted such that the results are
in the order of increasing distance to the point.
"""
function knn{T <: Real, P <: Real}(tree::NNTree{T}, points::AbstractArray{P}, k::Int, sortres::Bool=false)
    Tret = Distances.result_type(tree.metric, tree.data, points)
    check_input(tree, points)
    n_points = size(points, 2)
    n_dim = size(points, 1)

    if k > size(tree.data, 2) || k <= 0
        throw(ArgumentError("k > number of points in tree or ≦ 0"))
    end

    dists = Vector{Vector{Tret}}(n_points)
    idxs = Vector{Vector{Int}}(n_points)
    point = zeros(P, n_dim)
    for i in 1:n_points
        @devec point[:] = points[:, i]
        best_idxs, best_dists = _knn(tree, point, k)
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
    return do_return(idxs, dists, points)
end

do_return(idxs, dists, ::AbstractVector) = idxs[1], dists[1]
do_return(idxs, dists, ::AbstractMatrix) = idxs, dists
