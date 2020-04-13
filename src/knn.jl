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
function knn(tree::NNTree{V}, points::Vector{T}, k::Int, sortres=false, skip::Function=always_false) where {V, T <: AbstractVector}
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

function knn_point!(tree::NNTree{V}, point::AbstractVector{T}, sortres, dist, idx, skip) where {V, T <: Number}
    fill!(idx, -1)
    fill!(dist, typemax(get_T(eltype(V))))
    _knn(tree, point, idx, dist, skip)
    sortres && heap_sort_inplace!(dist, idx)
    if tree.reordered
        for j in eachindex(idx)
            @inbounds idx[j] = tree.indices[idx[j]]
        end
    end
end

function knn(tree::NNTree{V}, point::AbstractVector{T}, k::Int, sortres=false, skip::Function=always_false) where {V, T <: Number}
    check_k(tree, k)
    idx = Vector{Int}(undef, k)
    dist = Vector{get_T(eltype(V))}(undef, k)
    knn_point!(tree, point, sortres, dist, idx, skip)
    return idx, dist
end

function knn(tree::NNTree{V}, point::AbstractMatrix{T}, k::Int, sortres=false, skip::Function=always_false) where {V, T <: Number}
    dim = size(point, 1)
    npoints = size(point, 2)
    if isbitstype(T)
        new_data = copy_svec(T, point, Val(dim))
    else
        new_data = SVector{dim,T}[SVector{dim,T}(point[:, i]) for i in 1:npoints]
    end
    knn(tree, new_data, k, sortres, skip)
end

nn(tree::NNTree{V}, points::AbstractVector{T}, skip::Function=always_false) where {V, T <: Number}         = _nn(tree, points, skip) .|> first
nn(tree::NNTree{V}, points::AbstractVector{T}, skip::Function=always_false) where {V, T <: AbstractVector} = _nn(tree, points, skip)  |> _firsteach
nn(tree::NNTree{V}, points::AbstractMatrix{T}, skip::Function=always_false) where {V, T <: Number}         = _nn(tree, points, skip)  |> _firsteach

_nn(tree, points, skip) = knn(tree, points, 1, false, skip)

_firsteach(v::Tuple) = first.(first(v)), first.(last(v))
