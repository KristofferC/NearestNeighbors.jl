check_radius(r) = r < 0 && throw(ArgumentError("the query radius r must be ≧ 0"))

"""
    inrange(tree::NNTree, points, radius [, sortres=false, skip=always_false]) -> indices

Find all the points in the tree which is closer than `radius` to `points`. If
`sortres = true` the resulting indices are sorted. `skip` is an optional predicate
to determine if a point that would be returned should be skipped.
"""
function inrange(tree::NNTree,
                 points::Vector{T},
                 radius::Number,
                 sortres=false,
                 skip::F=always_false) where {T <: AbstractVector, F}
    check_input(tree, points)
    check_radius(radius)

    idxs = [Vector{Int}() for _ in 1:length(points)]

    for i in 1:length(points)
        inrange_point!(tree, points[i], radius, sortres, idxs[i], skip)
    end
    return idxs
end

function inrange_point!(tree, point, radius, sortres, idx, skip::F) where {F}
    _inrange(tree, point, radius, idx, skip)
    if tree.reordered
        @inbounds for j in 1:length(idx)
            idx[j] = tree.indices[idx[j]]
        end
    end
    sortres && sort!(idx)
    return
end

function inrange(tree::NNTree{V}, point::AbstractVector{T}, radius::Number, sortres=false, skip::F=always_false) where {V, T <: Number, F}
    check_input(tree, point)
    check_radius(radius)
    idx = Int[]
    inrange_point!(tree, point, radius, sortres, idx, skip)
    return idx
end

function inrange(tree::NNTree{V}, point::Matrix{T}, radius::Number, sortres=false, skip::F=always_false) where {V, T <: Number, F}
    dim = size(point, 1)
    npoints = size(point, 2)
    if isbitstype(T)
        new_data = copy_svec(T, point, Val(dim))
    else
        new_data = SVector{dim,T}[SVector{dim,T}(point[:, i]) for i in 1:npoints]
    end
    inrange(tree, new_data, radius, sortres, skip)
end
