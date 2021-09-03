check_radius(r) = r < 0 && throw(ArgumentError("the query radius r must be ≧ 0"))

"""
    inrange(tree::NNTree, points, radius [, sortres=false]) -> indices

Find all the points in the tree which is closer than `radius` to `points`. If
`sortres = true` the resulting indices are sorted.
"""
function inrange(tree::NNTree,
                 points::Vector{T},
                 radius::Number,
                 sortres=false) where {T <: AbstractVector}
    check_input(tree, points)
    check_radius(radius)

    idxs = [Vector{Int}() for _ in 1:length(points)]

    for i in 1:length(points)
        inrange_point!(tree, points[i], radius, sortres, idxs[i])
    end
    return idxs
end

function inrange_point!(tree, point, radius, sortres, idx)
    count = _inrange(tree, point, radius, idx)
    if idx !== nothing
        if tree.reordered
            @inbounds for j in 1:length(idx)
                idx[j] = tree.indices[idx[j]]
            end
        end
        sortres && sort!(idx)
    end
    return count
end

function inrange(tree::NNTree{V}, point::AbstractVector{T}, radius::Number, sortres=false) where {V, T <: Number}
    check_input(tree, point)
    check_radius(radius)
    idx = Int[]
    inrange_point!(tree, point, radius, sortres, idx)
    return idx
end

function inrange(tree::NNTree{V}, point::AbstractMatrix{T}, radius::Number, sortres=false) where {V, T <: Number}
    dim = size(point, 1)
    npoints = size(point, 2)
    if isbitstype(T)
        new_data = copy_svec(T, point, Val(dim))
    else
        new_data = SVector{dim,T}[SVector{dim,T}(point[:, i]) for i in 1:npoints]
    end
    inrange(tree, new_data, radius, sortres)
end

"""
    inrangecount(tree::NNTree, points, radius) -> count

Count all the points in the tree which is closer than `radius` to `points`.
"""
function inrangecount(tree::NNTree{V}, point::AbstractVector{T}, radius::Number) where {V, T <: Number}
    check_input(tree, point)
    check_radius(radius)
    return inrange_point!(tree, point, radius, false, nothing)
end

function inrangecount(tree::NNTree,
        points::Vector{T},
        radius::Number) where {T <: AbstractVector}
    check_input(tree, points)
    check_radius(radius)
    return inrange_point!.(Ref(tree), points, radius, false, nothing)
end

function inrangecount(tree::NNTree{V}, point::AbstractMatrix{T}, radius::Number) where {V, T <: Number}
    dim = size(point, 1)
    npoints = size(point, 2)
    if isbitstype(T)
        new_data = copy_svec(T, point, Val(dim))
    else
        new_data = SVector{dim,T}[SVector{dim,T}(point[:, i]) for i in 1:npoints]
    end
    return inrangecount(tree, new_data, radius)
end
