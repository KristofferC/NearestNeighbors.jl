"""
    inrange(tree::NNTree, points, radius [, sortres=false]) -> indices

Find all the points in the tree which is closer than `radius` to `points`. If
`sortres = true` the resulting indices are sorted.
"""
function inrange{T <: AbstractVector}(tree::NNTree,
                                      points::Vector{T},
                                      radius::Number,
                                      sortres=false)
    check_input(tree, points)

    if radius < 0
        throw(ArgumentError("the query radius r must be ≧ 0"))
    end

    idxs = Array(Vector{Int}, length(points))

    for i in 1:length(points)
        point = points[i]
        idx_in_ball = _inrange(tree, point, radius)
        if tree.reordered
            @inbounds for j in 1:length(idx_in_ball)
                idx_in_ball[j] = tree.indices[idx_in_ball[j]]
            end
        end
        if sortres
            sort!(idx_in_ball)
        end
        idxs[i] = idx_in_ball
    end
    return idxs
end

function inrange{V, T <: Number}(tree::NNTree{V}, point::AbstractVector{T}, radius::Number, sortres=false)
    idxs = inrange(tree, Vector{T}[point], radius, sortres)
    return idxs[1]
end

function inrange{V, T <: Number}(tree::NNTree{V}, point::Vector{T}, radius::Number, sortres=false)
    idxs = inrange(tree, [convert(SVector{length(point), T}, point)], radius, sortres)
    return idxs[1]
end

function inrange{V, T <: Number}(tree::NNTree{V}, point::Matrix{T}, radius::Number, sortres=false)
    dim = size(point, 1)
    npoints = size(point, 2)
    if isbits(T)
        new_data = reinterpret(SVector{dim, T}, point, (length(point) ÷ dim, ))
    else
        new_data = SVector{dim, T}[SVector{dim, T}(point[:, i]) for i in 1:npoints]
    end
    inrange(tree, new_data, radius, sortres)
end
