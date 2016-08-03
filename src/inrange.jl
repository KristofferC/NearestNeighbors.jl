"""
    inrange(tree::NNTree, points, radius [, sortres=false]) -> indices

Find all the points in the tree which is closer than `radius` to `points`. If
`sortres = true` the resulting indices are sorted.
"""
function inrange{T <: AbstractFloat}(tree::NNTree{T},
                                     points::AbstractArray{T},
                                     radius::Number,
                                     sortres=false)
    check_input(tree, points)

    if radius < 0
        throw(ArgumentError("the query radius r must be ≧ 0"))
    end

    idxs = Array(Vector{Int}, size(points, 2))
    point = zeros(T, size(points, 1))

    for i in 1:size(points, 2)
        for j in 1:size(points, 1)
            point[j] = points[j, i]
        end
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
    return do_return_inrange(idxs, points)
end

do_return_inrange(idxs, ::AbstractVector) = idxs[1]
do_return_inrange(idxs, ::AbstractMatrix) = idxs

function inrange{T <: AbstractFloat, P <: Real}(tree::NNTree{T},
                                                points::AbstractArray{P},
                                                radius::Number,
                                                sortres=false)
    inrange(tree, map(T, points), radius, sortres)
end
