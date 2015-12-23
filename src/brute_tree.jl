immutable BruteTree{T <: Real, M <: Metric} <: NNTree{T, M}
    data::Matrix{T}
    metric::M
    leafsize::Int
    reordered::Bool
end

"""
    BruteTree(data [, metric = Euclidean()) -> brutetree

Creates a `BruteTree` from the data using the given `metric`.
"""
function BruteTree{T <: Real}(data::Matrix{T}, metric::Metric=Euclidean();
                              reorder::Bool=false, leafsize=0, storedata::Bool=true)
    BruteTree(storedata ? data : similar(data,0,0), metric, 0, reorder)
end

function _knn{T, P}(tree::BruteTree{T},
                    point::AbstractVector{P},
                    k::Int)
    Tret = Distances.result_type(tree.metric, tree.data, point)
    best_idxs = [-1 for _ in 1:k]
    best_dists = [typemax(Tret) for _ in 1:k]
    knn_kernel!(tree, point, best_idxs, best_dists)
    return best_idxs, best_dists
end

function knn_kernel!{T, Tret, P}(tree::BruteTree{T},
                                 point::AbstractArray{P},
                                 best_idxs::Vector{Int},
                                 best_dists::Vector{Tret})

    for i in 1:size(tree.data, 2)
        @POINT 1
        dist_d = evaluate(tree.metric, tree.data, point, i)
        if dist_d <= best_dists[1]
            best_dists[1] = dist_d
            best_idxs[1] = i
            percolate_down!(best_dists, best_idxs, dist_d, i)
        end
    end
end

function _inrange{T, P}(tree::BruteTree{T},
                     point::AbstractVector{P},
                     radius::Number)
    idx_in_ball = Int[]
    inrange_kernel!(tree, point, radius, idx_in_ball)
    return idx_in_ball
end


function inrange_kernel!{T, P}(tree::BruteTree{T},
                            point::Vector{P},
                            r::Number,
                            idx_in_ball::Vector{Int})
    for i in 1:size(tree.data, 2)
        @POINT 1
        d = evaluate(tree.metric, tree.data, point, i)
        if d <= r
            push!(idx_in_ball, i)
        end
    end
end
