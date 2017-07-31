struct BruteTree{V <: AbstractVector,M <: Metric} <: NNTree{V,M}
    data::Vector{V}
    metric::M
    reordered::Bool
end

"""
    BruteTree(data [, metric = Euclidean()) -> brutetree

Creates a `BruteTree` from the data using the given `metric`.
"""
function BruteTree(data::Vector{V}, metric::Metric = Euclidean();
                   reorder::Bool=false, leafsize::Int=0, storedata::Bool=true) where {V <: AbstractVector}
    BruteTree(storedata ? data : Vector{V}(), metric, reorder)
end

function BruteTree(data::Matrix{T}, metric::Metric = Euclidean();
                   reorder::Bool=false, leafsize::Int=0, storedata::Bool=true) where {T}
    dim = size(data, 1)
    npoints = size(data, 2)
    BruteTree(reinterpret(SVector{dim,T}, data, (length(data)  ÷ dim,)),
              metric, reorder = reorder, leafsize = leafsize, storedata = storedata)
end

function _knn(tree::BruteTree{V},
                 point::AbstractVector,
                 best_idxs::Vector{Int},
                 best_dists::Vector,
                 skip::Function) where {V}

    knn_kernel!(tree, point, best_idxs, best_dists, skip)
    return
end

function knn_kernel!(tree::BruteTree{V},
                     point::AbstractVector,
                     best_idxs::Vector{Int},
                     best_dists::Vector,
                     skip::F) where {V, F}
    for i in 1:length(tree.data)
        if skip(i)
            continue
        end

        @POINT 1
        dist_d = evaluate(tree.metric, tree.data[i], point)
        if dist_d <= best_dists[1]
            best_dists[1] = dist_d
            best_idxs[1] = i
            percolate_down!(best_dists, best_idxs, dist_d, i)
        end
    end
end

function _inrange(tree::BruteTree,
                  point::AbstractVector,
                  radius::Number,
                  idx_in_ball::Vector{Int})
    inrange_kernel!(tree, point, radius, idx_in_ball)
    return
end


function inrange_kernel!(tree::BruteTree,
                         point::AbstractVector,
                         r::Number,
                         idx_in_ball::Vector{Int})
    for i in 1:length(tree.data)
        @POINT 1
        d = evaluate(tree.metric, tree.data[i], point)
        if d <= r
            push!(idx_in_ball, i)
        end
    end
end
