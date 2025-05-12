struct BruteTree{V <: AbstractVector,M <: PreMetric} <: NNTree{V,M}
    data::Vector{V}
    metric::M
    reordered::Bool
end

"""
    BruteTree(data [, metric = Euclidean()) -> brutetree

Creates a `BruteTree` from the data using the given `metric`.
"""
function BruteTree(data::AbstractVector{V}, metric::PreMetric = Euclidean();
                   reorder::Bool=false, leafsize::Int=0, storedata::Bool=true) where {V <: AbstractVector}
    if metric isa Distances.UnionMetrics
        p = parameters(metric)
        if p !== nothing && length(p) != length(V)
           throw(ArgumentError(
               "dimension of input points:$(length(V)) and metric parameter:$(length(p)) must agree"))
        end
    end

    BruteTree(storedata ? Vector(data) : Vector{V}(), metric, reorder)
end

function BruteTree(data::AbstractVecOrMat{T}, metric::PreMetric = Euclidean();
                   reorder::Bool=false, leafsize::Int=0, storedata::Bool=true) where {T}
    dim = size(data, 1)
    BruteTree(copy_svec(T, data, Val(dim)),
              metric, reorder = reorder, leafsize = leafsize, storedata = storedata)
end

function _knn(tree::BruteTree{V},
                 point::AbstractVector,
                 best_idxs::AbstractVector{<:Integer},
                 best_dists::AbstractVector,
                 skip::F) where {V, F}

    knn_kernel!(tree, point, best_idxs, best_dists, skip)
    return
end

function knn_kernel!(tree::BruteTree{V},
                     point::AbstractVector,
                     best_idxs::AbstractVector{<:Integer},
                     best_dists::AbstractVector,
                     skip::F) where {V, F}
    for i in 1:length(tree.data)
        if skip(i)
            continue
        end

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
                  idx_in_ball::Union{Nothing, Vector{<:Integer}},
                  point_index::Int = 1,
                  runtime_function::Union{Nothing, Function} = nothing)
    return inrange_kernel!(tree, point, radius, idx_in_ball, runtime_function, point_index)
end


function inrange_kernel!(tree::BruteTree,
                         point::AbstractVector,
                         r::Number,
                         idx_in_ball::Union{Nothing, Vector{<:Integer}},
                         runtime_function::Union{Nothing, Function},
                         point_index::Int)
    count = 0
    for i in 1:length(tree.data)
        d = evaluate(tree.metric, tree.data[i], point)
        if d <= r
            count += 1
            idx_in_ball !== nothing && push!(idx_in_ball, i)
            !isnothing(runtime_function) && runtime_function(point_index, i, point)
        end
    end
    return count
end
