struct BruteTree{V <: AbstractVector,M <: PreMetric} <: NNTree{V,M}
    data::Vector{V}
    metric::M
    reordered::Bool
end

"""
    BruteTree(data [, metric = Euclidean()])::Brutetree

Creates a `BruteTree` from the data using the given `metric`.

# Arguments
- `data`: Point data as a matrix of size `nd × np` or vector of vectors
- `metric`: Distance metric to use (can be any `PreMetric` from Distances.jl). Default: `Euclidean()`

# Returns
- `brutetree`: A `BruteTree` instance

BruteTree performs exhaustive linear search and is useful as a baseline or for small datasets.
Note: `leafsize` and `reorder` parameters are ignored for BruteTree.
"""
function BruteTree(data::AbstractVector{V}, metric::PreMetric = Euclidean();
                   reorder::Bool=false, leafsize::Int=0, storedata::Bool=true) where {V <: AbstractVector}
    check_for_nan(data)
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
    check_for_nan(data)
    dim = size(data, 1)
    BruteTree(copy_svec(T, data, Val(dim)),
              metric, reorder = reorder, leafsize = leafsize, storedata = storedata)
end

function _knn(tree::BruteTree{V},
                 point::AbstractVector,
                 best_idxs::Union{Integer, AbstractVector{<:Integer}},
                 best_dists::Union{Number, AbstractVector},
                 ::Union{Nothing, AbstractVector},
                 skip::F) where {V, F}

    return knn_kernel!(tree, point, best_idxs, best_dists, skip, nothing)
end

function knn_kernel!(tree::BruteTree{V},
                     point::AbstractVector,
                     best_idxs::Union{Integer, AbstractVector{<:Integer}},
                     best_dists::Union{Number, AbstractVector},
                     skip::F,
                     dedup::MaybeBitSet) where {V, F}
    has_set = dedup !== nothing
    for i in 1:length(tree.data)
        if skip(i)
            continue
        end

        dist_d = evaluate(tree.metric, tree.data[i], point)
        update_existing_neighbor!(dedup, i, dist_d, best_idxs, best_dists) && continue
        best_dist_1 = first(best_dists)
        if dist_d <= best_dist_1
            has_set && push!(dedup, i)
            best_dists = maybe_update_index(best_dists, 1, dist_d)
            best_idxs = maybe_update_index(best_idxs, 1, i)
            best_dists isa AbstractVector && percolate_down!(best_dists, best_idxs, dist_d, i)
        end
    end
    return best_idxs, best_dists
end

function _inrange(tree::BruteTree,
                  point::AbstractVector,
                  radius::Number,
                  idx_in_ball::Union{Nothing, AbstractVector{<:Integer}},
                  skip::F,) where {F}
    return inrange_kernel!(tree, point, radius, idx_in_ball, skip, nothing)
end


function inrange_kernel!(tree::BruteTree,
                         point::AbstractVector,
                         r::Number,
                         idx_in_ball::Union{Nothing, AbstractVector{<:Integer}},
                         skip::Function,
                         dedup::MaybeBitSet)
    count = 0
    has_set = dedup !== nothing
    for i in 1:length(tree.data)
        if skip(i)
            continue
        end
        d = evaluate(tree.metric, tree.data[i], point)
        if d <= r
            if has_set && i in dedup
                continue
            end
            has_set && push!(dedup, i)
            count += 1
            idx_in_ball !== nothing && push!(idx_in_ball, i)
        end
    end
    return count
end

function _inrange_pairs(tree::BruteTree{V}, radius::Number, sortres, skip::F) where {V, F}
    pairs = NTuple{2,Int}[]
    for i in 1:length(tree.data)
        skip(i) && continue
        for j in (i + 1):length(tree.data)
            skip(j) && continue
            d = evaluate(tree.metric, tree.data[i], tree.data[j])
            if d <= radius
                push!(pairs, (i, j))
            end
        end
    end
    sortres && sort!(pairs)
    return pairs
end
