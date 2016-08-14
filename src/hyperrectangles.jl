# abstract HyperRectangle{N, T}

immutable HyperRectangle{T}
    mins::Vector{T}
    maxes::Vector{T}
end

# Computes a bounding box around a point cloud
function compute_bbox{V <: AbstractVector}(data::Vector{V})
    @assert length(data) != 0
    T = eltype(V)
    n_dim = length(V)
    maxes = Vector{T}(n_dim)
    mins = Vector{T}(n_dim)
    @inbounds for j in 1:length(V)
        dim_max = typemin(T)
        dim_min = typemax(T)
        for k in 1:length(data)
            dim_max = max(data[k][j], dim_max)
            dim_min = min(data[k][j], dim_min)
        end
        maxes[j] = dim_max
        mins[j] = dim_min
    end
    return HyperRectangle(mins, maxes)
end

# Splits a hyper rectangle into two rectangles by dividing the
# rectangle at a specific value in a given dimension.
function Base.split(hyper_rec::HyperRectangle,
               dim::Int,
               value::Number)
    new_max = copy(hyper_rec.maxes)
    new_max[dim] = value

    new_min = copy(hyper_rec.mins)
    new_min[dim] = value

    return HyperRectangle(hyper_rec.mins, new_max),
           HyperRectangle(new_min, hyper_rec.maxes)
end

function find_maxspread{T}(hyper_rec::HyperRectangle{T})
    # Find the dimension where we have the largest spread.
    split_dim = 1
    max_spread = zero(T)

    for d in eachindex(hyper_rec.mins)
        @inbounds spread = hyper_rec.maxes[d] - hyper_rec.mins[d]
        if spread > max_spread
            max_spread = spread
            split_dim = d
        end
    end
    return split_dim
end

############################################
# Rectangle - Point functions
############################################
@inline function get_min_dim(rec::HyperRectangle, point::AbstractVector, dim::Int)
    @inbounds d = abs2(max(0, max(rec.mins[dim] - point[dim], point[dim] - rec.maxes[dim])))
    d
end

@inline function get_max_dim(rec::HyperRectangle, point::AbstractVector, dim::Int)
    @inbounds d = abs2(max(rec.maxes[dim] - point[dim], point[dim] - rec.mins[dim]))
    d
end

# Max distance between rectangle and point
@inline function get_max_distance{T}(rec::HyperRectangle, point::AbstractVector{T})
    max_dist = zero(T)
    @inbounds @simd for dim in eachindex(point)
        max_dist += abs2(max(rec.maxes[dim] - point[dim], point[dim] - rec.mins[dim]))
    end
    return max_dist
end

# Min distance between rectangle and point
@inline function get_min_distance{T}(rec::HyperRectangle, point::AbstractVector{T})
    min_dist = zero(T)
    @inbounds @simd for dim in eachindex(point)
        min_dist += abs2(max(0, max(rec.mins[dim] - point[dim], point[dim] - rec.maxes[dim])))
    end
    return min_dist
end

# (Min, Max) distance between rectangle and point
@inline function get_min_max_distance{T}(rec::HyperRectangle, point::AbstractVector{T})
    min_dist = get_min_distance(rec, point)
    max_dist = get_max_distance(rec, point)
    return min_dist, max_dist
end

# (Min, Max) distance between rectangle and point for a certain dim
@inline function get_min_max_dim{T}(rec::HyperRectangle, point::AbstractVector{T}, dim::Int)
    min_dist_dim = get_min_dim(rec, point, dim)
    max_dist_dim = get_max_dim(rec, point, dim)
    return min_dist_dim, max_dist_dim
end
