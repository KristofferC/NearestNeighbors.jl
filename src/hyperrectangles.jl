# abstract HyperRectangle{N, T}

struct HyperRectangle{T}
    mins::Vector{T}
    maxes::Vector{T}
end

# Computes a bounding box around a point cloud
function compute_bbox(data::Vector{V}) where {V <: AbstractVector}
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
@inline function get_max_distance(rec::HyperRectangle, point::AbstractVector{T}) where {T}
    max_dist = zero(T)
    @inbounds @simd for dim in eachindex(point)
        max_dist += abs2(max(rec.maxes[dim] - point[dim], point[dim] - rec.mins[dim]))
    end
    return max_dist
end

# Min distance between rectangle and point
@inline function get_min_distance(rec::HyperRectangle, point::AbstractVector{T}) where {T}
    min_dist = zero(T)
    @inbounds @simd for dim in eachindex(point)
        min_dist += abs2(max(0, max(rec.mins[dim] - point[dim], point[dim] - rec.maxes[dim])))
    end
    return min_dist
end
