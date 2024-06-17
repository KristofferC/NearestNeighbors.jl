# abstract HyperRectangle{N, T}

struct HyperRectangle{V <: AbstractVector}
    mins::V
    maxes::V
end

# Computes a bounding box around a point cloud
function compute_bbox(data::AbstractVector{V}) where {V <: AbstractVector}
    T = eltype(V)
    n_dim = length(V)
    maxes = zeros(MVector{n_dim, T})
    mins = zeros(MVector{n_dim, T})
    @inbounds for j in 1:n_dim
        dim_max = typemin(T)
        dim_min = typemax(T)
        for k in eachindex(data)
            dim_max = max(data[k][j], dim_max)
            dim_min = min(data[k][j], dim_min)
        end
        maxes[j] = dim_max
        mins[j] = dim_min
    end
    return HyperRectangle(SVector(mins), SVector(maxes))
end


#=
get_max_distance_sq(rec::HyperRectangle, point::AbstractVector) =
    sum(abs2(max(rec.maxes .- point, point .- rec.mins)))
=#

# Min distance between rectangle and point
get_min_distance_sq(rec::HyperRectangle, point::AbstractVector) =
    sum(abs2.(max.(0, max.(rec.mins .- point, point .- rec.maxes))))
