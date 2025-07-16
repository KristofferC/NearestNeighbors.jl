struct HyperRectangle{V <: AbstractVector}
    mins::V
    maxes::V
end

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

@inline distance_function_max(vald, maxd, mind) = max(abs(maxd - vald), abs(vald - mind))
@inline distance_function_min(vald, maxd, mind) = max(zero(eltype(vald)), max(mind - vald, vald - maxd))

function get_min_max_distance_no_end(f::Function, m::Metric, rec::HyperRectangle, point::AbstractVector{T}) where {T}
    s = zero(T)
    p = Distances.parameters(m)
    @inbounds @simd for dim in eachindex(point)
        v = f(point[dim], rec.maxes[dim], rec.mins[dim])
        v_op = p === nothing ? eval_op(m, v, zero(T)) : eval_op(m, v, zero(T), p[dim])
        s = eval_reduce(m, s, v_op)
    end
    return s
end

get_max_distance_no_end(m, rec, point) =
    get_min_max_distance_no_end(distance_function_max, m, rec, point)

get_min_distance_no_end(m, rec, point) =
    get_min_max_distance_no_end(distance_function_min, m, rec, point)
