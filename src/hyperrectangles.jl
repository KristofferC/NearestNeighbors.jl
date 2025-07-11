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

function update_distance_no_end(distance_func::Function, m::Metric, current_dist, point::AbstractVector{T},
                                old_min, old_max, new_min, new_max, dim::Integer) where {T}
    p_dim = point[dim]

    if new_min != old_min
        # Min boundary changed - split_val is the new min
        split_val = new_min
        split_diff = p_dim - split_val
        if split_diff > 0
            # Point is to the right of split_val
            ddiff = max(zero(T), p_dim - new_max)
        else
            # Point is to the left of split_val
            ddiff = max(zero(T), old_min - p_dim)
        end
    else
        # Max boundary changed - split_val is the new max
        split_val = new_max
        split_diff = p_dim - split_val
        if split_diff > 0
            # Point is to the right of split_val
            ddiff = max(zero(T), p_dim - old_max)
        else
            # Point is to the left of split_val
            ddiff = max(zero(T), new_min - p_dim)
        end
    end

    split_diff_pow = eval_pow(m, split_diff)
    ddiff_pow = eval_pow(m, ddiff)
    diff_tot = eval_diff(m, split_diff_pow, ddiff_pow, dim)

    return eval_reduce(m, current_dist, diff_tot)
end
update_min_distance_no_end(m::Metric, min_dist, point::AbstractVector, old_min, old_max, new_min, new_max, dim::Integer) =
    update_distance_no_end(distance_function_min, m, min_dist, point, old_min, old_max, new_min, new_max, dim)
