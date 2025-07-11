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

# Compute per-dimension contributions for max distance
function get_max_distance_contributions(m::Metric, rec::HyperRectangle{V}, point::AbstractVector{T}) where {V,T}
    p = Distances.parameters(m)
    return V(
        @inbounds begin
                v = distance_function_max(point[dim], rec.maxes[dim], rec.mins[dim])
                p === nothing ? eval_op(m, v, zero(T)) : eval_op(m, v, zero(T), p[dim])
            end for dim in eachindex(point)
    )
end

# Compute single dimension contribution for max distance
function get_max_distance_contribution_single(m::Metric, point_dim, min_bound::T, max_bound::T, dim::Integer) where {T}
    v = distance_function_max(point_dim, max_bound, min_bound)
    p = Distances.parameters(m)
    return p === nothing ? eval_op(m, v, zero(T)) : eval_op(m, v, zero(T), p[dim])
end

get_min_distance_no_end(m, rec, point) =
    get_min_max_distance_no_end(distance_function_min, m, rec, point)

# Combine all per-dimension contributions into final distance
function eval_reduce_all(m::Metric, contributions::SVector{N, T}) where {N, T}
    if m isa Chebyshev
        return maximum(contributions)
    else
        # For Lp norms, sum all contributions
        s = zero(T)
        for contrib in contributions
            s = eval_reduce(m, s, contrib)
        end
        return s
    end
end

# O(1) incremental update: remove old contribution, add new contribution
function update_max_distance_incremental(m::Metric, current_max_dist, old_contrib, new_contrib)
    # For Lp norms: current_max_dist - old_contrib + new_contrib
    temp = eval_reduce_inv(m, current_max_dist, old_contrib)  # Remove old
    return eval_reduce(m, temp, new_contrib)                  # Add new
end

# Inverse of eval_reduce for Lp norms (subtract contribution)
function eval_reduce_inv(m::Metric, current_sum, contrib_to_remove)
    # For Lp norms, this is just subtraction since eval_reduce is addition
    return current_sum - contrib_to_remove
end

function update_min_distance_no_end(m::Metric, current_dist, point::AbstractVector{T},
        old_min, old_max, new_min, new_max, dim
    ) where {T}
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
