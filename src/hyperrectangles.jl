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

@inline function update_new_min(M::Metric, old_min, hyper_rec, p_dim, split_dim, split_val)
    @inbounds begin
        lo = hyper_rec.mins[split_dim]
        hi = hyper_rec.maxes[split_dim]
    end
    ddiff = distance_function_min(p_dim, hi, lo)
    split_diff = abs(p_dim - split_val)
    split_diff_pow = eval_pow(M, split_diff)
    ddiff_pow = eval_pow(M, ddiff)
    diff_tot = eval_diff(M, split_diff_pow, ddiff_pow, split_dim)
    return old_min + diff_tot
end

# Compute min and max possible distances between two hyper rectangles
function get_min_max_distance(m::Metric, r1::HyperRectangle{V}, r2::HyperRectangle{V}) where {V}
    p = Distances.parameters(m)
    T = eltype(V)
    min_acc = zero(T)
    max_acc = zero(T)
    @inbounds for dim in eachindex(r1.mins)
        lo1 = r1.mins[dim]; hi1 = r1.maxes[dim]
        lo2 = r2.mins[dim]; hi2 = r2.maxes[dim]
        min_raw = if hi1 < lo2
            lo2 - hi1
        elseif hi2 < lo1
            lo1 - hi2
        else
            zero(T)
        end
        max_raw = max(abs(hi1 - lo2), abs(hi2 - lo1))
        min_op = p === nothing ? eval_op(m, min_raw, zero(T)) : eval_op(m, min_raw, zero(T), p[dim])
        max_op = p === nothing ? eval_op(m, max_raw, zero(T)) : eval_op(m, max_raw, zero(T), p[dim])
        min_acc = eval_reduce(m, min_acc, min_op)
        max_acc = eval_reduce(m, max_acc, max_op)
    end
    return min_acc, max_acc
end

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
