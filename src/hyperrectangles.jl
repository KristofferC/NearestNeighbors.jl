struct HyperRectangle{V <: AbstractVector}
    mins::V
    maxes::V
end

function compute_bbox(data::AbstractVector{V}) where {V <: AbstractVector}
    mins = mapreduce(identity, (a, b) -> min.(a, b), data; init=fill(Inf,V))
    maxes = mapreduce(identity, (a, b) -> max.(a, b), data; init=fill(-Inf,V))
    return HyperRectangle(mins, maxes)
end

#=
function get_max_distance_no_end(d::Metric, rec::HyperRectangle, point::AbstractVector{T}) where {T}
    s = zero(T)
    @inbounds @simd for dim in eachindex(point)
        z = max(abs(rec.maxes[dim] - point[dim]), abs(point[dim] - rec.mins[dim]))
        s = eval_reduce(d, s, eval_op(d, z, zero(T)))
    end
    return s
end
=#

function get_min_distance_no_end(d::Metric, rec::HyperRectangle, point::AbstractVector{T}) where {T}
    s = zero(T)
    @inbounds @simd for dim in eachindex(point)
        z = max(0, max(rec.mins[dim] - point[dim], point[dim] - rec.maxes[dim]))
        s = eval_reduce(d, s, eval_op(d, z, zero(T)))
    end
    return s
end
