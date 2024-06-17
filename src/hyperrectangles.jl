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
get_max_distance_sq(rec::HyperRectangle, point::AbstractVector) =
    sum(abs2(max(rec.maxes .- point, point .- rec.mins)))
=#

# Min distance between rectangle and point
get_min_distance_sq(rec::HyperRectangle, point::AbstractVector) =
    sum(abs2.(max.(0, max.(rec.mins .- point, point .- rec.maxes))))
