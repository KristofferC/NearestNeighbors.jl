const NormMetric = Union{Euclidean,Chebyshev,Cityblock,Minkowski,WeightedEuclidean,WeightedCityblock,WeightedMinkowski,Mahalanobis}

struct HyperSphere{N,T <: AbstractFloat}
    center::SVector{N,T}
    r::T
end

HyperSphere(center::SVector{N,T1}, r) where {N, T1} = HyperSphere(center, convert(T1, r))
HyperSphere(center::AbstractVector, r) = HyperSphere(SVector{length(center)}(center), r)

@inline function intersects(m::Metric,
                            s1::HyperSphere{N},
                            s2::HyperSphere{N}) where {N}
    evaluate(m, s1.center, s2.center) <= s1.r + s2.r
end

@inline function encloses(m::Metric,
                          s1::HyperSphere{N},
                          s2::HyperSphere{N}) where {N}
    evaluate(m, s1.center, s2.center) + s1.r <= s2.r
end

@inline function interpolate(::NormMetric,
                             c1::V,
                             c2::V,
                             x,
                             d,
                             ab) where {V <: AbstractVector}
    alpha = x / d
    @assert length(c1) == length(c2)
    @inbounds for i in eachindex(ab.center)
        ab.center[i] = (1 - alpha) .* c1[i] + alpha .* c2[i]
    end
    return ab.center, true
end

@inline function interpolate(::Metric,
                             c1::V,
                             ::V,
                             ::Any,
                             ::Any) where {V <: AbstractVector}
    return c1, false
end

function create_bsphere(data::AbstractVector{V}, metric::Metric, indices::Vector{Int}, low, high) where {V}
    T = get_T(eltype(V))
    n_points = high - low + 1
    # First find center of all points
    center = zero(SVector{length(V),T})
    @inbounds for i in low:high
        center += data[indices[i]]
    end
    center *= one(T) / n_points

    # Then find r
    r = zero(T)
    @inbounds for i in low:high
        r = max(r, evaluate(metric, data[indices[i]], center))
    end
    r += eps(T)
    return HyperSphere(SVector{length(V),eltype(V)}(center), r)
end

# Creates a bounding sphere from two other spheres
function create_bsphere(m::Metric,
                        s1::HyperSphere{N,T},
                        s2::HyperSphere{N,T}) where {N, T <: AbstractFloat}
    if encloses(m, s1, s2)
        return HyperSphere(s2.center, s2.r)
    elseif encloses(m, s2, s1)
        return HyperSphere(s1.center, s1.r)
    end

    # Compute the distance x along a geodesic from s1.center to s2.center
    # where the new center should be placed (note that 0 <= x <= d because
    # neither s1 nor s2 contains the other)
    dist = evaluate(m, s1.center, s2.center)
    x = (s2.r - s1.r + dist) / 2
    center, is_exact_center = interpolate(m, s1.center, s2.center, x, dist)
    if is_exact_center
        rad = (s2.r + s1.r + dist) / 2
    else
        rad = max(s1.r + evaluate(m, s1.center, center), s2.r + evaluate(m, s2.center, center))
    end

    return HyperSphere(SVector{N,T}(center), rad)
end
