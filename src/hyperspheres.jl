const NormMetric = Union{Euclidean,Chebyshev,Cityblock,Minkowski,WeightedEuclidean,WeightedCityblock,WeightedMinkowski,Mahalanobis}

struct HyperSphere{N,T <: AbstractFloat}
    center::SVector{N,T}
    r::T
end

HyperSphere(center::SVector{N,T1}, r::T2) where {N, T1, T2} = HyperSphere(center, convert(T1, r))

Base.:(==)(A::HyperSphere, B::HyperSphere) = A.center == B.center && A.r == B.r

@inline function intersects(m::M,
                            s1::HyperSphere{N,T},
                            s2::HyperSphere{N,T}) where {T <: AbstractFloat, N, M <: Metric}
    evaluate(m, s1.center, s2.center) <= s1.r + s2.r
end

@inline function encloses(m::M,
                          s1::HyperSphere{N,T},
                          s2::HyperSphere{N,T}) where {T <: AbstractFloat, N, M <: Metric}
    evaluate(m, s1.center, s2.center) + s1.r <= s2.r
end

# versions with no array buffer - still not allocating in sequential BallTree construction
using Statistics: mean
function create_bsphere(data::AbstractVector{V}, metric::Metric, indices::Vector{Int}, low, high) where {V}
    # find center
    center = mean(@views(data[indices[low:high]]))
    # Then find r
    r = zero(get_T(eltype(V)))
    for i in low:high
        r = max(r, evaluate(metric, data[indices[i]], center))
    end
    r += eps(get_T(eltype(V)))
    return HyperSphere(SVector{length(V),eltype(V)}(center), r)
end

# Creates a bounding sphere from two other spheres
function create_bsphere(m::Metric, s1::HyperSphere{N,T}, s2::HyperSphere{N,T}) where {N, T <: AbstractFloat}
    if encloses(m, s1, s2)
        return HyperSphere(s2.center, s2.r)
    elseif encloses(m, s2, s1)
        return HyperSphere(s1.center, s1.r)
    end

    # Compute the distance x along a geodesic from s1.center to s2.center
    # where the new center should be placed (note that 0 <= x <= d because
    # neither s1 nor s2 contains the other)
    dist = evaluate(m, s1.center, s2.center)
    x = 0.5 * (s2.r - s1.r + dist)
    center, is_exact_center = interpolate(m, s1.center, s2.center, x, dist)
    if is_exact_center
        rad = 0.5 * (s2.r + s1.r + dist)
    else
        rad = max(s1.r + evaluate(m, s1.center, center), s2.r + evaluate(m, s2.center, center))
    end

    return HyperSphere(SVector{N,T}(center), rad)
end

@inline function interpolate(::M, c1::V, c2::V, x, d) where {V <: AbstractVector, M <: NormMetric}
    length(c1) == length(c2) || throw(DimensionMismatch("interpolate arguments have length $(length(c1)) and $(length(c2))"))
    alpha = x / d
    center = (1 - alpha) * c1 + alpha * c2
    return center, true
end

@inline function interpolate(::M, c1::V, ::V, ::Any, ::Any) where {V <: AbstractVector, M <: Metric}
    return c1, false
end
