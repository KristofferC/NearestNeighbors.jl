const NormMetric = Union{Euclidean,Chebyshev,Cityblock,Minkowski,WeightedEuclidean,WeightedCityblock,WeightedMinkowski,Mahalanobis}

struct HyperSphere{N,T <: AbstractFloat}
    center::SVector{N,T}
    r::T
end

HyperSphere(center::SVector{N,T1}, r) where {N, T1} = HyperSphere(center, convert(T1, r))
HyperSphere(center::AbstractVector, r) = HyperSphere(SVector{length(center)}(center), r)

function _infinite_hypersphere(::Type{HyperSphere{N,T}})  where {N, T}
    return HyperSphere{N,T}(
        ntuple(i->zero(T), Val(N)), 
        convert(T, Inf) 
    )
end 

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
                             d) where {V <: AbstractVector}
    alpha = x / d
    center = (1 - alpha) * c1 + alpha * c2
    return center, true
end

@inline function interpolate(::Metric,
                             c1::V,
                             ::V,
                             ::Any,
                             ::Any) where {V <: AbstractVector}
    return c1, false
end

# Computes a bounding sphere for a set of points
function create_bsphere(data::AbstractVector{V}, metric::Metric, indices::Vector{Int}, range) where {V}
    T = get_T(eltype(V))
    center = sum(data[indices[r]] for r in range) * (one(T) / length(range))
    r = maximum(evaluate(metric, data[indices[i]], center) for i in range)
    r += eps(T)
    return HyperSphere(center, r)
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

function distance_to_sphere(metric::Metric, point, sphere::HyperSphere)
    dist = evaluate(metric, point, sphere.center) - sphere.r
    return max(zero(eltype(dist)), dist)
end
