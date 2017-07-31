const NormMetric = Union{Euclidean,Chebyshev,Cityblock,Minkowski,WeightedEuclidean,WeightedCityblock,WeightedMinkowski,Mahalanobis}

struct HyperSphere{N,T <: AbstractFloat}
    center::SVector{N,T}
    r::T
end

HyperSphere(center::SVector{N,T1}, r::T2) where {N, T1, T2} = HyperSphere(center, convert(T1, r))

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

@inline function interpolate(::M,
                             c1::V,
                             c2::V,
                             x,
                             d,
                             ab) where {V <: AbstractVector, M <: NormMetric}
    alpha = x / d
    @assert length(c1) == length(c2)
    @inbounds for i in eachindex(ab.center)
        ab.center[i] = (1 - alpha) .* c1[i] + alpha .* c2[i]
    end
    return ab.center, true
end

@inline function interpolate(::M,
                             c1::V,
                             ::V,
                             ::Any,
                             ::Any,
                             ::Any) where {V <: AbstractVector, M <: Metric}
    return c1, false
end

function create_bsphere(data::Vector{V}, metric::Metric, indices::Vector{Int}, low, high, ab) where {V}
    n_dim = size(data, 1)
    n_points = high - low + 1
    # First find center of all points
    fill!(ab.center, 0.0)
    for i in low:high
        for j in 1:length(ab.center)
            ab.center[j] += data[indices[i]][j]
        end
    end
    scale!(ab.center, 1 / n_points)

    # Then find r
    r = zero(get_T(eltype(V)))
    for i in low:high
        r = max(r, evaluate(metric, data[indices[i]], ab.center))
    end
    r += eps(get_T(eltype(V)))
    return HyperSphere(SVector{length(V),eltype(V)}(ab.center), r)
end

# Creates a bounding sphere from two other spheres
function create_bsphere(m::Metric,
                        s1::HyperSphere{N,T},
                        s2::HyperSphere{N,T},
                        ab) where {N, T <: AbstractFloat}
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
    center, is_exact_center = interpolate(m, s1.center, s2.center, x, dist, ab)
    if is_exact_center
        rad = 0.5 * (s2.r + s1.r + dist)
    else
        rad = max(s1.r + evaluate(m, s1.center, center), s2.r + evaluate(m, s2.center, center))
    end

    return HyperSphere(SVector{N,T}(center), rad)
end
