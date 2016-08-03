typealias NormMetric Union{Euclidean, Chebyshev, Cityblock, Minkowski, WeightedEuclidean, WeightedCityblock, WeightedMinkowski, Mahalanobis}

immutable HyperSphere{T <: AbstractFloat}
    center::Vector{T}
    r::T
end

HyperSphere{T <: AbstractFloat}(center::Vector{T}, r) = HyperSphere(center, T(r))

@inline ndim(hs::HyperSphere) = length(hs.center)

@inline function intersects{T <: AbstractFloat, M <: Metric}(m::M,
                                                             s1::HyperSphere{T},
                                                             s2::HyperSphere{T})
    evaluate(m, s1.center, s2.center) <= s1.r + s2.r
end

@inline function encloses{T <: AbstractFloat, M <: Metric}(m::M,
                                                           s1::HyperSphere{T},
                                                           s2::HyperSphere{T})
    evaluate(m, s1.center, s2.center) + s1.r <= s2.r
end

@inline function interpolate{T <: AbstractFloat, M <: NormMetric}(m::M,
                                                                  c1::Vector{T},
                                                                  c2::Vector{T},
                                                                  x,
                                                                  d)
    alpha = x / d
    @assert length(c1) == length(c2)
    c = similar(c1)
    @inbounds for i in eachindex(c)
        c[i] = (1 - alpha) .* c1[i] + alpha .* c2[i]
    end
    return c, true
end

@inline function interpolate{T <: AbstractFloat, M <: Metric}(m::M,
                                                              c1::Vector{T},
                                                              c2::Vector{T},
                                                              x,
                                                              d)
    return copy(c1), false
end

function create_bsphere{T}(data::Matrix{T}, metric::Metric, indices::Vector{Int}, low, high)
    n_dim = size(data,1)
    n_points = high - low + 1

    # First find center of all points
    center = zeros(T, n_dim)
    for i in low:high
       for j in 1:n_dim
           center[j] += data[j, indices[i]]
       end
    end
    scale!(center, 1 / n_points)

    # Then find r
    r = zero(T)
    for i in low:high
        r = max(r, evaluate(metric, data, center, indices[i]))
    end
    r += eps(T)
    return HyperSphere(center, r)
end

# Creates a bounding sphere from two other spheres
function create_bsphere{T <: AbstractFloat}(m::Metric,
                                            s1::HyperSphere{T},
                                            s2::HyperSphere{T},
                                            ab)
    if encloses(m, s1, s2)
        return HyperSphere{T}(copy(s2.center), s2.r)
    elseif encloses(m, s2, s1)
        return HyperSphere{T}(copy(s1.center), s1.r)
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

    HyperSphere{T}(center, rad)
end
