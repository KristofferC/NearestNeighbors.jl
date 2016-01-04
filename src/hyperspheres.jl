immutable HyperSphere{T <: AbstractFloat}
    center::Vector{T}
    r::T
end

HyperSphere{T <: AbstractFloat}(center::Vector{T}, r) = HyperSphere(center, T(r))

@inline get_hstype(::Integer) = Float64
@inline get_hstype{T <: AbstractFloat}(::T) = T


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

function create_bsphere{T <: Real}(data::Matrix{T}, metric::Metric, indices::Vector{Int}, low, high)
    n_dim = size(data,1)
    n_points = high - low + 1
    Ths = get_hstype(data[1, 1])


    # First find center of all points
    center = zeros(Ths, n_dim)
    for i in low:high
       for j in 1:n_dim
           center[j] += data[j, indices[i]]
       end
    end
    scale!(center, 1 / n_points)

    # Then find r
    r = zero(Ths)
    for i in low:high
        r = max(r, evaluate(metric, data, center, indices[i]))
    end
    r += eps(Ths)
    return HyperSphere{Ths}(center, r)
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

    # Create unitvector from s1 to s2
    @devec ab.v12[:] = s2.center - s1.center
    invdist = 1 / evaluate(m, ab.v12, ab.zerobuf)
    scale!(ab.v12, invdist)

    # The two points furthest away from the center
    @devec ab.left[:] = s1.center - ab.v12 .* s1.r
    @devec ab.right[:] = s2.center + ab.v12 .* s2.r

    # r is half distance between edges
    rad = evaluate(m, ab.left, ab.right) * 0.5
    @devec center = (ab.left + ab.right) .* 0.5

    HyperSphere{T}(center, rad)
end
