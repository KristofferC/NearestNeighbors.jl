immutable HyperSphere{T <: AbstractFloat} <: HyperObject{T}
    center::Vector{T}
    r::T
end

@inline ndim(hs::HyperSphere) = length(hs.center)

@inline function intersects{T <: AbstractFloat}(m::Metric,
                                                s1::HyperSphere{T},
                                                s2::HyperSphere{T})
    evaluate(m, s1.center, s2.center) <= s1.r + s2.r
end

@inline function encloses{T <: AbstractFloat}(m::Metric,
                                              s1::HyperSphere{T},
                                              s2::HyperSphere{T})
    evaluate(m, s1.center, s2.center) + s1.r <= s2.r
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
    return HyperSphere(center, r)
end

# Creates a bounding sphere from two other spheres
function create_bsphere{T <: AbstractFloat}(m::Metric,
                                            s1::HyperSphere{T},
                                            s2::HyperSphere{T},
                                            array_buffs)
    @unpack array_buffs: left: left, right, v12, zerobuf
    # Create unitvector from s1 to s2
    @devec v12[:] = s2.center - s1.center
    invdist = one(T) / evaluate(m, v12, zerobuf)
    scale!(v12, invdist)

    # The two points furthest away from the center
    @devec left[:] = s1.center - v12 .* s1.r
    @devec right[:] = s2.center + v12 .* s2.r

    # r is half distance between edges
    rad = evaluate(m, left, right) * 0.5

    @devec center = (left + right) .* 0.5

    HyperSphere{T}(center, rad)
end
