@inline eval_pow(::MinkowskiMetric, s) = abs(s)
@inline eval_pow(::Euclidean, s) = abs2(s)
@inline eval_pow(d::Minkowski, s) = abs(s)^d.p

@inline eval_diff(::MinkowskiMetric, a, b) = a - b
@inline eval_diff(::Chebyshev, ::Any, b) = b

function Distances.evaluate(d::Distances.UnionMetrics, a::AbstractVector,
                            b::AbstractVector, do_end::Bool)
    p = Distances.parameters(d)
    s = eval_start(d, a, b)
    if p === nothing
        @simd for i in eachindex(b)
            @inbounds ai = a[i]
            @inbounds bi = b[i]
            s = eval_reduce(d, s, eval_op(d, ai, bi))
        end
    else
        @simd for i in eachindex(b)
            @inbounds ai = a[i]
            @inbounds bi = b[i]
            @inbounds pi = p[i]
            s = eval_reduce(d, s, eval_op(d, ai, bi, pi))
        end
    end
    if do_end
        return eval_end(d, s)
    else
        return s
    end
end

function Distances.evaluate(d::Distances.PreMetric, a::AbstractVector,
                            b::AbstractVector, ::Bool)
    evaluate(d, a, b)
end
