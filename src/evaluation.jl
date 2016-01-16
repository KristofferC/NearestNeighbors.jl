typealias MinkowskiMetric Union{Euclidean, Chebyshev, Cityblock, Minkowski}

@inline eval_start(m::Metric, a::AbstractMatrix, b::AbstractArray, col::Int) = eval_start(m, a, b)
@inline eval_start(::Chebyshev, a::AbstractMatrix, b::AbstractArray, col::Int) = abs(a[1,col] - b[1])
@inline function eval_start(::SpanNormDist, a::AbstractArray, b::AbstractArray, col::Int)
    a[1,col] - b[1], a[1, col]- b[1]
end

function Distances.evaluate(d::Distances.UnionMetrics, a::AbstractMatrix,
                            b::AbstractArray, col::Int, do_end::Bool=true)
    s = eval_start(d, a, b, col)
    @simd for I in eachindex(b)
        @inbounds ai = a[I, col]
        @inbounds bi = b[I]
        s = eval_reduce(d, s, eval_op(d, ai, bi))
    end
    if do_end
        return eval_end(d, s)
    else
        return s
    end
end

# As above but stops evaluating after cumulutative sum gets larger than
# break_at
# TODO: Use this in the inrange methods
function Distances.evaluate(d::Distances.UnionMetrics, a::AbstractMatrix,
                            b::AbstractArray, col::Int, break_at::Number, do_end::Bool=true)
    s = eval_start(d, a, b, col)
    for I in eachindex(b)
        @inbounds ai = a[I, col]
        @inbounds bi = b[I]
        s = eval_reduce(d, s, eval_op(d, ai, bi))
        if s > break_at
            break
        end
    end
    if do_end
        return eval_end(d, s)
    else
        return s
    end
end

function Distances.evaluate(d::Distances.PreMetric, a::AbstractMatrix,
                            b::AbstractArray, col::Int, do_end::Bool=true)
    evaluate(d, slice(a, :, col), b)
end

function Distances.evaluate(d::Distances.PreMetric, a::AbstractMatrix,
                            b::AbstractArray, col::Int, break_at::Number, do_end::Bool=true)
    evaluate(d, slice(a, :, col), b)
end

@inline eval_pow(::MinkowskiMetric, s) = abs(s)
@inline eval_pow(::Euclidean, s) = abs2(s)
@inline eval_pow(d::Minkowski, s) = abs(s)^d.p

@inline eval_diff(::MinkowskiMetric, a, b) = a - b
@inline eval_diff(::Chebyshev, a, b) = b
