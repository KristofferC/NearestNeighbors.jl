typealias MinkowskiMetric Union{Euclidean, Chebyshev, Cityblock, Minkowski}

function Distances.evaluate(d::Distances.UnionMetrics, a::AbstractMatrix,
                            b::AbstractArray, col::Int, do_end::Bool=true)
    s = eval_start(d, a, b)
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

# As above but stops evaluating after break_at is hit
# TODO: Use this in the inrange methods
function Distances.evaluate(d::Distances.UnionMetrics, a::AbstractMatrix,
                            b::AbstractArray, col::Int, break_at::Number, do_end::Bool=true)
    s = eval_start(d, a, b)
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

inv_eval_end(::MinkowskiMetric, s) = s
inv_eval_end(::Euclidean, s) = abs2(s)
inv_eval_end(d::Minkowski, s) = s^d.p
