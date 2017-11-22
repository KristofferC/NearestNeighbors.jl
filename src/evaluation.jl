@inline eval_pow(::MinkowskiMetric, s) = abs(s)
@inline eval_pow(::Euclidean, s) = abs2(s)
@inline eval_pow(d::Minkowski, s) = abs(s)^d.p

@inline eval_diff(::MinkowskiMetric, a, b) = a - b
@inline eval_diff(::Chebyshev, ::Any, b) = b

function Distances.evaluate(d::Distances.UnionMetrics, a::AbstractVector,
                            b::AbstractVector, do_end::Bool)
    s = eval_start(d, a, b)
    @simd for i in eachindex(b)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        s = eval_reduce(d, s, eval_op(d, ai, bi))
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

function Distances.evaluate{Ti,Tv}(d::Distances.UnionMetrics, a::SparseVector{Ti,Tv},
                            b::SparseVector{Ti,Tv}, do_end::Bool)
    # we ignore all entires where both ai and bi are zero
    const zeroval = zero(eltype(a))

    s = eval_start(d, a, b)

    # get nonzero entires
    nzixa, nzvala = findnz(a)
    nzixb, nzvalb = findnz(b)

    # total nonzero entries
    Na = length(nzixa)
    Nb = length(nzixb)

    # counters for nonzero entries
    ia = 1
    ib = 1

    while (ia <= Na) || (ib <= Nb)
        if ia > Na
            # ran out of a entries, just process b
            ai = zeroval
            @inbounds bi = nzvalb[ib]
            ib += 1
        elseif ib > Nb
            # ran out of b entries, just process a
            @inbounds ai = nzvala[ia]
            bi = zeroval
            ia += 1
        else
            # get actual indices
            @inbounds ixa = nzixa[ia]
            @inbounds ixb = nzixb[ib]

            if ixa < ixb
                # only eval ai
                @inbounds ai = nzvala[ia]
                bi = zeroval
                ia += 1
            elseif ixa > ixb
                # only eval bi
                ai = zeroval
                @inbounds bi = nzvalb[ib]
                ib += 1
            else
                # eval both
                @inbounds ai = nzvala[ia]
                @inbounds bi = nzvalb[ib]
                ia += 1
                ib += 1
            end
        end
        s = eval_reduce(d, s, eval_op(d, ai, bi))
    end

    if do_end
        return eval_end(d, s)
    else
        return s
    end
end

# this version is used by brute tree
Distances.evaluate{Ti,Tv}(d::Distances.UnionMetrics, a::SparseVector{Ti,Tv}, b::SparseVector{Ti,Tv}) = Distances.evaluate(d, a, b, true)
