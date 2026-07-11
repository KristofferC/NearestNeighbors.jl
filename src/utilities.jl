# Find the dimension with the largest spread.
function find_largest_spread(data::AbstractVector{V}, indices, range) where {V}
    T = eltype(V)
    n_dim = length(V)
    mins = MVector{n_dim, T}(undef)
    maxes = MVector{n_dim, T}(undef)
    fill!(mins, typemax(T))
    fill!(maxes, typemin(T))
    @inbounds for i in range
        p = data[indices[i]]
        for dim in 1:n_dim
            v = p[dim]
            mins[dim] = min(mins[dim], v)
            maxes[dim] = max(maxes[dim], v)
        end
    end
    split_dim = 1
    max_spread = zero(T)
    @inbounds for dim in 1:n_dim
        spread = maxes[dim] - mins[dim]
        if spread > max_spread # Found new max_spread, update split dimension
            max_spread = spread
            split_dim = dim
        end
    end
    return split_dim
end

# Taken from https://github.com/JuliaLang/julia/blob/v0.3.5/base/sort.jl
# and modified to compare against a matrix
@inline function select_spec!(v::Vector{Int}, k::Int, lo::Int,
                              hi::Int, data::AbstractVector, dim::Int)
    lo <= k <= hi || error("select index $k is out of range $lo:$hi")
    @inbounds while lo < hi
        if hi - lo == 1
            if data[v[hi]][dim] < data[v[lo]][dim]
                v[lo], v[hi] = v[hi], v[lo]
            end
            return
        end
        pivot = v[(lo + hi) >>> 1]
        i, j = lo, hi
        while true
            while data[v[i]][dim] < data[pivot][dim]; i += 1; end
            while data[pivot][dim] < data[v[j]][dim]; j -= 1; end
            i <= j || break
            v[i], v[j] = v[j], v[i]
            i += 1; j -= 1
        end
        if k <= j
            hi = j
        elseif i <= k
            lo = i
        else
            return
        end
    end
    return
end

# In place heap sort
@inline function heap_sort_inplace!(xs, xis)
    @inbounds for i in length(xs):-1:2
        xs[i], xs[1] = xs[1], xs[i]
        xis[i], xis[1] = xis[1], xis[i]
        percolate_down!(xs, xis, xs[1], xis[1], 1, i - 1)
    end
    return
end

# Binary max-heap percolate down.
@inline function percolate_down!(xs::AbstractArray,
                         xis::AbstractArray,
                         dist::Number,
                         index::Integer,
                         offset::Integer=1,
                         len::Integer=length(xs))
    i = offset
    @inbounds while (l = getleft(i)) <= len
        r = getright(i)
        j = ifelse(r > len || (xs[l] > xs[r]), l, r)
        if xs[j] > dist
            xs[i] = xs[j]
            xis[i] = xis[j]
            i = j
        else
            break
        end
    end
    xs[i] = dist
    xis[i] = index
    return
end

# Instead of ReinterpretArray wrapper, copy an array, interpreting it as a vector of SVectors
copy_svec(::Type{T}, data, ::Val{dim}) where {T, dim} =
        [SVector{dim,T}(ntuple(i -> data[n+i], Val(dim))) for n in 0:dim:(length(data)-1)]::Vector{SVector{dim,T}}

# Check that a metric with per-dimension parameters (e.g. weights) matches the
# data dimension
function check_metric_dimension(metric, ::Type{V}) where {V}
    if metric isa Distances.UnionMetrics
        p = parameters(metric)
        if p !== nothing && length(p) != length(V)
            throw(ArgumentError(
                "dimension of input points:$(length(V)) and metric parameter:$(length(p)) must agree"))
        end
    end
    return
end

# Take the data vector of a tree being consumed by a mutating constructor
# (KDTree!/BallTree!) for use as the reorder buffer. It is only reusable when
# it is an internal copy (i.e. the old tree was reordered), and it must not
# alias the new input data.
function harvest_reorderbuffer(old_tree, data::AbstractVector{V}, reorder::Bool) where {V}
    reorderbuffer = old_tree.reordered ? old_tree.data : Vector{V}()
    if reorder && data === reorderbuffer
        throw(ArgumentError(
            "`data` aliases the storage of `old_tree`; pass an independent array"))
    end
    return reorderbuffer
end

# Check for NaN values in data; throw if any are present
function check_for_nan(data)
    @inbounds for p in data
        if any(isnan, p)
            throw(ArgumentError("Tree cannot be constructed from data containing NaN values"))
        end
    end
    return
end

# Check for NaN values in input points; throw if any are present
function check_for_nan_in_points(points::Union{AbstractVector, AbstractMatrix})
    if any(isnan, points)
        throw(ArgumentError("Tree cannot be queried with points containing NaN values"))
    end
    return
end

function check_for_nan_in_points(points::AbstractVector{<:AbstractVector})
    for p in points
        check_for_nan_in_points(p)
    end
    return
end
