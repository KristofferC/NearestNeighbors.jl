# Find the dimension witht the largest spread.
function find_largest_spread{T}(data::Matrix{T}, indices, low, high)
    n_points = high - low + 1
    n_dim = size(data, 1)
    split_dim = 1
    max_spread = zero(T)
    for dim in 1:n_dim
        xmin = typemax(T)
        xmax = typemin(T)
        # Find max and min in this dim
        for coordinate in 1:n_points
            xmin = min(xmin, data[dim, indices[coordinate + low - 1]])
            xmax = max(xmax, data[dim, indices[coordinate + low - 1]])
        end

        if xmax - xmin > max_spread # Found new max_spread, update split dimension
            max_spread = xmax - xmin
            split_dim = dim
        end
    end
    return split_dim
end

# Taken from https://github.com/JuliaLang/julia/blob/v0.3.5/base/sort.jl
# and modified to compare against a matrix
function select_spec!{T <: AbstractFloat}(v::AbstractVector, k::Int, lo::Int,
                                          hi::Int, data::Matrix{T}, dim::Int)
    @inbounds lo <= k <= hi || error("select index $k is out of range $lo:$hi")
     while lo < hi
        if hi-lo == 1
            if data[dim, v[hi]] < data[dim, v[lo]]
                v[lo], v[hi] = v[hi], v[lo]
            end
            return
        end
        pivot = v[(lo+hi)>>>1]
        i, j = lo, hi
        while true
            while data[dim, v[i]] < data[dim, pivot]; i += 1; end
            while  data[dim, pivot] <  data[dim, v[j]] ; j -= 1; end
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
function heap_sort_inplace!(xs, xis)
    @inbounds for i in length(xs):-1:2
        xs[i], xs[1] = xs[1], xs[i]
        xis[i], xis[1] = xis[1], xis[i]
        percolate_down!(xs, xis, xs[1], xis[1], i-1)
    end
    return
end

# Binary max-heap percolate down.
function percolate_down!(xs::AbstractArray,
                         xis::AbstractArray,
                         dist::Number,
                         index::Int,
                         len::Int=length(xs))
    i = 1
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
