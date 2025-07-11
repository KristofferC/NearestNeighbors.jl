check_radius(r) = r < 0 && throw(ArgumentError("the query radius r must be ≧ 0"))

"""
    inrange(tree::NNTree, points, radius [, sortres=false]) -> indices

Find all the points in the tree which is closer than `radius` to `points`. If
`sortres = true` the resulting indices are sorted.

See also: `inrange!`, `inrangecount`.
"""
inrange(tree::NNTree{V}, points, radius::Number, sortres=false) where {V} = inrange_callback_default(tree, points, radius, sortres)

"""
    inrange!(idxs, tree, point, radius)

Same functionality as `inrange` but stores the results in the input vector `idxs`.
Useful if one want to avoid allocations or specify the element type of the output vector.

See also: `inrange`, `inrangecount`.
"""
function inrange!(idxs::AbstractVector, tree::NNTree{V}, point::AbstractVector{T}, radius::Number, sortres=false) where {V, T <: Number}
    check_input(tree, point)
    check_radius(radius)
    length(idxs) == 0 || throw(ArgumentError("idxs must be empty"))

    f(a, b) = index_returning_runtime_function(a, b, idxs)
    inrange_callback!(tree, point, radius, f)

    sortres && sort!(idxs)
    return idxs
end

"""
    inrangecount(tree::NNTree, points, radius) -> count

Count all the points in the tree which are closer than `radius` to `points`.
"""
function inrangecount(tree::NNTree{V}, point::AbstractVector{T}, radius::Number) where {V, T <: Number}
    check_input(tree, point)
    check_radius(radius)
    return _inrange(tree, point, radius)
end

function inrangecount(tree::NNTree,
        points::AbstractVector{T},
        radius::Number) where {T <: AbstractVector}
    check_input(tree, points)
    check_radius(radius)
    return _inrange.(Ref(tree), points, radius)
end

function inrangecount(tree::NNTree{V}, point::AbstractMatrix{T}, radius::Number) where {V, T <: Number}
    dim = size(point, 1)
    npoints = size(point, 2)
    if isbitstype(T)
        new_data = copy_svec(T, point, Val(dim))
    else
        new_data = SVector{dim,T}[SVector{dim,T}(point[:, i]) for i in 1:npoints]
    end
    return inrangecount(tree, new_data, radius)
end

"""
    inrange_callback!(tree::NNTree{V}, points::AbstractVector{T}, radius::Number, callback::Function)

Compute a runtime function for all in range queries.
Instead of returning the indicies, the `callback` is called for each point in points
and is given the points, the index of the point, and the index of the neighbor.
The `callback` should return nothing.
The `callback` should be of the form:
callback(point_index::Int, neighbor_index::Int)
where `point_index` is the index of the point in `points`, `neighbor_index` is the index of the neighbor in the tree.

For example:
```julia
function callback(point_index, neighbor_index, random_storage_of_results, neightbors_data)
    # do something with the points
    return nothing
end

random_storage_of_results = rand(3, 100)
neightbors_data = rand(3, 100)
f(a, b) = callback(a, b, random_storage_of_results, neightbors_data)
```
"""
function inrange_callback!(tree::NNTree{V}, points::AbstractVector{T}, radius::Number, callback::F) where {V, T <: AbstractVector, F}
    check_input(tree, points)
    check_radius(radius)

    for i in eachindex(points)
        _inrange(tree, points[i], radius, i, callback)
    end
    return nothing
end

function inrange_callback!(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, callback::F) where {V, T <: Number, F}
    return inrange_callback!(tree, points, radius, callback, Val(size(points, 1)))
end

function inrange_callback!(tree::NNTree{V}, points::AbstractVector{T}, radius::Number, callback::F) where {V, T <: Number, F}
    points = reshape(points, size(points, 1), 1)
    return inrange_callback!(tree, points, radius, callback, Val(size(points, 1)))
end

function inrange_callback!(tree::NNTree{V}, points::AbstractMatrix{T}, radius::Number, callback::F, ::Val{dim}) where {V, T <: Number, F, dim}
    check_input(tree, points)
    check_radius(radius)
    n_points = size(points, 2)
    for i in 1:n_points
        point = SVector{dim,T}(ntuple(j -> points[j, i], Val(dim)))
        _inrange(tree, point, radius, i, callback)
    end
    return nothing
end

function index_returning_runtime_function(point_index::Int, neighbor_index::Int, idxs)
    if eltype(idxs) <: Integer
        push!(idxs, eltype(idxs)(neighbor_index))
    else
        push!(idxs[point_index], neighbor_index)
    end
    return nothing
end

function inrange_callback_default(tree::NNTree{V}, points, radius::Number, sortres=false) where {V}    
    if points isa AbstractVector{<:AbstractVector}
        n_points = length(points)
    elseif points isa AbstractMatrix{<:Number}
        n_points = size(points, 2)
    elseif points isa AbstractVector{<:Number}
        n_points = 1
    end
    
    idxs = [Int[] for _ in 1:n_points]
    f(a, b) = index_returning_runtime_function(a, b, idxs)
    inrange_callback!(tree, points, radius, f)

    if sortres
        for i in eachindex(idxs)
            sort!(idxs[i])
        end
    end

    if length(idxs) == 1
        idxs = idxs[1]  # If only one point, return a single vector instead of a vector of vectors
    end
    return idxs
end

