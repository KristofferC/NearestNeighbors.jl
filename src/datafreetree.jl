# A DataFreeTree wraps a descendant of NNTree
# which does not contain a copy of the data
struct DataFreeTree{N <: NNTree}
    size::Tuple{Int,Int}
    hash::UInt64
    tree::N
end

function get_points_dim(data)
    if eltype(data) <: AbstractVector
        ndim = eltype(eltype(data))
        npoints = length(data)
    elseif typeof(data) <: AbstractMatrix
        ndim = size(data, 1)
        npoints = size(data, 2)
    else
        error("Unknown input data format")
    end
    return ndim, npoints
end

"""
    DataFreeTree(treetype, data[, reorderbufffer = similar(data), kwargs...]) -> datafreetree

Creates a `DataFreeTree` which wraps a `KDTree` or `BallTree`. Keywords arguments are passed
to their respective constructors.

The `KDTree` or `BallTree` will be stored without a reference to the underlaying data. `injectdata`
has to be used to re-link them to a data array before use.
"""
function DataFreeTree(::Type{T}, data, args...; reorderbuffer = data[:, 1:0], kargs...) where {T <: NNTree}
    tree = T(data, args...; storedata = false, reorderbuffer = reorderbuffer, kargs...)
    ndim, npoints = get_points_dim(data)
    DataFreeTree((ndim, npoints), hash(tree.reordered ? reorderbuffer : data), tree)
end

"""
    injectdata(datafreetree, data) -> tree

Returns the `KDTree`/`BallTree` wrapped by `datafreetree`, set up to use `data` for the points data.
"""
function injectdata(datafreetree::DataFreeTree, data::AbstractMatrix{T}) where {T}
    dim = size(data, 1)
    npoints = size(data, 2)
    if isbitstype(T)
        new_data = copy_svec(T, data, Val(dim))
    else
        new_data = SVector{dim,T}[SVector{dim,T}(data[:, i]) for i in 1:npoints]
    end
    new_hash = hash(data)
    injectdata(datafreetree, new_data, new_hash)
end

function injectdata(datafreetree::DataFreeTree, data::AbstractVector{V}, new_hash::UInt64=0) where {V <: AbstractVector}
    if new_hash == 0
        new_hash = hash(data)
    end
    if length(V) != datafreetree.size[1] || length(data) != datafreetree.size[2]
        throw(DimensionMismatch("NearestNeighbors:injectdata: The size of 'data' $(length(data)) × $(length(V)) does not match the data array used to construct the tree $(datafreetree.size)."))
    end
    if new_hash != datafreetree.hash
        throw(ArgumentError("NearestNeighbors:injectdata: The hash of 'data' does not match the hash of the data array used to construct the tree."))
    end

    typ = typeof(datafreetree.tree)
    fields = map(x -> getfield(datafreetree.tree, x), fieldnames(typeof(datafreetree.tree)))[2:end]
    typ(data, fields...)
end
