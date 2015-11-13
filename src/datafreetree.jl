# A DataFreeTree wraps a descendant of NNTree
# which does not contain a copy of the data
immutable DataFreeTree{T <: AbstractFloat, M <: Metric}
    size::Tuple{Int,Int}
    hash::UInt
    tree::NNTree{T,M}
end

"""
    DataFreeTree(treetype, data[, reorderbufffer = similar(data), kargs...]) -> datafreetree

Creates a `DataFreeTree` which wraps a `KDTree` or `BallTree`. Keywords arguments are passed
to their respective constructors.

The `KDTree` or `BallTree` will be stored without a reference to the underlaying data. `injectdata` 
has to be used to re-link them to a data array before use.

By default the `reorder` feature of `KDTree`/`BallTree` is turned off. In case a `reorderbuffer`
is provided, reordering is performed and the contents of `reorderbuffer` have to be later provided to 
`injectdata`.
"""
function DataFreeTree{T<:NNTree}(::Type{T}, data, args...; reorderbuffer = data[:,1:0], kargs...)
    tree = T(data, args...; storedata = false, reorderbuffer = reorderbuffer, kargs...)
    DataFreeTree(size(data), hash(tree.reordered ? reorderbuffer : data), tree)
end

"""
    injectdata(datafreetree, data) -> tree

Returns the `KDTree`/`BallTree` wrapped by `datafreetree`, set up to use `data` for the points data.
"""
function injectdata{T,M}(datafreetree::DataFreeTree{T,M}, data::Matrix{T})
    if size(data) != datafreetree.size
        throw(DimensionMismatch("NearestNeighbors:injectdata: The size of 'data' $(data) does not match the data array used to construct the tree $(datafreetree.size)."))
    end

    if hash(data) != datafreetree.hash
        throw(ArgumentError("NearestNeighbors:injectdata: The hash of 'data' does not match the hash of the data array used to construct the tree."))
    end

    typ = typeof(datafreetree.tree)
    fields = map(x->datafreetree.tree.(x), fieldnames(datafreetree.tree))[2:end]
    typ(data, fields...)
end

