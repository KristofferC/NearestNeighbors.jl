# A DataFreeTree wraps a descendant of NNTree
# which does not contain a copy of the data
immutable DataFreeTree{N <: NNTree}
    hash::UInt64
    tree::N
end

"""
    DataFreeTree(treetype, data[, reorderbufffer = similar(data), indicesfor = :data, kargs...]) -> datafreetree

Creates a `DataFreeTree` which wraps a `KDTree` or `BallTree`. Keywords arguments are passed
to their respective constructors.

The `KDTree` or `BallTree` will be stored without a reference to the underlaying data. `injectdata`
has to be used to re-link them to a data array before use.

By default the `reorder` feature of `KDTree`/`BallTree` is turned off. In case a `reorderbuffer`
is provided, reordering is performed and the contents of `reorderbuffer` have to be later provided to
`injectdata`.

`indicesfor` controlls whether the indices returned by the query functions should refer to `data` or the `reorderbuffer`. Valid values are `:data` and `:reordered`.
"""
function DataFreeTree{T<:NNTree}(::Type{T}, data, args...; reorderbuffer = data[:, 1:0], kargs...)
    tree = T(data, args...; storedata = false, reorderbuffer = reorderbuffer, kargs...)

    DataFreeTree(hash(tree.reordered ? reorderbuffer : data), tree)
end

"""
    injectdata(datafreetree, data) -> tree

Returns the `KDTree`/`BallTree` wrapped by `datafreetree`, set up to use `data` for the points data.
"""
function injectdata{T}(datafreetree::DataFreeTree, data::Matrix{T})
    dim = size(data, 1)
    npoints = size(data, 2)
     if isbits(T)
        new_data = reinterpret(SVector{dim, T}, data, (npoints, ))
    else
        new_data = SVector{dim, T}[SVector{dim, T}(data[:, i]) for i in 1:npoints]
    end
    injectdata(datafreetree, new_data)
end

function injectdata(datafreetree::DataFreeTree, data::Vector)
    if hash(data) != datafreetree.hash
        throw(ArgumentError("NearestNeighbors:injectdata: The hash of 'data' does not match the hash of the data array used to construct the tree."))
    end


    typ = typeof(datafreetree.tree)
    fields = map(x-> getfield(datafreetree.tree, x), fieldnames(datafreetree.tree))[2:end]
    typ(data, fields...)
end

