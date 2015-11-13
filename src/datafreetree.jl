immutable DataFreeTree{T <: AbstractFloat, M <: Metric}
    size::Tuple{Int,Int}
    hash::UInt
    tree::NNTree{T,M}
end

function DataFreeTree{T<:NNTree}(::Type{T}, data, args...; reorderbuffer = data[:,1:0], kargs...)
    tree = T(data, args...; storedata = false, reorderbuffer = reorderbuffer, kargs...)
    DataFreeTree(size(data), hash(tree.reordered ? reorderbuffer : data), tree)
end

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
