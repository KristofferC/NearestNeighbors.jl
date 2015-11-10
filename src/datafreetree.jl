immutable DataFreeTree{T <: AbstractFloat, M <: Metric}
    size::Tuple{Int,Int}
    hash::UInt64
    tree::NNTree{T,M}
end
DataFreeTree(data, tree) = DataFreeTree(size(data), hash(data), tree)

function injectdata{T,M}(datafreetree::DataFreeTree{T,M}, data::Matrix{T})
    if size(data) != datafreetree.size
        error("NearestNeighbors:injectdata: The size of 'data' $(data) does not match the data array used to construct the tree $(datafreetree.size).")
    end

    if hash(data) != datafreetree.hash
        error("NearestNeighbors:injectdata: The hash of 'data' does not match the hash of the data array used to construct the tree.")
    end

    typ = typeof(datafreetree.tree)
    fields = map(x->datafreetree.tree.(x), fieldnames(datafreetree.tree))[2:end]
    typ(data, fields...)
end

