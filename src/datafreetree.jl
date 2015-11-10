type DataFreeTree{T <: AbstractFloat, M <: Metric}
    size::Tuple{Int,Int}
    tree::NNTree{T,M}
end

function injectdata{T,M}(datafreetree::DataFreeTree{T,M}, data::Matrix{T})
    if size(data) != datafreetree.size
        error("NearestNeighbors:injectdata: The size of 'data' $(data) does not match the data array used to construct the tree $(datafreetree.size).")
    end
    typ = typeof(datafreetree.tree)
    fields = map(x->datafreetree.tree.(x), fieldnames(datafreetree.tree))[2:end]
    typ(data, fields...)
end

