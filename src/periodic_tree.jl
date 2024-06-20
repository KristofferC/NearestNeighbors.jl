struct PeriodicTree{V<:AbstractVector, M, Tree <: NNTree{V, M}, D} <: NNTree{V,M}
    tree::Tree
    bbox::HyperRectangle{V}
    combos::Vector{SVector{D, Int}}

    function PeriodicTree(tree::NNTree{V,M}, bounds_min, bounds_max) where {V,M}
        dim = length(V)
        if length(bounds_min) != dim || length(bounds_max) != dim
            throw(ArgumentError("Bounding box dimensions do not match data dimensions"))
        end

        combos = SVector(ntuple(i -> -1:1, Val(dim)))
        box_widths = SVector(ntuple(i -> bounds_max[i] - bounds_min[i], Val(dim)))

        for i in 1:dim
            if box_widths[i] <= 0 || isinf(box_widths[i])
                combos = setindex(combos, 0:0, i)
            end
        end
        combos = SVector{dim, Int}.(collect(Iterators.product(combos...)))

        # Put the (0, 0, 0, ...) combo first in the list of combos
        filtered_product = filter(x -> x != zero(SVector{dim, Int}), combos)
        combos_reordered = pushfirst!(filtered_product, zero(SVector{dim, Int}))
        return new{V, M, typeof(tree), dim}(tree, HyperRectangle(SVector{dim}(bounds_min), SVector{dim}(bounds_max)), combos_reordered)
    end
end

get_tree(tree::PeriodicTree) = tree.tree

function Base.show(io::IO, tree::PeriodicTree{V}) where {V}
    println(io, "Periodic Tree: $(typeof(tree.tree))")
    println(io, "  Bounding box: ", tree.bbox.mins, " ", tree.bbox.maxes)
    println(io, "  Number of points: ", length(tree.tree.data))
    println(io, "  Dimensions: ", length(V))
    println(io, "  Metric: ", tree.tree.metric)
    print(io,   "  Reordered: ", tree.tree.reordered)
end

function _knn(tree::PeriodicTree{V,M},
    point::AbstractVector,
    best_idxs::AbstractVector{<:Integer},
    best_dists::AbstractVector,
    skip::F) where {V, M, F}
    dim = length(V)
    box_widths = SVector(ntuple(i -> tree.bbox.maxes[i] - tree.bbox.mins[i], Val(dim)))
    for combo in tree.combos
        shift_vector = box_widths .* combo
        point_shifted = point + shift_vector

        min_dist_to_canonical = get_min_distance_no_end(tree.tree.metric, tree.bbox, point_shifted)

        # TODO: Only search the mirror boxes that are relevant

        if tree.tree isa KDTree
            knn_kernel!(tree.tree, 1, point_shifted, best_idxs, best_dists, min_dist_to_canonical, tree.tree.hyper_rec, skip, true)
        elseif tree.tree isa BallTree
            knn_kernel!(tree.tree, 1, point_shifted, best_idxs, best_dists, skip, true)
        else
            @assert tree.tree isa BruteTree
            knn_kernel!(tree.tree, point_shifted, best_idxs, best_dists, skip, true)
        end
    end

    if tree.tree isa KDTree
        @simd for i in eachindex(best_dists)
            @inbounds best_dists[i] = eval_end(tree.tree.metric, best_dists[i])
        end
    end


    @assert allunique(best_idxs)
    return
end

function _inrange(tree::PeriodicTree{V},
    point::AbstractVector,
    radius::Number,
    idx_in_ball::Union{Nothing, Vector{<:Integer}},
    skip::F) where {V, F}

    dim = length(V)

    box_widths = SVector(ntuple(i -> tree.bbox.maxes[i] - tree.bbox.mins[i], Val(dim)))

    for combo in tree.combos
        shift_vector = box_widths .* combo
        point_shifted = point + shift_vector

        # TODO: Only search the mirror boxes that are relevant

        if tree.tree isa KDTree
            min_dist_to_canonical = get_min_distance_no_end(tree.tree.metric, tree.bbox, point_shifted)
            inrange_kernel!(tree.tree, 1, point_shifted, eval_op(tree.tree.metric, radius, zero(min_dist_to_canonical)), idx_in_ball,
            tree.tree.hyper_rec, min_dist_to_canonical, skip, true)
        elseif tree.tree isa BallTree
            ball = HyperSphere(convert(V, point_shifted), convert(eltype(V), radius))
            inrange_kernel!(tree.tree, 1, point_shifted, ball, idx_in_ball, skip, true)
         else
            @assert tree.tree isa BruteTree
            inrange_kernel!(tree.tree, point, radius, idx_in_ball, skip, true)
        end
    end

    @assert allunique(idx_in_ball)
    return
end
