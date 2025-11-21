using NearestNeighbors
using Distances: evaluate
using StableRNGs
using StaticArrays
using Test

@testset "tree-tree inrange" begin
    rng = StableRNG(12)
    data = rand(rng, 3, 30)
    tree = KDTree(data; leafsize=6)
    btree = BallTree(data; leafsize=6)
    brutree = BruteTree(data)
    radius = 0.25

    expected_self_raw = inrange(tree, data, radius, true)
    expected_self = [filter(!=(i), expected_self_raw[i]) for i in 1:size(data, 2)]
    npoints = size(data, 2)
    function adjacency_from_pairs(pairs, n)
        res = [Int[] for _ in 1:n]
        for (i, j) in pairs
            push!(res[i], j); push!(res[j], i)
        end
        for v in res
            sort!(v)
        end
        res
    end

    actual_self = adjacency_from_pairs(inrange_pairs(tree, radius, true), npoints)
    @test actual_self == expected_self

    skip_idx = 1
    skip_first = x -> x == skip_idx
    points_vec = [SVector{3,Float64}(data[:, i]) for i in 1:size(data, 2)]
    expected_skip_raw = inrange(tree, points_vec, radius, true, skip_first)
    expected_skip = [i == skip_idx ? Int[] : filter(x -> x != skip_idx && x != i, expected_skip_raw[i]) for i in 1:npoints]
    actual_skip = adjacency_from_pairs(inrange_pairs(tree, radius, true, skip_first), npoints)
    @test actual_skip == expected_skip

    # Self pairs: do not double count mirrored neighbors or include self
    radius_self = 0.6
    full = inrange(tree, data, radius_self, true)
    selfpairs = inrange_pairs(tree, radius_self, true)

    @test all(p -> p[1] < p[2], selfpairs)
    adj_pairs = Set{Tuple{Int,Int}}()
    for i in eachindex(full)
        for j in full[i]
            i == j && continue
            push!(adj_pairs, i < j ? (i, j) : (j, i))
        end
    end
    @test Set(selfpairs) == adj_pairs

    # BallTree parity
    b_full_raw = inrange(btree, data, radius_self, true)
    b_full = [filter(!=(i), b_full_raw[i]) for i in 1:npoints]
    b_pairs = inrange_pairs(btree, radius_self, true)
    b_adj_pairs = adjacency_from_pairs(b_pairs, npoints)
    @test b_adj_pairs == b_full

    # BruteTree parity
    br_pairs = inrange_pairs(brutree, radius_self, true)
    @test Set(br_pairs) == Set(selfpairs)

    # All trees produce identical results
    @test Set(selfpairs) == Set(b_pairs) == Set(br_pairs)
end
