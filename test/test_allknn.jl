module TestAllKNN
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "TestSetup.jl"))

using NearestNeighbors
using ..Main.TestSetup: trees_with_brute
using StableRNGs
using StaticArrays
using Test

@testset "allknn" begin
    rng = StableRNG(42)
    data = rand(rng, 3, 40)
    k = 3
    bounds_min = fill(0.0, 3)
    bounds_max = fill(1.0, 3)

    for reorder in (false, true)
        for Tree in trees_with_brute
            # BruteTree doesn't support reorder (no indices field)
            tree = Tree === BruteTree ? Tree(data) : Tree(data; leafsize=8, reorder)
            for treelike in (tree, PeriodicTree(tree, bounds_min, bounds_max))
                idxs, dists = allknn(treelike, k, true)
                @test length(idxs) == size(data, 2)
                @test length(dists) == size(data, 2)
                @test all(length(v) == k for v in idxs)
                @test all(length(v) == k for v in dists)

                # Matches per-point knn with an explicit self-skip (original indices)
                for i in 1:size(data, 2)
                    point = SVector{3,Float64}(data[:, i])
                    skip_self = x -> x == i
                    exp_idxs, exp_dists = knn(treelike, point, k, true, skip_self)
                    @test idxs[i] == exp_idxs
                    @test dists[i] == exp_dists
                    @test !(i in idxs[i])
                end
            end
        end
    end
end

@testset "allnn" begin
    rng = StableRNG(123)
    data = rand(rng, 3, 30)
    bounds_min = fill(0.0, 3)
    bounds_max = fill(1.0, 3)

    for reorder in (false, true)
        for Tree in trees_with_brute
            # BruteTree doesn't support reorder (no indices field)
            tree = Tree === BruteTree ? Tree(data) : Tree(data; leafsize=8, reorder)
            for treelike in (tree, PeriodicTree(tree, bounds_min, bounds_max))
                idxs, dists = allnn(treelike)
                @test length(idxs) == size(data, 2)
                @test length(dists) == size(data, 2)

                # Matches per-point nn with an explicit self-skip
                for i in 1:size(data, 2)
                    point = SVector{3,Float64}(data[:, i])
                    skip_self = x -> x == i
                    exp_idx, exp_dist = nn(treelike, point, skip_self)
                    @test idxs[i] == exp_idx
                    @test dists[i] == exp_dist
                    @test idxs[i] != i
                end
            end
        end
    end
end

end # module
