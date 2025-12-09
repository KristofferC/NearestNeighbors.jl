module TestUnitful
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "TestSetup.jl"))
using ..Main.TestSetup
using NearestNeighbors
using StaticArrays
using Test
using Unitful

@testset "Unitful support" begin
    strip_units(x::Number) = ustrip(x)
    strip_units(x::AbstractVector) = ustrip.(x)
    strip_units(x::SVector{N}) where N = SVector{N}(ustrip.(x))

    # Data with units
    m = u"m"
    data_unitful = [SVector(1.0m, 2.0m, 3.0m), SVector(2.0m, 3.0m, 4.0m), SVector(10.0m, 10.0m, 10.0m)]
    query_unitful = [1.0m, 2.0m, 3.0m]
    bounds_min_unitful = [0.0m, 0.0m, 0.0m]
    bounds_max_unitful = [20.0m, 20.0m, 20.0m]

    # Same data without units
    data_plain = [strip_units(v) for v in data_unitful]
    query_plain = strip_units(query_unitful)
    bounds_min_plain = strip_units(bounds_min_unitful)
    bounds_max_plain = strip_units(bounds_max_unitful)

    # Build all tree pairs (plain, unitful)
    tree_pairs = Pair{String, Tuple{NNTree, NNTree}}[]
    for TreeType in [KDTree, BallTree, BruteTree]
        push!(tree_pairs, "$TreeType" => (TreeType(data_plain), TreeType(data_unitful)))
        push!(tree_pairs, "PeriodicTree($TreeType)" => (
            PeriodicTree(TreeType(data_plain), bounds_min_plain, bounds_max_plain),
            PeriodicTree(TreeType(data_unitful), bounds_min_unitful, bounds_max_unitful)
        ))
    end

    for (name, (tree_plain, tree_unitful)) in tree_pairs
        @testset "$name" begin
            @testset "nn" begin
                idx_plain, dist_plain = nn(tree_plain, query_plain)
                idx_unitful, dist_unitful = nn(tree_unitful, query_unitful)
                @test idx_plain == idx_unitful
                @test dist_plain ≈ strip_units(dist_unitful)
            end

            @testset "knn" begin
                idxs_plain, dists_plain = knn(tree_plain, query_plain, 2)
                idxs_unitful, dists_unitful = knn(tree_unitful, query_unitful, 2)
                @test idxs_plain == idxs_unitful
                @test dists_plain ≈ strip_units(dists_unitful)
            end

            @testset "knn!" begin
                idxs_plain = Vector{Int}(undef, 2)
                dists_plain = Vector{Float64}(undef, 2)
                knn!(idxs_plain, dists_plain, tree_plain, query_plain, 2)

                idxs_unitful = Vector{Int}(undef, 2)
                dists_unitful = Vector{typeof(1.0m)}(undef, 2)
                knn!(idxs_unitful, dists_unitful, tree_unitful, query_unitful, 2)

                @test idxs_plain == idxs_unitful
                @test dists_plain ≈ strip_units(dists_unitful)
            end

            @testset "inrange" begin
                idxs_plain = inrange(tree_plain, query_plain, 2.0)
                idxs_unitful = inrange(tree_unitful, query_unitful, 2.0m)
                @test sort(idxs_plain) == sort(idxs_unitful)
            end

            # inrange_pairs is only available on base trees, not PeriodicTree
            if !(tree_plain isa PeriodicTree)
                @testset "inrange_pairs" begin
                    pairs_plain = inrange_pairs(tree_plain, 2.0)
                    pairs_unitful = inrange_pairs(tree_unitful, 2.0m)
                    @test pairs_plain == pairs_unitful
                end
            end
        end
    end
end

end # module
