module TestMisc
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "TestSetup.jl"))
using ..Main.TestSetup
using NearestNeighbors
using NearestNeighbors: HyperRectangle, get_min_distance_no_end, get_max_distance_no_end
using StaticArrays
using Test
using Distances: Chebyshev, Cityblock, Minkowski, Euclidean, PeriodicEuclidean

@testset "views of SVector" begin
    x = [rand(SVector{3}) for i in 1:20]
    for T in (KDTree, BruteTree, BallTree)
        for reorder in (true, false)
            S = T(x; reorder)
            @test S isa T
        end
    end
end

@testset "periodic euclidean" begin
    pred = PeriodicEuclidean([Inf, 2.5])
    l = [0.0 0.0; 0.0 2.5]
    S = BallTree(l, pred)
    @test inrange(S,[0.0,0.0], 1e-2, true) == [1, 2]
end

@testset "hyperrectangle" begin
    ms = (Chebyshev(), Cityblock(), Minkowski(3.5), Euclidean())
    hr = HyperRectangle([-1.0, -2.0], [1.0, 2.0])

    # Point inside
    point = [-0.5, 0.3]
    closest_point = [-0.5, 0.3]
    furthest_point = [1.0, -2.0]
    for m in ms
        @test get_min_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(closest_point, point))
        @test get_max_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(furthest_point, point))
    end

    # Point outside both axis
    point = [1.5, 2.3]
    closest_point = [1.0, 2.0]
    furthest_point = [-1.0, -2.0]
    for m in ms
        @test get_min_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(closest_point, point))
        @test get_max_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(furthest_point, point))
    end

    # Point outside one axis
    point = [0.5, 2.3]
    closest_point = [0.5, 2.0]
    furthest_point = [-1.0, -2.0]
    for m in ms
        @test get_min_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(closest_point, point))
        @test get_max_distance_no_end(m, hr, point) ≈ NearestNeighbors.eval_pow(m, m(furthest_point, point))
    end

    for m in ms
        hyper_rec = NearestNeighbors.HyperRectangle{SVector{1, Float32}}(Float32[0.5553872], Float32[0.6169486])
        point = [0.5]
        min_dist = NearestNeighbors.get_min_distance_no_end(m, hyper_rec, point)
        split_dim = 1
        split_val = 0.5844354f0
        hyper_rec_far = NearestNeighbors.HyperRectangle{SVector{1, Float32}}(Float32[0.5844354], Float32[0.6169486])
        new_min = NearestNeighbors.update_new_min(m, min_dist, hyper_rec, point[split_dim], split_dim, split_val)
        new_min_true = NearestNeighbors.get_min_distance_no_end(m, hyper_rec_far, point)
        @test new_min ≈ new_min_true
    end

    for m in ms
        hyper_rec = NearestNeighbors.HyperRectangle{SVector{2, Float64}}([0.07935189250034036, 0.682552911042077], [0.1619776648454222, 0.8046815005307764])
        point = [0.06630748183735935, 0.7541470744398973]
        min_dist = NearestNeighbors.get_min_distance_no_end(m, hyper_rec, point)
        split_dim = 2
        split_val = 0.7388396209627084
        hyper_rec_far = NearestNeighbors.HyperRectangle{SVector{2, Float64}}([0.07935189250034036, 0.682552911042077], [0.1619776648454222, 0.7388396209627084])
        new_min = NearestNeighbors.update_new_min(m, min_dist, hyper_rec, point[split_dim], split_dim, split_val)
        new_min_true = NearestNeighbors.get_min_distance_no_end(m, hyper_rec_far, point)
        @test new_min ≈ new_min_true broken = m isa Chebyshev
    end
end

end # module
