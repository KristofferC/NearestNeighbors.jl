module TestMisc
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "TestSetup.jl"))
using ..Main.TestSetup
using NearestNeighbors
using NearestNeighbors: HyperRectangle, get_min_distance_no_end, get_max_distance_no_end
using StaticArrays
using Test
using Distances: Chebyshev, Cityblock, Minkowski, Euclidean, PeriodicEuclidean, WeightedMinkowski, Hamming

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

@testset "parallel construction" begin
    # parallel=true must produce a tree identical to a serial build
    # (also exercises the Threads.@spawn path even with a single thread)
    data = rand(3, 2000)
    for T in trees
        for metric in (Euclidean(), WeightedMinkowski(rand(3) .+ 0.5, 3.0))
            for reorder in (true, false)
                serial = T(data, metric; leafsize = 5, reorder, parallel = false)
                threaded = T(data, metric; leafsize = 5, reorder, parallel = true)
                @test serial.indices == threaded.indices
                q = rand(3)
                @test knn(serial, q, 5, true) == knn(threaded, q, 5, true)
                @test inrange(serial, q, 0.3, true) == inrange(threaded, q, 0.3, true)
            end
        end
    end
end

@testset "integer data" begin
    # Large coordinate range so exact distance ties are improbable
    data = [SVector{2,Int}(rand(1:10^9), rand(1:10^9)) for _ in 1:100]
    btree = BruteTree(data)
    for T in trees
        tree = T(data)
        q = SVector(5 * 10^8, 5 * 10^8)
        @test knn(tree, q, 5, true) == knn(btree, q, 5, true)
        @test inrange(tree, q, 3 * 10^8, true) == inrange(btree, q, 3 * 10^8, true)
    end

    # Integer matrix input (issues #13, #196)
    data_mat = rand(1:10^9, 2, 100)
    btree_mat = BruteTree(data_mat)
    for T in trees
        tree = T(data_mat)
        q = rand(2) .* 10^9
        idx, dist = knn(tree, q, 5, true)
        bidx, bdist = knn(btree_mat, q, 5, true)
        @test idx == bidx
        @test dist ≈ bdist
    end
    # BallTree with Hamming on integer data (issue #13); Hamming distances are
    # small integers so ties are common — compare distances and counts only
    data_h = rand(1:4, 10, 100)
    bh = BallTree(data_h, Hamming())
    bth = BruteTree(data_h, Hamming())
    @test knn(bh, data_h[:, 1], 3, true)[2] == knn(bth, data_h[:, 1], 3, true)[2]
    @test inrangecount(bh, data_h[:, 1], 3) == inrangecount(bth, data_h[:, 1], 3)
end

@testset "vector of Vector queries" begin
    # Queries as a vector of plain (non-static) vectors (issue #85)
    data = rand(3, 100)
    for T in trees_with_brute
        tree = T(data)
        points = [rand(3) for _ in 1:5]
        idxs, dists = knn(tree, points, 2, true)
        idxs_s, dists_s = knn(tree, SVector{3}.(points), 2, true)
        @test idxs == idxs_s
        # SIMD accumulation order differs between Vector and SVector points,
        # so distances can differ by an ulp
        @test all(dists .≈ dists_s)
        @test inrange(tree, points, 0.3, true) == inrange(tree, SVector{3}.(points), 0.3, true)
        # Dimension mismatch is caught per point
        @test_throws ArgumentError knn(tree, [rand(3), rand(2)], 2)
    end
end

@testset "tree data with non-static point type" begin
    # BruteTree can store plain `Vector` points whose length is not encoded in
    # the type; the input checks must read the dimension from the data (issue #249)
    data = [rand(3) for _ in 1:20]
    tree = BruteTree(data)
    stree = BruteTree(SVector{3}.(data))
    q = rand(3)
    Q = rand(3, 5)

    @test nn(tree, data[5]) == (5, 0.0)
    idxs, dists = knn(tree, q, 3, true)
    idxs_s, dists_s = knn(stree, q, 3, true)
    @test idxs == idxs_s
    @test dists ≈ dists_s
    @test knn(tree, Q, 2, true)[1] == knn(stree, Q, 2, true)[1]
    @test inrange(tree, q, 0.5, true) == inrange(stree, q, 0.5, true)
    @test inrange(tree, Q, 0.5, true) == inrange(stree, Q, 0.5, true)
    @test inrangecount(tree, q, 0.5) == inrangecount(stree, q, 0.5)
    @test allnn(tree)[1] == allnn(stree)[1]
    @test allknn(tree, 3)[1] == allknn(stree, 3)[1]
    @test occursin("Dimensions: 3", sprint(show, tree))

    @test_throws ArgumentError nn(tree, rand(4))
    @test_throws ArgumentError knn(tree, rand(4, 2), 2)
    @test_throws ArgumentError inrange(tree, rand(2), 0.5)

    # With no stored data the dimension is unknown and the check is skipped
    empty_tree = BruteTree(data; storedata=false)
    @test occursin("Dimensions: unknown", sprint(show, empty_tree))
end

@testset "knn! output eltype" begin
    data = rand(2, 20)
    for T in trees_with_brute
        tree = T(data)
        q = rand(2)
        idxs64, dists64 = knn(tree, q, 3, true)
        idxs32 = zeros(Int32, 3)
        dists32 = zeros(Float32, 3)
        knn!(idxs32, dists32, tree, q, 3, true)
        @test idxs32 == idxs64
        @test dists32 ≈ Float32.(dists64)
    end
end

@testset "rebuilding constructors ($T!)" for (T, T!) in ((KDTree, KDTree!), (BallTree, BallTree!))
    v1 = [rand(SVector{3,Float64}) for _ in 1:500]
    v2 = [rand(SVector{3,Float64}) for _ in 1:300]
    q = rand(3)

    t1 = T(v1; leafsize = 10)
    buf = t1.data
    ibuf = t1.indices
    t2 = T!(t1, v2)
    ref = T(v2; leafsize = 10) # leafsize is inherited from the old tree
    @test knn(t2, q, 5, true) == knn(ref, q, 5, true)
    @test inrange(t2, q, 0.3, true) == inrange(ref, q, 0.3, true)
    @test t2.data === buf # storage was actually reused
    @test t2.indices === ibuf
    @test t2.tree_data.leafsize == 10

    # The consumed tree is invalidated: using it in any way throws
    @test_throws ArgumentError knn(t1, q, 5)
    @test_throws ArgumentError inrange(t1, q, 0.3)
    @test_throws ArgumentError allknn(t1, 2)
    @test_throws ArgumentError allnn(t1)
    @test_throws ArgumentError inrange_pairs(t1, 0.3)
    @test_throws ArgumentError treeroot(t1)
    @test_throws ArgumentError T!(t1, v2)
    @test_throws ArgumentError PeriodicTree(t1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    @test occursin("invalid", sprint(show, t1))

    # Rebuilding with more points grows the buffers
    t3 = T!(t2, v1)
    @test knn(t3, q, 5, true) == knn(T(v1; leafsize = 10), q, 5, true)

    # A PeriodicTree wrapping a consumed tree also throws on queries
    pdata = [rand(SVector{3,Float64}) for _ in 1:50]
    pk = T(pdata)
    pt = PeriodicTree(pk, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    T!(pk, v2)
    @test_throws ArgumentError knn(pt, q, 2)

    # data must not alias the old tree's storage
    @test_throws ArgumentError T!(t3, t3.data)

    # Non-reordered trees store the input array directly
    t4 = T(v1; reorder = false, leafsize = 10)
    t5 = T!(t4, v2)
    @test !t5.reordered && t5.data === v2
    @test knn(t5, q, 5, true) == knn(T(v2; reorder = false, leafsize = 10), q, 5, true)
end

@testset "rebuilding constructors (BruteTree!)" begin
    v1 = [rand(SVector{3,Float64}) for _ in 1:100]
    v2 = [rand(SVector{3,Float64}) for _ in 1:60]
    q = rand(3)

    b1 = BruteTree(v1)
    buf = b1.data
    b2 = BruteTree!(b1, v2)
    @test b2.data === buf
    @test knn(b2, q, 3, true) == knn(BruteTree(v2), q, 3, true)
    @test_throws ArgumentError knn(b1, q, 1)
    @test_throws ArgumentError BruteTree!(b1, v2)
    @test occursin("invalid", sprint(show, b1))

    # BruteTree just copies the data, so rebuilding from the tree's own
    # storage is fine
    b3 = BruteTree!(b2, b2.data)
    @test knn(b3, q, 3, true) == knn(BruteTree(v2), q, 3, true)
end

@testset "skip all points" begin
    data = rand(2, 20)
    for T in trees_with_brute
        tree = T(data)
        @test_throws ArgumentError nn(tree, rand(2), Returns(true))
        @test_throws ArgumentError allnn(tree, Returns(true))
        # knn returns fewer than k results when everything is skipped
        idxs, dists = knn(tree, rand(2), 3, true, Returns(true))
        @test isempty(idxs) && isempty(dists)
    end
end

end # module
