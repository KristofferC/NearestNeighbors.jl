# Does not test leafsize
@testset "inrange" begin
    @testset "metric" for metric in [Euclidean()]
        @testset "tree type" for TreeType in trees_with_brute
            function test(data)
                tree = TreeType(data, metric; leafsize=2)
                dosort = true

                idxs = inrange(tree, [1.1, 1.1, 1.1], 0.2, dosort)
                @test idxs == [8] # Only corner 8 at least 0.2 distance away from [1.1, 1.1, 1.1]
                counts = inrangecount(tree, [1.1, 1.1, 1.1], 0.2)
                @test counts == 1

                idxs = inrange(tree, [0.0, 0.0, 0.5], 0.6, dosort)
                @test idxs == [1, 2] # Corner 1 and 2 at least 0.6 distance away from [0.0, 0.0, 0.5]
                counts = inrangecount(tree, [0.0, 0.0, 0.5], 0.6)
                @test counts == 2

                idxs = inrange(tree, [0, 0, 0], 0.6, dosort)
                @test idxs == [1]
                counts = inrangecount(tree, [0, 0, 0], 0.6)
                @test counts == 1

                X = [0.0 0.0; 0.0 0.0; 0.5 0.0]
                idxs1 = inrange(tree, X, 0.6, dosort)
                idxs2 = inrange(tree, view(X,:,1:2), 0.6, dosort)
                @test idxs1 == idxs2
                @test idxs1[1] == [1,2]
                @test idxs1[2] == [1]
                counts1 = inrangecount(tree, X, 0.6)
                counts2 = inrangecount(tree, view(X,:,1:2), 0.6)
                @test counts1 == counts2
                @test counts1 == [2, 1]

                idxs = inrange(tree, [SVector{3,Float64}(0.0, 0.0, 0.5), SVector{3,Float64}(0.0, 0.0, 0.0)], 0.6, dosort)
                @test idxs[1] == [1,2]
                @test idxs[2] == [1]
                counts = inrangecount(tree, [SVector{3,Float64}(0.0, 0.0, 0.5), SVector{3,Float64}(0.0, 0.0, 0.0)], 0.6)
                @test counts == [2, 1]

                idxs = inrange(tree, [0.33333333333, 0.33333333333, 0.33333333333], 1, dosort)
                @test idxs == [1, 2, 3, 5]
                counts = inrangecount(tree, [0.33333333333, 0.33333333333, 0.33333333333], 1)
                @test counts == 4

                idxs = inrange(tree, [0.5, 0.5, 0.5], 0.2, dosort)
                @test idxs == []
                counts = inrangecount(tree, [0.5, 0.5, 0.5], 0.2)
                @test counts == 0

                idxs = inrange(tree, [0.5, 0.5, 0.5], 1.0, dosort)
                @test idxs == [1, 2, 3, 4, 5, 6, 7, 8]
                counts = inrangecount(tree, [0.5, 0.5, 0.5], 1.0)
                @test counts == 8

                @test_throws ArgumentError inrange(tree, rand(3), -0.1)
                @test_throws ArgumentError inrange(tree, rand(5), 1.0)

                empty_tree = TreeType(rand(3,0), metric)
                idxs = inrange(empty_tree, [0.5, 0.5, 0.5], 1.0)
                @test idxs == []
                counts = inrangecount(empty_tree, [0.5, 0.5, 0.5], 1.0)
                @test counts == 0

                one_point_tree = TreeType([0.5, 0.5, 0.5], metric)
                idxs = inrange(one_point_tree, data, 1.0)
                @test idxs == repeat([[1]], size(data, 2))
                counts = inrangecount(one_point_tree, data, 1.0)
                @test counts == repeat([1], size(data, 2))
            end
            data = [0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0;
                    0.0 0.0 1.0 1.0 0.0 0.0 1.0 1.0;
                    0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0] # 8 node cube
            test(data)
            test(view(data, :, :))
        end
    end
end

@testset "view" begin
    points = rand(SVector{3, Float64}, 100)
    kdtree = KDTree(points)
    idxs = inrange(kdtree, view(points, 1:10), 0.1)
    @test eltype(idxs) <: AbstractVector{Int}
end

@testset "mutating" begin
    for T in (KDTree, BallTree, BruteTree)
        data = T(rand(3, 100))
        idxs = Vector{Int32}(undef, 0)
        inrange!(idxs, data, [0.5, 0.5, 0.5], 1.0)
        idxs2 = inrange(data, [0.5, 0.5, 0.5], 3)
        @test idxs == idxs2
    end
end

@testset "inferrability matrix" begin
    function foo(data, point)
        b = KDTree(data)
        return inrange(b, point, 0.1)
    end

    @inferred foo([1.0 3.4; 4.5 3.4], [4.5; 3.4])
end
