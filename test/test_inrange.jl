# Does not test leafsize
@testset "inrange" begin
    @testset "metric" for metric in [Euclidean()]
        @testset "tree type" for TreeType in trees_with_brute
            data = [0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0;
                    0.0 0.0 1.0 1.0 0.0 0.0 1.0 1.0;
                    0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0] # 8 node cube

            tree = TreeType(data, metric; leafsize=2)
            dosort = true

            idxs = inrange(tree, [1.1, 1.1, 1.1], 0.2, dosort)
            @test idxs == [8] # Only corner 8 at least 0.2 distance away from [1.1, 1.1, 1.1]

            idxs = inrange(tree, [0.0, 0.0, 0.5], 0.6, dosort)
            @test idxs == [1, 2] # Corner 1 and 2 at least 0.6 distance away from [0.0, 0.0, 0.5]

            idxs = inrange(tree, [0, 0, 0], 0.6, dosort)
            @test idxs == [1]

            idxs = inrange(tree, [0.0 0.0; 0.0 0.0; 0.5 0.0], 0.6, dosort)
            @test idxs[1] == [1,2]
            @test idxs[2] == [1]

            idxs = inrange(tree, [SVector{3,Float64}(0.0, 0.0, 0.5), SVector{3,Float64}(0.0, 0.0, 0.0)], 0.6, dosort)
            @test idxs[1] == [1,2]
            @test idxs[2] == [1]

            idxs = inrange(tree, [0.33333333333, 0.33333333333, 0.33333333333], 1, dosort)
            @test idxs == [1, 2, 3, 5]

            idxs = inrange(tree, [0.5, 0.5, 0.5], 0.2, dosort)
            @test idxs == []

            idxs = inrange(tree, [0.5, 0.5, 0.5], 1.0, dosort)
            @test idxs == [1, 2, 3, 4, 5, 6, 7, 8]

            @test_throws ArgumentError inrange(tree, rand(3), -0.1)
            @test_throws ArgumentError inrange(tree, rand(5), 1.0)

            empty_tree = TreeType(rand(3,0), metric)
            idxs = inrange(empty_tree, [0.5, 0.5, 0.5], 1.0)
            @test idxs == []
        end
    end
end
