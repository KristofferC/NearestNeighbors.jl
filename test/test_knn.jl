# Does not test leafsize
# Does not test different metrics
import Distances.evaluate

@testset "knn" begin
    @testset "metric" for metric in metrics
        @testset "tree type" for TreeType in trees_with_brute
            # 8 node rectangle
            data = [0.0 0.0 0.0 0.5 0.5 1.0 1.0 1.0;
                    0.0 0.5 1.0 0.0 1.0 0.0 0.5 1.0]

            tree = TreeType(data, metric; leafsize=2)

            idxs, dists = knn(tree, [0.8, 0.8], 1)
            @test idxs[1] == 8 # Should be closest to top right corner
            @test evaluate(metric, [0.2, 0.2], zeros(2)) ≈ dists[1]

            idxs, dists = knn(tree, [0.1, 0.8], 3)
            @test idxs == [5, 2, 3]

            idxs, dists = knn(tree, [1//10, 8//10], 3)
            @test idxs == [5, 2, 3]

            @test_throws ArgumentError knn(tree, [0.1, 0.8], 10) # k > n_points
            @test_throws ArgumentError knn(tree, [0.1], 10) # n_dim != trees dim

        end
    end
end

@testset "knn skip" begin
    @testset "tree type" for TreeType in trees_with_brute
        data = rand(2, 1000)
        tree = TreeType(data)
        
        idxs, dists = knn(tree, data[:, 10], 2, true)
        first_idx = idxs[1]
        second_idx = idxs[2]
        
        idxs, dists = knn(tree, data[:, 10], 2, true, i -> i == first_idx)
        @test idxs[1] == second_idx
    end
end
