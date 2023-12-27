# Does not test leafsize
# Does not test different metrics
import Distances.evaluate

@testset "knn" begin
    @testset "metric" for metric in [metrics; WeightedEuclidean(ones(2))]
        @testset "tree type" for TreeType in trees_with_brute
            function test(data)
                tree = TreeType(data, metric; leafsize=2)

                idxs, dists = knn(tree, [0.8, 0.8], 1)
                @test idxs[1] == 8 # Should be closest to top right corner
                @test evaluate(metric, [0.2, 0.2], zeros(2)) ≈ dists[1]

                idxs, dists = nn(tree, [0.8, 0.8])
                @test idxs == 8
                @test evaluate(metric, [0.2, 0.2], zeros(2)) ≈ dists

                idxs, dists = knn(tree, [0.1, 0.8], 3, true)
                @test idxs == [3, 2, 5]

                X = [0.8 0.1; 0.8 0.8]
                idxs1, dists1 = knn(tree, X, 1, true)
                idxs2, dists2 = knn(tree, view(X,:,1:2), 1, true)
                @test idxs1 == idxs2
                @test dists1 == dists2
                @test idxs1[1][1] == 8
                @test idxs1[2][1] == 3

                idxs, dists = nn(tree, X)
                @test idxs[1] == 8
                @test idxs[2] == 3

                idxs, dists = knn(tree, [SVector{2, Float64}(0.8,0.8), SVector{2, Float64}(0.1,0.8)], 1, true)
                @test idxs[1][1] == 8
                @test idxs[2][1] == 3

                idxs, dists = nn(tree, [SVector{2, Float64}(0.8,0.8), SVector{2, Float64}(0.1,0.8)])
                @test idxs[1] == 8
                @test idxs[2] == 3

                idxs, dists = knn(tree, [1//10, 8//10], 3, true)
                @test idxs == [3, 2, 5]

                @test_throws ArgumentError knn(tree, [0.1, 0.8], -1) # k < 0
                @test_throws ArgumentError knn(tree, [0.1, 0.8], 10) # k > n_points
                @test_throws ArgumentError knn(tree, [0.1], 10) # n_dim != trees dim

                empty_tree = TreeType(rand(2,0), metric; leafsize=2)
                idxs, dists = knn(empty_tree, [0.5, 0.5], 0, true)
                @test idxs == Int[]
                @test_throws ArgumentError knn(empty_tree, [0.1, 0.8], -1) # k < 0
                @test_throws ArgumentError knn(empty_tree, [0.1, 0.8], 1)  # k > n_points

                one_point_tree = TreeType([0.2, 0.8], metric)
                idxs, dists = knn(one_point_tree, data, 1)
                @test idxs == repeat([[1]], size(data, 2))
                @test_throws ArgumentError knn(one_point_tree, [0.1, 0.8], -1) # k < 0
                @test_throws ArgumentError knn(one_point_tree, [0.1, 0.8], 2)  # k > n_points
            end
            # 8 node rectangle
            data = [0.0 0.0 0.0 0.5 0.5 1.0 1.0 1.0;
                    0.0 0.5 1.0 0.0 1.0 0.0 0.5 1.0]
            test(data)
            test(view(data, :, :))
        end
    end
end

@testset "knn skip" begin
    @testset "tree type" for TreeType in trees_with_brute
        data = rand(2, 1000)
        function test(data)
            tree = TreeType(data)

            idxs, dists = knn(tree, data[:, 10], 2, true)
            first_idx = idxs[1]
            second_idx = idxs[2]

            idxs, dists = knn(tree, data[:, 10], 2, true, i -> i == first_idx)
            @test idxs[1] == second_idx
        end
        data = rand(2, 1000)
        test(data)
        test(view(data, :, :))
    end

    data = [[0.13380863416387367, 0.7845254987714512],[0.1563342025559629, 0.7956456895676077],[0.23320094627474594, 0.9055515160266435]]
    tree = KDTree(hcat(map(p -> [p[1], p[2]], data)...))
    nearest, distance = knn(tree, [0.15, 0.8], 3, true, x -> x == 2)
    @test nearest == [1, 3]
    @test distance ≈ [0.02239688629947563, 0.13440059522389006]
end

@testset "weighted" begin
    m = WeightedEuclidean([1e-5, 1])
    data = [
        0 0 1
        0 1 0.
    ]
    tree = KDTree(data, m, leafsize=1)
    p = [1, 0.9]
    @test nn(tree, p)[1] == 2
end

import Tensors
@testset "Tensors.Vec (no `StaticArrays.setindex` defined)" begin
    vdata = [rand(Tensors.Vec{2}) for _ in 1:10]
    sdata = SVector{2}.(vdata)
    vpoints = [rand(Tensors.Vec{2}) for _ in 1:2]
    spoints = SVector{2}.(vpoints)
    for TreeType in trees_with_brute
        vtree = TreeType(vdata)
        stree = TreeType(sdata)
        @test nn(vtree, vpoints) == nn(vtree, spoints) ==
              nn(stree, vpoints) == nn(stree, spoints)
    end
end


@testset "subarray" begin
    dynamic_view(X) = [NearestNeighbors.SizedVector{length(v)}(v) for v in eachslice(X; dims = ndims(X))]
    data = randn(1000, 1000)
    view = dynamic_view(data)
    @test KDTree(view) isa KDTree
end
