import NearestNeighbors.MinkowskiMetric
# This contains a bunch of random tests that should hopefully detect if
# some edge case has been missed in the real tests
@testset "metric $metric" for metric in fullmetrics
    nrep = 30
    @testset "tree type $TreeType" for TreeType in trees_with_brute
        @testset "element type $T" for T in (Float32, Float64)
            @testset "knn monkey" begin
                # Checks that we find existing point in the tree
                # and that it is the closest
                if TreeType == KDTree && !isa(metric, MinkowskiMetric)
                    continue
                elseif TreeType == BallTree && isa(metric, Hamming)
                    continue
                end
                for i in 1:nrep
                    dim_data = rand(1:4)
                    size_data = rand(1000:1300)
                    data = rand(T, dim_data, size_data)
                    for j = 1:5
                        tree = TreeType(data, metric; leafsize = rand(1:15))
                        n = rand(1:size_data)
                        idx, dist = knn(tree, data[:,n], rand(1:30), true)
                        @test issorted(dist) == true
                        @test n == idx[1]
                    end
                end

                # Compares vs Brute Force
                for i in 1:nrep
                    dim_data = rand(1:5)
                    size_data = rand(100:151)
                    data = rand(T, dim_data, size_data)
                    tree = TreeType(data, metric; leafsize = rand(1:15))
                    btree = BruteTree(data, metric)
                    k = rand(1:12)
                    p = rand(dim_data)
                    idx, dist = knn(tree, p, k)
                    bidx, bdist = knn(tree, p, k)
                    @test idx == bidx
                    @test dist ≈ bdist
                end
            end

            @testset "inrange monkey" begin
                # Test against brute force
                for i in 1:nrep
                    dim_data = rand(1:6)
                    size_data = rand(20:250)
                    data = rand(T, dim_data, size_data)
                    tree = TreeType(data, metric; leafsize = rand(1:8))
                    btree = BruteTree(data, metric)
                    p = 0.5 * ones(dim_data)
                    r = 0.3

                    idxs = inrange(tree, p, r, true)
                    bidxs = inrange(btree, p, r, true)

                    @test idxs == bidxs
                end
            end

            @testset "coupled monkey" begin
                for i in 1:nrep
                    dim_data = rand(1:5)
                    size_data = rand(100:1000)
                    data = randn(T, dim_data, size_data)

                    lf = rand(1:8)
                    tree = TreeType(data, metric; leafsize = lf)

                    if TreeType == BallTree # this caught a race-condition in an early version of the parallel BallTree code
                        tree2 = TreeType(data, metric; leafsize = lf, parallel = true, parallel_size = 0) # triggering parallel code
                        @test tree.data == tree2.data
                        @test tree.hyper_spheres[1] == tree2.hyper_spheres[1]
                        @test tree.indices == tree2.indices
                        @test tree.metric == tree2.metric
                        @test tree.tree_data == tree2.tree_data
                        @test tree.reordered == tree2.reordered
                    end

                    point = randn(dim_data)
                    idxs_ball = Int[]
                    r = 0.1
                    while length(idxs_ball) < 10
                        r *= 2.0
                        idxs_ball = inrange(tree, point, r, true)
                    end
                    idxs_knn, dists = knn(tree, point, length(idxs_ball))

                    @test sort(idxs_knn) == sort(idxs_ball)
                end
            end
        end
    end
end
