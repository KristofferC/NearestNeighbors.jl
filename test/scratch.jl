NearestNeighbors.knn_point!(tree, [.1,.2,.3], false, Vector{Float64}(5), Vector{Int}(5), NearestNeighbors.always_false)

NearestNeighbors._knn(tree, [.1,.2,.3], fill(-1,5), fill(typemax(Float64), 5), NearestNeighbors.always_false)

NearestNeighbors.heap_sort_inplace!(rand(Float64, 5), [1,4,5,3,2])

import Distances.evaluate


for metric in metrics
    for TreeType in trees_with_brute
        # 8 node rectangle
        data = [0.0 0.0 0.0 0.5 0.5 1.0 1.0 1.0;
                0.0 0.5 1.0 0.0 1.0 0.0 0.5 1.0]

        tree = TreeType(data, metric; leafsize=2)

        idxs, dists = knn(tree, [0.8, 0.8], 1)
         idxs[1] == 8 # Should be closest to top right corner
         evaluate(metric, [0.2, 0.2], zeros(2)) ≈ dists[1]

        idxs, dists = knn(tree, [0.1, 0.8], 3, true)
         idxs == [3, 2, 5]

        idxs, dists = knn(tree, [0.8 0.1; 0.8 0.8], 1, true)
         idxs[1][1] == 8
         idxs[2][1] == 3

        idxs, dists = knn(tree, [SVector{2, Float64}(0.8,0.8), SVector{2, Float64}(0.1,0.8)], 1, true)
         idxs[1][1] == 8
         idxs[2][1] == 3

        idxs, dists = knn(tree, [1//10, 8//10], 3, true)
         idxs == [3, 2, 5]
    end
end

for metric in [Euclidean()]
for TreeType in trees_with_brute
    data = [0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0;
            0.0 0.0 1.0 1.0 0.0 0.0 1.0 1.0;
            0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0] # 8 node cube

    tree = TreeType(data, metric; leafsize=2)
    dosort = true

    idxs = inrange(tree, [1.1, 1.1, 1.1], 0.2, dosort)
     idxs == [8] # Only corner 8 at least 0.2 distance away from [1.1, 1.1, 1.1]

    idxs = inrange(tree, [0.0, 0.0, 0.5], 0.6, dosort)
     idxs == [1, 2] # Corner 1 and 2 at least 0.6 distance away from [0.0, 0.0, 0.5]

    idxs = inrange(tree, [0, 0, 0], 0.6, dosort)
     idxs == [1]

    idxs = inrange(tree, [0.0 0.0; 0.0 0.0; 0.5 0.0], 0.6, dosort)
     idxs[1] == [1,2]
     idxs[2] == [1]

    idxs = inrange(tree, [SVector{3,Float64}(0.0, 0.0, 0.5), SVector{3,Float64}(0.0, 0.0, 0.0)], 0.6, dosort)
     idxs[1] == [1,2]
     idxs[2] == [1]

    idxs = inrange(tree, [0.33333333333, 0.33333333333, 0.33333333333], 1, dosort)
     idxs == [1, 2, 3, 5]

    idxs = inrange(tree, [0.5, 0.5, 0.5], 0.2, dosort)
     idxs == []

    idxs = inrange(tree, [0.5, 0.5, 0.5], 1.0, dosort)
     idxs == [1, 2, 3, 4, 5, 6, 7, 8]

    empty_tree = TreeType(rand(3,0), metric)
    idxs = inrange(empty_tree, [0.5, 0.5, 0.5], 1.0)
     idxs == []
end
end

for metric in fullmetrics
for TreeType in trees_with_brute
    @testset "type" for T in (Float32, Float64)
        @testset "knn monkey" begin
            # Checks that we find existing point in the tree
            # and that it is the closest
            if TreeType == KDTree && !isa(metric, MinkowskiMetric)
                continue
            elseif TreeType == BallTree && isa(metric, Hamming)
                continue
            end
            for i in 1:30
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
            for i in 1:30
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
            for i in 1:30
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
            for i in 1:50
                dim_data = rand(1:5)
                size_data = rand(100:1000)
                data = randn(T, dim_data, size_data)
                tree = TreeType(data, metric; leafsize = rand(1:8))
                point = randn(dim_data)
                idxs_ball = Int[]
                r = 0.1
                while length(idxs_ball) < 10
                    r *= 2.0
                    idxs_ball = inrange(tree,  point, r, true)
                end
                idxs_knn, dists = knn(tree, point, length(idxs_ball))

                @test sort(idxs_knn) == sort(idxs_ball)
            end
        end
    end
end
end


data = rand(2,100)
data2 = rand(2,100)
data3 = rand(3,100)
t = DataFreeTree(KDTree, data)

for typ in [KDTree, BallTree]
    dfilename = tempname()
    rfilename = tempname()
    d = 2
    n = 100
    data = Mmap.mmap(dfilename, Matrix{Float32}, (d, n))
    data[:] = rand(Float32, d, n)
    reorderbuffer = Mmap.mmap(rfilename, Matrix{Float32}, (d, n))
    t = injectdata(DataFreeTree(typ, data), data)
    tr = injectdata(DataFreeTree(typ, data, reorderbuffer = reorderbuffer), reorderbuffer)
    for i = 1:n
        knn(t, data[:,i], 3) == knn(tr, data[:,i], 3)
    end
    rm(dfilename)
    rm(rfilename)
end

data = rand(2,1000)
buf = zeros(data)
for typ in [KDTree, BallTree]
    t = injectdata(DataFreeTree(typ, data, indicesfor = :data), data)
    t2 = injectdata(DataFreeTree(typ, data, reorderbuffer = buf, indicesfor = :reordered), buf)
    data[:,knn(t, data[:,1], 3)[1]] == buf[:,knn(t2, data[:,1], 3)[1]]
end
end
