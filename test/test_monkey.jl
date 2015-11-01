import NearestNeighbors.MinkowskiMetric
# This contains a bunch of random tests that should hopefully detect if
# some edge case has been missed in the real tests

facts("NearestNeighbors.monkey") do

context("NearestNeighbors.monkey.knn") do

# Checks that we find existing point in the tree
# and that it is the closest
for metric in fullmetrics
    for TreeType in trees_with_brute
        if TreeType == KDTree && !isa(metric, MinkowskiMetric)
            continue
        end
        for i in 1:10
            dim_data = rand(1:4)
            size_data = rand(1000:1300)
            data = rand(dim_data, size_data)
            for j = 1:5
                if BruteTree == TreeType
                    tree = TreeType(data, metric)
                else
                    tree = TreeType(data, metric; leafsize = rand(1:15))
                end
                n = rand(1:size_data)
                idx, dist = knn(tree, data[:,n], rand(1:30), true)
                @fact issorted(dist) --> true
                @fact n --> idx[1]
            end
        end
    end
end


for metric in fullmetrics
    for TreeType in trees
        # Compares vs Brute Force
        if TreeType == KDTree && !isa(metric, MinkowskiMetric)
            continue
        end
        for i in 1:10
            dim_data = rand(1:5)
            size_data = rand(100:151)
            data = rand(dim_data, size_data)
            tree = TreeType(data, metric; leafsize = rand(1:15))
            btree = BruteTree(data, metric)
            k = rand(1:12)
            p = rand(dim_data)
            idx, dist = knn(tree, p, k)
            bidx, bdist = knn(tree, p, k)
            @fact idx --> bidx
            @fact dist --> roughly(bdist)
        end
    end
end
end # context

context("NearestNeighbors.monkey.inrange") do

# Test against brute force
for metric in fullmetrics
    for TreeType in trees
        if TreeType == KDTree && !isa(metric, MinkowskiMetric)
            continue
        end
        for i in 1:10
            dim_data = rand(1:6)
            size_data = rand(20:250)
            data = rand(dim_data, size_data)
            tree = TreeType(data, metric; leafsize = rand(1:8))
            btree = BruteTree(data, metric)
            p = 0.5 * ones(dim_data)
            r = 0.3

            idxs = inrange(tree, p, r, true)
            bidxs = inrange(btree, p, r, true)

            @fact idxs --> bidxs
        end
    end
end
end # context


context("NearestNeighbors.monkey.coupled") do
# Tests that the n-points in a random hyper sphere around
# a random point are all the n-closest points to that point
for metric in fullmetrics 
    for TreeType in trees_with_brute
        if TreeType == KDTree && !isa(metric, MinkowskiMetric)
            continue
        end
        for i in 1:50
            dim_data = rand(1:5)
            size_data = rand(100:1000)
            data = randn(dim_data, size_data)
            if BruteTree == TreeType
                tree = TreeType(data, metric)
            else
                tree = TreeType(data, metric; leafsize = rand(1:8))
            end
            point = randn(dim_data)
            idxs_ball = Int[]
            r = 0.1
            while length(idxs_ball) < 10
                r *= 2.0
                idxs_ball = inrange(tree,  point, r, true)
            end
            idxs_knn, dists = knn(tree, point, length(idxs_ball))

            @fact sort(idxs_knn) --> sort(idxs_ball)
        end
    end
end

end # context

end # facts
