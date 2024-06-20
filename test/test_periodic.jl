using Test

using NearestNeighbors, StaticArrays, Distances

function create_trees(data, bounds_max, reorder)
    kdtree = KDTree(data; leafsize=1, reorder)
    balltree = BallTree(data; leafsize=1, reorder)
    bounds_min = zeros(length(bounds_max))

    pkdtree = PeriodicTree(kdtree, bounds_min, bounds_max)
    pballtree = PeriodicTree(balltree, bounds_min, bounds_max)
    btree = BruteTree(data, PeriodicEuclidean(bounds_max))
    return pkdtree, pballtree, btree
end

function test_periodic_euclidean_against_brute_inrange(pkdtree, pballtree, btree, point, r)
    idx_btree = sort(inrange(btree, point, r))
    idx_pkdtree = sort(inrange(pkdtree, point, r))
    idx_pballtree = sort(inrange(pballtree, point, r))
    @test idx_btree == idx_pkdtree == idx_pballtree
end

function test_periodic_euclidean_against_brute_knn(pkdtree, pballtree, btree, point, k)
    idx_btree, dists_btree = knn(btree, point, k, true)
    idx_pkdtree, dists_pkdtree = knn(pkdtree, point, k, true)
    idx_pballtree, dists_pballtree = knn(pballtree, point, k, true)

    @test dists_btree ≈ dists_pkdtree ≈ dists_pballtree
    @test idx_btree == idx_pkdtree == idx_pballtree

    return dists_pkdtree
end

function test_data_bounds_point(data, bounds_max, point)
    for reorder = (false, true)
        pkdtree, pballtree, btree = create_trees(data, bounds_max, reorder)
        for k in 1:length(data)
            dists = test_periodic_euclidean_against_brute_knn(pkdtree, pballtree, btree, point, k)
            r = maximum(dists) + 0.001
            test_periodic_euclidean_against_brute_inrange(pkdtree, pballtree, btree, point, r)
        end
    end
end

data = SVector{2, Float64}.([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)])
bounds_max = (10.0, 10.0)
point = [8.9, 1.9]
test_data_bounds_point(data, bounds_max, point)

data = SVector{3, Float64}.([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15)])
bounds_max = (20.0, 20.0, 20.0)
point = [18.0, 19.0, 0.0]
test_data_bounds_point(data, bounds_max, point)
