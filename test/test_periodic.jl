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

    # The key requirement: distances should be equal (this is the main correctness test)
    @test dists_btree ≈ dists_pkdtree ≈ dists_pballtree

    # For indices, with ties, different trees may return different valid indices
    # We verify that all distances match the expected k-th nearest distance
    max_dist_brute = maximum(dists_btree)
    max_dist_kd = maximum(dists_pkdtree)
    max_dist_ball = maximum(dists_pballtree)

    # All maximum distances should be approximately equal (ensuring same k-th nearest distance)
    @test max_dist_brute ≈ max_dist_kd ≈ max_dist_ball

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

# Test mixed periodic/non-periodic dimensions
@testset "Mixed periodic/non-periodic dimensions" begin
    # Create simpler data that fits within bounds
    data = SVector{2, Float64}.([(1.0, 2.0), (8.0, 9.0)])
    bounds_min = [0.0, 0.0]
    bounds_max = [10.0, Inf]  # y-dimension is non-periodic

    kdtree = KDTree(data)
    ptree = PeriodicTree(kdtree, bounds_min, bounds_max)

    # Test that 3 combinations are generated (3^1 * 1^1 = 3 combinations)
    # For periodic x-dimension: [-1, 0, 1] = 3 combinations
    # For non-periodic y-dimension: [0] = 1 combination
    # Total: 3 combinations
    @test length(ptree.combos) == 3

    # Verify the combinations are correct
    combo_values = [combo[1] for combo in ptree.combos]  # x-dimension values
    @test 0 in combo_values  # Should have original box
    @test -1 in combo_values # Should have left periodic image
    @test 1 in combo_values  # Should have right periodic image

    # All y-dimension values should be 0 (non-periodic)
    y_values = [combo[2] for combo in ptree.combos]
    @test all(y -> y == 0, y_values)

    # Test actual KNN/inrange functionality for mixed dimensions
    # Use PeriodicEuclidean as ground truth (with Inf for non-periodic dimensions)
    btree = BruteTree(data, PeriodicEuclidean([10.0, Inf]))

    # Test various query points
    test_points = [
        [1.5, 2.5],   # Near first data point
        [8.5, 8.5],   # Near second data point
        [0.5, 5.0],   # Near left boundary (should wrap)
        [9.5, 5.0],   # Near right boundary (should wrap)
        [5.0, 1.0],   # Middle x, near bottom y
        [5.0, 10.0]   # Middle x, near top y
    ]

    for query_point in test_points
        # Test KNN
        for k in 1:length(data)
            idx_btree, dists_btree = knn(btree, query_point, k, true)
            idx_ptree, dists_ptree = knn(ptree, query_point, k, true)

            # Distances should match (main correctness test)
            @test dists_btree ≈ dists_ptree
        end

        # Test inrange
        for radius in [1.0, 2.0, 5.0, 10.0]
            idx_btree = sort(inrange(btree, query_point, radius))
            idx_ptree = sort(inrange(ptree, query_point, radius))
            @test idx_btree == idx_ptree
        end
    end
end

# Test comprehensive mixed periodic/non-periodic scenarios
@testset "Comprehensive mixed dimensions" begin
    # Test different combinations of periodic/non-periodic dimensions

    # Scenario 1: First dimension periodic, second non-periodic
    data1 = SVector{2, Float64}.([(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)])
    bounds_min1 = [0.0, 0.0]
    bounds_max1 = [8.0, Inf]

    kdtree1 = KDTree(data1)
    ptree1 = PeriodicTree(kdtree1, bounds_min1, bounds_max1)
    btree1 = BruteTree(data1, PeriodicEuclidean([8.0, Inf]))

    # Test boundary wrapping behavior
    test_points1 = [
        [0.5, 3.0],   # Near left boundary
        [7.5, 6.0],   # Near right boundary
        [4.0, 2.0],   # Middle
        [8.5, 4.0],   # Outside right boundary (should wrap to 0.5)
        [-0.5, 7.0]   # Outside left boundary (should wrap to 7.5)
    ]

    for query_point in test_points1
        for k in 1:length(data1)
            idx_btree, dists_btree = knn(btree1, query_point, k, true)
            idx_ptree, dists_ptree = knn(ptree1, query_point, k, true)
            @test dists_btree ≈ dists_ptree
        end

        for radius in [1.0, 2.0, 3.0, 5.0]
            idx_btree = sort(inrange(btree1, query_point, radius))
            idx_ptree = sort(inrange(ptree1, query_point, radius))
            @test idx_btree == idx_ptree
        end
    end

    # Scenario 2: First non-periodic, second periodic
    data2 = SVector{2, Float64}.([(2.0, 1.0), (5.0, 4.0), (8.0, 7.0)])
    bounds_min2 = [0.0, 0.0]
    bounds_max2 = [Inf, 8.0]

    kdtree2 = KDTree(data2)
    ptree2 = PeriodicTree(kdtree2, bounds_min2, bounds_max2)
    btree2 = BruteTree(data2, PeriodicEuclidean([Inf, 8.0]))

    test_points2 = [
        [3.0, 0.5],   # Near bottom boundary
        [6.0, 7.5],   # Near top boundary
        [4.0, 4.0],   # Middle
        [7.0, 8.5],   # Outside top boundary (should wrap to 0.5)
        [1.0, -0.5]   # Outside bottom boundary (should wrap to 7.5)
    ]

    for query_point in test_points2
        for k in 1:length(data2)
            idx_btree, dists_btree = knn(btree2, query_point, k, true)
            idx_ptree, dists_ptree = knn(ptree2, query_point, k, true)
            @test dists_btree ≈ dists_ptree
        end

        for radius in [1.0, 2.0, 3.0, 5.0]
            idx_btree = sort(inrange(btree2, query_point, radius))
            idx_ptree = sort(inrange(ptree2, query_point, radius))
            @test idx_btree == idx_ptree
        end
    end

    # Scenario 3: 3D with mixed dimensions
    data3 = SVector{3, Float64}.([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)])
    bounds_min3 = [0.0, 0.0, 0.0]
    bounds_max3 = [8.0, Inf, 10.0]  # x and z periodic, y non-periodic

    kdtree3 = KDTree(data3)
    ptree3 = PeriodicTree(kdtree3, bounds_min3, bounds_max3)
    btree3 = BruteTree(data3, PeriodicEuclidean([8.0, Inf, 10.0]))

    test_points3 = [
        [1.5, 3.0, 4.0],   # Near first data point
        [7.5, 6.0, 9.5],   # Near boundaries in periodic dimensions
        [4.0, 5.0, 5.0],   # Middle
        [8.5, 7.0, 10.5]   # Outside periodic boundaries
    ]

    for query_point in test_points3
        for k in 1:min(2, length(data3))  # Test fewer k values for 3D
            idx_btree, dists_btree = knn(btree3, query_point, k, true)
            idx_ptree, dists_ptree = knn(ptree3, query_point, k, true)
            @test dists_btree ≈ dists_ptree
        end

        for radius in [2.0, 4.0, 6.0]
            idx_btree = sort(inrange(btree3, query_point, radius))
            idx_ptree = sort(inrange(ptree3, query_point, radius))
            @test idx_btree == idx_ptree
        end
    end
end

# Test boundary cases and edge conditions
@testset "Boundary cases and edge conditions" begin
    # Test near-boundary points that should find neighbors through periodic wrapping
    data = SVector{2, Float64}.([(0.5, 1.0), (9.5, 8.0), (5.0, 5.0)])
    bounds_max = [10.0, 10.0]

    # Query point very close to boundary - should find wrapped neighbors
    query_point = [0.1, 1.5]  # Very close to (0.5, 1.0) and should also find (9.5, 8.0) through wrapping

    for reorder in [false, true]
        pkdtree, pballtree, btree = create_trees(data, bounds_max, reorder)

        # Test that periodic tree finds same neighbors as brute tree
        test_periodic_euclidean_against_brute_knn(pkdtree, pballtree, btree, query_point, 3)

        # Test with radius that should capture wrapped neighbors
        test_periodic_euclidean_against_brute_inrange(pkdtree, pballtree, btree, query_point, 2.0)
    end

    # Test point exactly at boundary
    query_point = [0.0, 5.0]
    for reorder in [false, true]
        pkdtree, pballtree, btree = create_trees(data, bounds_max, reorder)
        test_periodic_euclidean_against_brute_knn(pkdtree, pballtree, btree, query_point, 2)
        test_periodic_euclidean_against_brute_inrange(pkdtree, pballtree, btree, query_point, 1.5)
    end

    # Test point exactly at opposite boundary
    query_point = [10.0, 5.0]
    for reorder in [false, true]
        pkdtree, pballtree, btree = create_trees(data, bounds_max, reorder)
        test_periodic_euclidean_against_brute_knn(pkdtree, pballtree, btree, query_point, 2)
        test_periodic_euclidean_against_brute_inrange(pkdtree, pballtree, btree, query_point, 1.5)
    end
end

# Test with different data distributions
@testset "Different data distributions" begin
    # Dense data near boundaries
    data = SVector{2, Float64}.([(0.1, 0.1), (0.2, 0.3), (9.8, 9.9), (9.7, 9.6), (5.0, 5.0)])
    bounds_max = [10.0, 10.0]

    query_points = [
        [0.0, 0.0],    # Corner
        [10.0, 10.0],  # Opposite corner
        [0.15, 0.2],   # Near dense cluster
        [9.75, 9.75]   # Near other dense cluster
    ]

    for query_point in query_points
        for reorder in [false, true]
            pkdtree, pballtree, btree = create_trees(data, bounds_max, reorder)

            # Test all k values
            for k in 1:length(data)
                test_periodic_euclidean_against_brute_knn(pkdtree, pballtree, btree, query_point, k)
            end

            # Test multiple radii
            for radius in [0.5, 1.0, 2.0, 5.0]
                test_periodic_euclidean_against_brute_inrange(pkdtree, pballtree, btree, query_point, radius)
            end
        end
    end
end

# Test specific periodic scenarios with known correct answers
@testset "Periodic boundary verification" begin
    # Simple case: two points that should be closest through periodic boundary
    data = SVector{2, Float64}.([(0.5, 5.0), (9.5, 5.0)])
    bounds_max = [10.0, 10.0]

    # Query at x=0.0 should find (0.5, 5.0) as closest, but (9.5, 5.0) should be very close too via periodicity
    query_point = [0.0, 5.0]

    for reorder in [false, true]
        pkdtree, pballtree, btree = create_trees(data, bounds_max, reorder)

        # Test KNN - both trees should give same results as brute tree
        test_periodic_euclidean_against_brute_knn(pkdtree, pballtree, btree, query_point, 2)

        # Test inrange with small radius that should capture periodic neighbor
        test_periodic_euclidean_against_brute_inrange(pkdtree, pballtree, btree, query_point, 1.0)
    end

    # Test another scenario: query outside the box should wrap around
    query_point = [10.5, 5.0]  # Should be equivalent to [0.5, 5.0] due to periodicity

    for reorder in [false, true]
        pkdtree, pballtree, btree = create_trees(data, bounds_max, reorder)
        test_periodic_euclidean_against_brute_knn(pkdtree, pballtree, btree, query_point, 2)
        test_periodic_euclidean_against_brute_inrange(pkdtree, pballtree, btree, query_point, 2.0)
    end
end

# Test extensive periodic scenarios
@testset "Extensive periodic testing" begin
    # Larger dataset with more complex periodic interactions
    data = SVector{2, Float64}.([(0.1, 0.1), (2.0, 3.0), (5.0, 5.0), (8.0, 2.0), (9.9, 9.9)])
    bounds_max = [10.0, 10.0]

    # Test many query points systematically
    test_points = [
        [0.0, 0.0],     # Corner
        [5.0, 5.0],     # Center
        [10.0, 0.0],    # Corner
        [0.0, 10.0],    # Corner
        [10.0, 10.0],   # Corner
        [0.05, 0.05],   # Very close to boundary
        [9.95, 9.95],   # Very close to opposite boundary
        [11.0, 1.0],    # Outside box (should wrap)
        [-1.0, 9.0]     # Outside box (should wrap)
    ]

    for query_point in test_points
        for reorder in [false, true]
            pkdtree, pballtree, btree = create_trees(data, bounds_max, reorder)

            # Test KNN for various k values
            for k in 1:min(3, length(data))
                test_periodic_euclidean_against_brute_knn(pkdtree, pballtree, btree, query_point, k)
            end

            # Test inrange for various radii
            for radius in [0.5, 1.0, 2.0, 4.0]
                test_periodic_euclidean_against_brute_inrange(pkdtree, pballtree, btree, query_point, radius)
            end
        end
    end
end

# Test data validation
@testset "Data validation" begin
    # Test that data outside bounds is rejected
    data_good = SVector{2, Float64}.([(1.0, 2.0), (3.0, 4.0)])
    data_bad = SVector{2, Float64}.([(1.0, 2.0), (11.0, 4.0)])  # 11.0 > 10.0

    kdtree_good = KDTree(data_good)
    kdtree_bad = KDTree(data_bad)

    bounds_min = [0.0, 0.0]
    bounds_max = [10.0, 10.0]

    # Should work with good data
    @test isa(PeriodicTree(kdtree_good, bounds_min, bounds_max), PeriodicTree)

    # Should fail with bad data
    @test_throws ArgumentError PeriodicTree(kdtree_bad, bounds_min, bounds_max)

    # Test dimension mismatch
    @test_throws ArgumentError PeriodicTree(kdtree_good, [0.0], bounds_max)
    @test_throws ArgumentError PeriodicTree(kdtree_good, bounds_min, [10.0])

    # Test invalid box dimensions
    @test_throws ArgumentError PeriodicTree(kdtree_good, [0.0, 0.0], [-1.0, 10.0])
    @test_throws ArgumentError PeriodicTree(kdtree_good, [5.0, 0.0], [3.0, 10.0])
end
