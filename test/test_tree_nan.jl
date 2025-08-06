# Tests for KDTree, BallTree, BruteTree that reject data containing NaNs

@testset "Trees reject NaNs" begin
    data_vec = [SVector{2,Float64}(NaN, 0.0), SVector{2,Float64}(1.0, 1.0)]
    data_mat = [NaN 0.0; 1.0 1.0]

    for TreeType in (KDTree, BallTree, BruteTree)
        @test_throws DomainError TreeType(data_vec)
        @test_throws DomainError TreeType(data_mat)
    end
end

@testset "knn rejects NaNs" begin
    for TreeType in (KDTree, BallTree, BruteTree)
        data = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 1.0)]
        tree = TreeType(data)

        # Single query point (vector) containing NaN
        @test_throws DomainError knn(tree, [NaN, 0.0], 1)

        # Vector-of-vectors query containing NaN
        query_vec = [SVector{2,Float64}(NaN, 0.0)]
        @test_throws DomainError knn(tree, query_vec, 1)

        # Matrix query containing NaN
        query_mat = [NaN 0.0; 0.0 1.0]
        @test_throws DomainError knn(tree, query_mat, 1)
    end
end
