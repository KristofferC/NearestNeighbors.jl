# Tests for KDTree, BallTree, BruteTree that reject data containing NaNs

@testset "KDTree rejects NaNs" begin
    data_vec = [SVector{2,Float64}(NaN, 0.0), SVector{2,Float64}(1.0, 1.0)]
    @test_throws DomainError KDTree(data_vec)

    data_mat = [NaN 0.0; 1.0 1.0]
    @test_throws DomainError KDTree(data_mat)
end

@testset "BallTree rejects NaNs" begin
    data_vec = [SVector{2,Float64}(NaN, 0.0), SVector{2,Float64}(1.0, 1.0)]
    @test_throws DomainError BallTree(data_vec)

    data_mat = [NaN 0.0; 1.0 1.0]
    @test_throws DomainError BallTree(data_mat)
end

@testset "BruteTree rejects NaNs" begin
    data_vec = [SVector{2,Float64}(NaN, 0.0), SVector{2,Float64}(1.0, 1.0)]
    @test_throws DomainError BruteTree(data_vec)

    data_mat = [NaN 0.0; 1.0 1.0]
    @test_throws DomainError BruteTree(data_mat)
end
