# Test that KDTree rejects data containing NaNs
@testset "KDTree rejects NaNs" begin
    data_vec = [SVector{2,Float64}(NaN, 0.0), SVector{2,Float64}(1.0, 1.0)]
    @test_throws DomainError KDTree(data_vec)

    data_mat = [NaN 0.0; 1.0 1.0]
    @test_throws DomainError KDTree(data_mat)
end