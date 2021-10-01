using NearestNeighbors
using Test

# Test for issue #78
@testset "infs on data" begin
    for _ in 1:11

        coords = [
            29882.5 25974.3 Inf  Inf 17821.8 Inf Inf Inf Inf Inf 16322.0;
            9279.86 9286.35 Inf Inf 10320.4 Inf Inf Inf Inf Inf 11459.0;
            0.0 0.0 Inf Inf 0.0 Inf Inf Inf Inf Inf 0.0]
        point =  [17889.55, 2094.45, 0.0]

        tree = BallTree(coords)
        @show idx, _ = knn(tree, point, 1)
        @test 1 <= idx[1] <= 11
    end
end

# Test for issue #125
@testset "infs on data" begin
    for _ in 1:111

        Ndim = 35
        Npt = 408

        data = randn(Ndim, Npt)

        data[:,1] .= Inf
        # tree = KDTree(data)
        tree = BallTree(data)

        pointnan = repeat([NaN], Ndim)
        indnan,distnan = nn(tree, pointnan)
        # @test 1 <= indnan <= Npt

        pointinf = repeat([Inf], Ndim)
        indinf, distinf = nn(tree, pointinf)
        @test 1 <= indinf <= Npt
    end
end
