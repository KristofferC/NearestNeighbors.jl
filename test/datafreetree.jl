@testset "datafreetree" begin
    data = rand(2,100)
    data2 = rand(2,100)
    data3 = rand(3,100)
    t = DataFreeTree(KDTree, data)
    @test_throws ArgumentError injectdata(t, data2) 
    @test_throws DimensionMismatch injectdata(t, data3) 
end
