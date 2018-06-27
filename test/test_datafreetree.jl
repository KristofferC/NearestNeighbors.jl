using Mmap

@testset "datafreetree" begin
    data = rand(2,100)
    data2 = rand(2,100)
    data3 = rand(3,100)
    t = DataFreeTree(KDTree, data)
    @test_throws ArgumentError injectdata(t, data2)
    @test_throws DimensionMismatch injectdata(t, data3)
    for typ in [KDTree, BallTree]
        dfilename = tempname()
        d = 2
        n = 100
        data = Mmap.mmap(dfilename, Matrix{Float32}, (d, n))
        data[:] = rand(Float32, d, n)
        t = injectdata(DataFreeTree(typ, data), data)
        tr = typ(data)
        for i = 1:n
            @test knn(t, data[:,i], 3) == knn(tr, data[:,i], 3)
        end
        rm(dfilename)
    end
end
