@testset "datafreetree" begin
    data = rand(2,100)
    data2 = rand(2,100)
    data3 = rand(3,100)
    t = DataFreeTree(KDTree, data)
    @test_throws ArgumentError injectdata(t, data2) 
    @test_throws DimensionMismatch injectdata(t, data3) 

    for typ in [KDTree, BallTree]
        dfilename = tempname()
        rfilename = tempname()
        d = 2
        n = 1000
        data = Mmap.mmap(dfilename, Matrix{Float32}, (d, n))
        data[:] = rand(Float32, d, n)
        reorderbuffer = Mmap.mmap(rfilename, Matrix{Float32}, (d, n))
        t = injectdata(DataFreeTree(typ, data), data)
        tr = injectdata(DataFreeTree(typ, data, reorderbuffer = reorderbuffer), reorderbuffer)
        for i = 1:n
            @test knn(t, data[:,i], 3)[2] == knn(tr, data[:,i], 3)[2]
        end
        rm(dfilename)
        rm(rfilename)
    end
end
