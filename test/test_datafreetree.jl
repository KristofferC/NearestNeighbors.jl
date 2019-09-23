using Mmap

@testset "datafreetree" begin
    function test(data, data2, data3)
        t = DataFreeTree(KDTree, data)
        @test_throws ArgumentError injectdata(t, data2)
        @test_throws DimensionMismatch injectdata(t, data3)
        for typ in [KDTree, BallTree]
            dfilename = tempname()
            d = 2
            n = 100
            mktemp() do _, io
                data = Mmap.mmap(io, Matrix{Float32}, (d, n))
                data[:] = rand(Float32, d, n)
                t = injectdata(DataFreeTree(typ, data), data)
                tr = typ(data)
                for i = 1:n
                    @test knn(t, data[:,i], 3) == knn(tr, data[:,i], 3)
                end
                finalize(data)
            end
        end
    end
    data = rand(2,100)
    data2 = rand(2,100)
    data3 = rand(3,100)
    test(data, data2, data3)
    test(view(data, :, :), view(data2, :, :), view(data3, :, :))
end
