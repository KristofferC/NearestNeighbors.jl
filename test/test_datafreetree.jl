module TestDatafreetree
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "TestSetup.jl"))
using ..Main.TestSetup
using NearestNeighbors
using Test
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

@testset "Matrix reorder buffers" begin
    for Tree in (KDTree, BallTree), T in (Float32, Float64)
        data = T[8 1 7 2 6 3 5 4; 1 2 3 4 5 6 7 8]
        for storedata in (false, true)
            buffer = Matrix{T}(undef, size(data))
            if storedata
                tree = Tree(data; reorderbuffer=buffer, leafsize=1)
            else
                df = DataFreeTree(Tree, data; reorderbuffer=buffer, leafsize=1)
                tree = injectdata(df, buffer)
            end
            @test buffer == data[:, tree.indices]
            reference = BruteTree(data)
            @test knn(tree, T[1, 2], 3, true) == knn(reference, T[1, 2], 3, true)
            @test inrange(tree, T[1, 2], T(4), true) == inrange(reference, T[1, 2], T(4), true)
        end
        @test_throws DimensionMismatch Tree(data; reorderbuffer=zeros(T, 1, 8))
    end
end

end # module
