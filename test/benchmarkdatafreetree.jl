using NearestNeighbors

runtimes = []
runtimesreordered = []
for n in [10, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    dfilename = tempname()
    rfilename = tempname()
    d = 2
    data = Mmap.mmap(dfilename, Matrix{Float32}, (d, n))
    data[:] = rand(Float32, d, n)
    reorderbuffer = Mmap.mmap(rfilename, Matrix{Float32}, (d, n))
    t = injectdata(DataFreeTree(KDTree, data), data)
    tr = injectdata(DataFreeTree(KDTree, data, reorderbuffer = reorderbuffer), reorderbuffer)

    r = @elapsed for i = 1:1000
        ind = rand(1:n)
        knn(t, data[:,ind], 3)[2]
    end
    push!(runtimes, r)

    r = @elapsed for i = 1:1000
        ind = rand(1:n)
        knn(tr, data[:,ind], 3)[2]
    end
    push!(runtimesreordered, r)
end
@show runtimes
@show runtimesreordered

