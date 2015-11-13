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

    r = @elapsed knn(t, data[:,rand(1:n, 1000)], 3)
    push!(runtimes, r)

    r = @elapsed knn(tr, data[:,rand(1:n,1000)], 3)
    push!(runtimesreordered, r)
end
println("Speedup of reordered over unordered:")
println(runtimes[2:end]./runtimesreordered[2:end])

