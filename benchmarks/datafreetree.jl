using NearestNeighbors
using Benchmarks

runtimes = []
runtimesreordered = []

function create_tree(n, reorder=false)
    filename = tempname()
    d = 10
    data = Mmap.mmap(filename, Matrix{Float32}, (d, n))
    data[:] = rand(Float32, d, n)
    if reorder
        reorderbuffer = Mmap.mmap(filename, Matrix{Float32}, (d, n))
        t = injectdata(DataFreeTree(KDTree, data, reorderbuffer = reorderbuffer), reorderbuffer)
    else
        t = injectdata(DataFreeTree(KDTree, data), data)
    end

    return t, data, filename
end

function knnbench(tree, data, n, N)
    ind = rand(1:n, N)
    knn(tree, data[:,ind], 3)[2]
end

function bench()
    runtimes = []
    runtimesreordered = []
    ns = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    for n in ns
        t, data, filename = create_tree(n)
        tr, datar, filenamer = create_tree(n, true)

        bm = @benchmark knnbench(t, data, n, 1000)
        push!(runtimes, mean(bm.samples.elapsed_times) / 1e9)

        bmr = @benchmark knnbench(tr, datar, n, 1000)
        push!(runtimesreordered, mean(bmr.samples.elapsed_times) / 1e9)

        rm(filename)
        rm(filenamer)
    end

    println("Speedups through reordering:")
    for i in 1:length(ns)
        println("$(ns[i]): $(runtimes[i] ./ runtimesreordered[i])")
    end
    return
end
bench()
