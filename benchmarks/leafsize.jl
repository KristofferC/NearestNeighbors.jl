# The purpose of these benchmark is to see the effect of using different leafsize.
# These benchmarks are replicates of what can be find at 
# https://github.com/jlblancoc/nanoflann#21-kdtreesingleindexadaptorparamsleaf_max_size

using NearestNeighbors 
using ProgressMeter
using Benchmarks
using Plots

function knnbench(tree, data, N, k)
    ind = rand(1:size(data,2), N)
    knn(tree, data[:,ind], k)
end

function bench_cloud(data, N, N_queries, dim, k, leafsizes)
    
    res_tree_builds = []
    res_knns = []
    res_tree_builds_reord = []
    res_knns_reord = []
    
    n = 2 * length(leafsizes)
    p = Progress(n, 1, "Benchmarking leafsize...", 50)
    for leafsize in leafsizes
        for reorder in [true, false]
            res_tree = @benchmark KDTree(data, leafsize = leafsize, reorder = reorder)
            tree = KDTree(data, leafsize = leafsize, reorder = reorder)
            res_knn = @benchmark knnbench(tree, data, N_queries, k)
            if reorder
                push!(res_tree_builds_reord, res_tree)
                push!(res_knns_reord, res_knn)
            else
                push!(res_tree_builds, res_tree)
                push!(res_knns, res_knn)
            end
            next!(p)
        end
    end
    return (res_tree_builds, res_knns, res_tree_builds_reord, res_knns_reord)
end


function visualize!(plot_handle, knn_benchs, build_benchs, N_queries, label)

    # TODO add C.I ellipsis
    avgs_knn = Float64[]
    avgs_build = Float64[]
    for (knn_bench, build_bench) in zip(knn_benchs, build_benchs)
        stats_knn = Benchmarks.SummaryStatistics(knn_bench)
        stats_build = Benchmarks.SummaryStatistics(build_bench)
        # In microseconds
        avg_knn = stats_knn.elapsed_time_center / 1e3 / N_queries
        upper_knn = get(stats_knn.elapsed_time_upper) / 1e3 / N_queries
        lower_knn = get(stats_knn.elapsed_time_lower) / 1e3 / N_queries

        # In milliseconds
        avg_build = stats_build.elapsed_time_center / 1e6 
        upper_build = get(stats_build.elapsed_time_upper) / 1e6
        lower_build = get(stats_build.elapsed_time_lower) / 1e6 

        push!(avgs_knn, avg_knn)
        push!(avgs_build, avg_build)
    end
    
    plot!(plot_handle, avgs_build, avgs_knn, m = :circle, label = label)
end

function do_leafsize_bench()
    N = 10^5
    N_queries = 10^4
    dim = 3
    k = 1
    data = rand(dim, N)
    leafsizes = [1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1000, 10000]
    res_tree_builds, res_knns, res_tree_builds_reord, res_knns_reord = 
        bench_cloud(data, N, N_queries, dim, k, leafsizes)
    p = plot(title = "Performance vs leafsize", xlabel = "Tree build time [ms]",
             ylabel= "Tree query time [micro s]")
    yaxis!(:log10)

    visualize!(p, res_knns, res_tree_builds, N_queries, "reorder")
    visualize!(p, res_knns_reord, res_tree_builds_reord, N_queries, "unordered")
end

do_leafsize_bench()
