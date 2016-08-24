using NearestNeighbors
using BenchmarkTools
using JLD

include("generate_report.jl")

const EXTENSIVE_BENCHMARK = false

const SUITE = BenchmarkGroup()
SUITE["build tree"] = BenchmarkGroup()
SUITE["knn"] = BenchmarkGroup()
SUITE["inrange"] = BenchmarkGroup()

for n_points in (EXTENSIVE_BENCHMARK ? (10^3, 10^5) : 10^5)
    for dim in (EXTENSIVE_BENCHMARK ? (1, 3) : 3)
        data = rand(MersenneTwister(1), dim, n_points)
        for leafsize in (EXTENSIVE_BENCHMARK ? (1, 10) : 10)
            for reorder in (true, false)
                for (tree_type, SUITE_name) in ((KDTree, "kd tree"),
                                                (BallTree, "ball tree"))
                    tree = tree_type(data; leafsize = leafsize, reorder = reorder)
                    SUITE["build tree"]["$(tree_type.name.name) $dim × $n_points, ls = $leafsize"] = @benchmarkable $(tree_type)($data; leafsize = $leafsize, reorder = $reorder)
                    for input_size in (1, 1000)
                        input_data = rand(MersenneTwister(1), dim, input_size)
                        for k in (EXTENSIVE_BENCHMARK ? (1, 10) : 10)
                            SUITE["knn"]["$(tree_type.name.name) $dim × $n_points, ls = $leafsize, input_size = $input_size, k = $k"] = @benchmarkable knn($tree, $input_data, $k)
                        end
                        perc = 0.01
                        V = π^(dim / 2) / gamma(dim / 2 + 1) * (1 / 2)^dim
                        r = (V * perc * gamma(dim / 2 + 1))^(1/dim)
                        r_formatted = @sprintf("%3.2e", r)
                        SUITE["inrange"]["$(tree_type.name.name) $dim × $n_points, ls = $leafsize, input_size = $input_size, r = $r_formatted"] = @benchmarkable inrange($tree, $input_data, $r)
                    end
                end
            end
        end
    end
end

function run_benchmarks(name)
    const paramspath = joinpath(dirname(@__FILE__), "params.jld")
    if !isfile(paramspath)
        println("Tuning benchmarks...")
        tune!(SUITE)
        JLD.save(paramspath, "SUITE", params(SUITE))
    end
    loadparams!(SUITE, JLD.load(paramspath, "SUITE"), :evals, :samples)
    results = run(SUITE, verbose = true, seconds = 2)
    JLD.save(joinpath(dirname(@__FILE__), name * ".jld"), "results", results)
end

function generate_report(v1, v2)
    v1_res = load(joinpath(dirname(@__FILE__), v1 * ".jld"), "results")
    v2_res = load(joinpath(dirname(@__FILE__), v2 * ".jld"), "results")
    open(joinpath(dirname(@__FILE__), "results_compare.md"), "w") do f
        printreport(f, judge(minimum(v1_res), minimum(v2_res)); iscomparisonjob = true)
    end
end

function generate_report(v1)
    v1_res = load(joinpath(dirname(@__FILE__), v1 * ".jld"), "results")
    open(joinpath(dirname(@__FILE__), "results_single.md"), "w") do f
        printreport(f, minimum(v1_res); iscomparisonjob = false)
    end
end

# run_benchmarks("primary")
# generate_report("primary") # generate a report with stats about a run
# run_benchmarks("secondary")
# generate_report("secondary", "primary") # generate report comparing two runs
