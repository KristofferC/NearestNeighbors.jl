using NearestNeighbors
using BenchmarkTools
using StableRNGs
import Printf: @sprintf
import SpecialFunctions: gamma

const EXTENSIVE_BENCHMARK = false

const SUITE = BenchmarkGroup()
SUITE["build tree"] = BenchmarkGroup()
SUITE["knn"] = BenchmarkGroup()
SUITE["inrange"] = BenchmarkGroup()

for n_points in (EXTENSIVE_BENCHMARK ? (10^3, 10^5) : 10^5)
    for dim in (EXTENSIVE_BENCHMARK ? (1, 3) : 3)
        data = rand(StableRNG(123), dim, n_points)
        for leafsize in (EXTENSIVE_BENCHMARK ? (1, 10) : 10)
            for reorder in (true, false)
                for (tree_type, SUITE_name) in ((KDTree, "kd tree"),
                                                (BallTree, "ball tree"))
                    tree = tree_type(data; leafsize = leafsize, reorder = reorder)
                    SUITE["build tree"]["$(tree_type) $dim × $n_points, ls = $leafsize"] = @benchmarkable $(tree_type)($data; leafsize = $leafsize, reorder = $reorder)

                    # Add periodic tree benchmarks
                    bounds_min = zeros(dim)
                    bounds_max = ones(dim)  # Unit cube [0,1]^dim
                    ptree = PeriodicTree(tree, bounds_min, bounds_max)
                    SUITE["build tree"]["Periodic$(tree_type) $dim × $n_points, ls = $leafsize"] = @benchmarkable PeriodicTree($tree, $bounds_min, $bounds_max)

                    # Add mixed periodic/non-periodic benchmarks (only for multi-dimensional cases)
                    if dim > 1
                        bounds_max_mixed = [ones(dim-1); Inf]  # Last dimension non-periodic
                        ptree_mixed = PeriodicTree(tree, bounds_min, bounds_max_mixed)
                        SUITE["build tree"]["PeriodicMixed$(tree_type) $dim × $n_points, ls = $leafsize"] = @benchmarkable PeriodicTree($tree, $bounds_min, $bounds_max_mixed)
                    end

                    for input_size in (1, 1000)
                        input_data = rand(StableRNG(123), dim, input_size)
                        for k in (EXTENSIVE_BENCHMARK ? (1, 10) : 10)
                            SUITE["knn"]["$(tree_type) $dim × $n_points, ls = $leafsize, input_size = $input_size, k = $k"] = @benchmarkable knn($tree, $input_data, $k)
                            SUITE["knn"]["Periodic$(tree_type) $dim × $n_points, ls = $leafsize, input_size = $input_size, k = $k"] = @benchmarkable knn($ptree, $input_data, $k)
                            if dim > 1
                                SUITE["knn"]["PeriodicMixed$(tree_type) $dim × $n_points, ls = $leafsize, input_size = $input_size, k = $k"] = @benchmarkable knn($ptree_mixed, $input_data, $k)
                            end
                        end
                        perc = 0.01
                        V = π^(dim / 2) / gamma(dim / 2 + 1) * (1 / 2)^dim
                        r = (V * perc * gamma(dim / 2 + 1))^(1/dim)
                        r_formatted = @sprintf("%3.2e", r)
                        SUITE["inrange"]["$(tree_type) $dim × $n_points, ls = $leafsize, input_size = $input_size, r = $r_formatted"] = @benchmarkable inrange($tree, $input_data, $r)
                        SUITE["inrange"]["Periodic$(tree_type) $dim × $n_points, ls = $leafsize, input_size = $input_size, r = $r_formatted"] = @benchmarkable inrange($ptree, $input_data, $r)
                        if dim > 1
                            SUITE["inrange"]["PeriodicMixed$(tree_type) $dim × $n_points, ls = $leafsize, input_size = $input_size, r = $r_formatted"] = @benchmarkable inrange($ptree_mixed, $input_data, $r)
                        end
                    end
                end
            end
        end
    end
end
