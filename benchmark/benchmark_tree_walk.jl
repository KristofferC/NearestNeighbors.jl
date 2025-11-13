using BenchmarkTools
using NearestNeighbors
using StaticArrays
using AbstractTrees
using NearestNeighbors: children, _treeindex, children2, preorder2, leaves2, _isleaf2

# Create test trees of various sizes
function make_trees(n_points, ndim=3)
    data = [SVector{ndim}(rand(ndim)...) for _ in 1:n_points]
    kdtree = KDTree(data; leafsize=10)
    balltree = BallTree(data; leafsize=10)
    return kdtree, balltree
end

# Benchmark 1: Count total nodes
function count_nodes_v1(tree)
    root = treeroot(tree)
    count = 0
    for _ in PreOrderDFS(root)
        count += 1
    end
    return count
end

function count_nodes_v2(tree)
    count = 0
    for _ in preorder2(tree)
        count += 1
    end
    return count
end

# Benchmark 3: Sum of all leaf point norms
function sum_leaf_norms_v1(tree)
    root = treeroot(tree)
    total = 0.0
    for node in Leaves(root)
        for pt in leafpoints(node)
            total += sum(abs2, pt)
        end
    end
    return total
end

function sum_leaf_norms_v2(tree)
    total = 0.0
    for node in leaves2(tree)
        for pt in NearestNeighbors.leafpoints2(tree, node)
            total += sum(abs2, pt)
        end
    end
    return total
end

# Benchmark 4: Collect all node indices
function collect_indices_v1(tree)
    root = treeroot(tree)
    indices = Int[]
    for node in PreOrderDFS(root)
        push!(indices, _treeindex(node))
    end
    return indices
end

function collect_indices_v2(tree)
    indices = Int[]
    for node in preorder2(tree)
        push!(indices, node.index)
    end
    return indices
end

# Benchmark 5: Filter internal nodes
function count_internal_nodes_v1(tree)
    root = treeroot(tree)
    count = 0
    for node in PreOrderDFS(root)
        if !isempty(children(node))
            count += 1
        end
    end
    return count
end

function count_internal_nodes_v2(tree)
    count = 0
    for node in preorder2(tree)
        if !_isleaf2(tree, node)
            count += 1
        end
    end
    return count
end

# Run benchmarks
function run_benchmarks()
    sizes = [1_000, 50_000]

    println("="^80)
    println("Tree Walking Benchmark Comparison")
    println("AbstractTrees (v1) vs isbits nodes (v2)")
    println("="^80)

    for n in sizes
        println("\n" * "="^80)
        println("Tree size: $n points")
        println("="^80)

        kdtree, balltree = make_trees(n)

        # Verify both approaches give same results
        @assert count_nodes_v1(kdtree) == count_nodes_v2(kdtree)
        @assert sum_leaf_norms_v1(kdtree) ≈ sum_leaf_norms_v2(kdtree)
        @assert collect_indices_v1(kdtree) == collect_indices_v2(kdtree)
        @assert count_internal_nodes_v1(kdtree) == count_internal_nodes_v2(kdtree)

        println("\n--- KDTree ---")

        print("Count nodes (v1): ")
        @btime count_nodes_v1($kdtree)
        print("Count nodes (v2): ")
        @btime count_nodes_v2($kdtree)

        print("\nSum leaf norms (v1): ")
        @btime sum_leaf_norms_v1($kdtree)
        print("Sum leaf norms (v2): ")
        @btime sum_leaf_norms_v2($kdtree)

        print("\nCollect indices (v1): ")
        @btime collect_indices_v1($kdtree)
        print("Collect indices (v2): ")
        @btime collect_indices_v2($kdtree)

        print("\nCount internal (v1): ")
        @btime count_internal_nodes_v1($kdtree)
        print("Count internal (v2): ")
        @btime count_internal_nodes_v2($kdtree)

        println("\n--- BallTree ---")

        print("Count nodes (v1): ")
        @btime count_nodes_v1($balltree)
        print("Count nodes (v2): ")
        @btime count_nodes_v2($balltree)

        print("\nSum leaf norms (v1): ")
        @btime sum_leaf_norms_v1($balltree)
        print("Sum leaf norms (v2): ")
        @btime sum_leaf_norms_v2($balltree)

        print("\nCollect indices (v1): ")
        @btime collect_indices_v1($balltree)
        print("Collect indices (v2): ")
        @btime collect_indices_v2($balltree)

        print("\nCount internal (v1): ")
        @btime count_internal_nodes_v1($balltree)
        print("Count internal (v2): ")
        @btime count_internal_nodes_v2($balltree)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
