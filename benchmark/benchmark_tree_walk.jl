using BenchmarkTools
using NearestNeighbors
using StaticArrays
using AbstractTrees
using NearestNeighbors: children, _treeindex, preorder, postorder, leaves

function make_trees(n_points, ndim=3)
    data = [SVector{ndim}(rand(ndim)...) for _ in 1:n_points]
    kdtree = KDTree(data; leafsize=10)
    balltree = BallTree(data; leafsize=10)
    return kdtree, balltree
end

# Benchmark: Count total nodes
function count_nodes_abstracttrees(tree)
    root = treeroot(tree)
    count = 0
    for _ in PreOrderDFS(root)
        count += 1
    end
    return count
end

function count_nodes_custom(tree)
    count = 0
    for _ in preorder(tree)
        count += 1
    end
    return count
end

# Benchmark: Sum of all leaf point norms
function sum_leaf_norms_abstracttrees(tree)
    root = treeroot(tree)
    total = 0.0
    for node in Leaves(root)
        for pt in leafpoints(node)
            total += sum(abs2, pt)
        end
    end
    return total
end

function sum_leaf_norms_custom(tree)
    total = 0.0
    for node in leaves(tree)
        for pt in leafpoints(node)
            total += sum(abs2, pt)
        end
    end
    return total
end

# Benchmark: Collect all node indices
function collect_indices_abstracttrees(tree)
    root = treeroot(tree)
    indices = Int[]
    for node in PreOrderDFS(root)
        push!(indices, _treeindex(node))
    end
    return indices
end

function collect_indices_custom(tree)
    indices = Int[]
    for node in preorder(tree)
        push!(indices, _treeindex(node))
    end
    return indices
end

function run_benchmarks()
    sizes = [1_000, 50_000]

    println("="^70)
    println("Tree Walking Benchmark: Custom Walkers vs AbstractTrees")
    println("="^70)

    for n in sizes
        println("\n" * "="^70)
        println("Tree size: $n points")
        println("="^70)

        kdtree, balltree = make_trees(n)

        # Verify both approaches give same results
        @assert count_nodes_abstracttrees(kdtree) == count_nodes_custom(kdtree)
        @assert sum_leaf_norms_abstracttrees(kdtree) ≈ sum_leaf_norms_custom(kdtree)
        @assert collect_indices_abstracttrees(kdtree) == collect_indices_custom(kdtree)

        println("\nCount nodes:")
        print("  AbstractTrees: ")
        @btime count_nodes_abstracttrees($kdtree)
        print("  Custom:        ")
        @btime count_nodes_custom($kdtree)

        println("\nSum leaf norms:")
        print("  AbstractTrees: ")
        @btime sum_leaf_norms_abstracttrees($kdtree)
        print("  Custom:        ")
        @btime sum_leaf_norms_custom($kdtree)

        println("\nCollect indices:")
        print("  AbstractTrees: ")
        @btime collect_indices_abstracttrees($kdtree)
        print("  Custom:        ")
        @btime collect_indices_custom($kdtree)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
