using BenchmarkTools
using NearestNeighbors
using StaticArrays
using AbstractTrees
using NearestNeighbors: children, _treeindex,
    preorder_isbits, leaves_isbits, leafpoints_isbits,
    preorder_custom, leaves_custom,
    treeroot_indexnode, preorder_indexnode, leaves_indexnode, leafpoints_indexnode, _treeindex_indexnode

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
    for _ in preorder_isbits(tree)
        count += 1
    end
    return count
end

function count_nodes_v3(tree)
    count = 0
    for _ in preorder_custom(tree)
        count += 1
    end
    return count
end

function count_nodes_v4(tree)
    count = 0
    for _ in preorder_indexnode(tree)
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
    for node in leaves_isbits(tree)
        for pt in leafpoints_isbits(tree, node)
            total += sum(abs2, pt)
        end
    end
    return total
end

function sum_leaf_norms_v3(tree)
    total = 0.0
    for node in leaves_custom(tree)
        for pt in leafpoints(node)
            total += sum(abs2, pt)
        end
    end
    return total
end

function sum_leaf_norms_v4(tree)
    total = 0.0
    for node in leaves_indexnode(tree)
        for pt in leafpoints_indexnode(node)
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
    for node in preorder_isbits(tree)
        push!(indices, node.index)
    end
    return indices
end

function collect_indices_v3(tree)
    indices = Int[]
    for node in preorder_custom(tree)
        push!(indices, _treeindex(node))
    end
    return indices
end

function collect_indices_v4(tree)
    indices = Int[]
    for node in preorder_indexnode(tree)
        push!(indices, _treeindex_indexnode(node))
    end
    return indices
end

# Run benchmarks
function run_benchmarks()
    sizes = [1_000, 50_000]

    println("="^80)
    println("Tree Walking Benchmark Comparison")
    println("v1: TreeNode + AbstractTrees | v2: isbits + custom | v3: TreeNode + custom | v4: IndexNode + AbstractTrees")
    println("="^80)

    for n in sizes
        println("\n" * "="^80)
        println("Tree size: $n points")
        println("="^80)

        kdtree, balltree = make_trees(n)

        # Verify all approaches give same results
        @assert count_nodes_v1(kdtree) == count_nodes_v2(kdtree) == count_nodes_v3(kdtree) == count_nodes_v4(kdtree)
        @assert sum_leaf_norms_v1(kdtree) ≈ sum_leaf_norms_v2(kdtree) ≈ sum_leaf_norms_v3(kdtree) ≈ sum_leaf_norms_v4(kdtree)
        @assert collect_indices_v1(kdtree) == collect_indices_v2(kdtree) == collect_indices_v3(kdtree) == collect_indices_v4(kdtree)

        print("Count nodes (v1): ")
        @btime count_nodes_v1($kdtree)
        print("Count nodes (v2): ")
        @btime count_nodes_v2($kdtree)
        print("Count nodes (v3): ")
        @btime count_nodes_v3($kdtree)
        print("Count nodes (v4): ")
        @btime count_nodes_v4($kdtree)

        print("\nSum leaf norms (v1): ")
        @btime sum_leaf_norms_v1($kdtree)
        print("Sum leaf norms (v2): ")
        @btime sum_leaf_norms_v2($kdtree)
        print("Sum leaf norms (v3): ")
        @btime sum_leaf_norms_v3($kdtree)
        print("Sum leaf norms (v4): ")
        @btime sum_leaf_norms_v4($kdtree)

        print("\nCollect indices (v1): ")
        @btime collect_indices_v1($kdtree)
        print("Collect indices (v2): ")
        @btime collect_indices_v2($kdtree)
        print("Collect indices (v3): ")
        @btime collect_indices_v3($kdtree)
        print("Collect indices (v4): ")
        @btime collect_indices_v4($kdtree)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
