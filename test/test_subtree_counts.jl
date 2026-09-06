module TestSubtreeCounts
using NearestNeighbors, StaticArrays, Test
using NearestNeighbors: TreeData, subtree_npoints, get_leaf_range

function count_leaves(td, index)
    if index > td.n_internal_nodes
        return length(get_leaf_range(td, index))
    end
    return count_leaves(td, 2index) + count_leaves(td, 2index + 1)
end

@testset "Subtree populations" begin
    @test subtree_npoints(TreeData(SVector{1,Float64}[], 2), 1) == 0
    for n in 1:100, leafsize in 1:16
        td = TreeData(fill(SVector(0.0), n), leafsize)
        for index in 1:td.last_full_node
            @test subtree_npoints(td, index) == count_leaves(td, index)
        end
    end
end

@testset "Enclosed counts preserve skipping and periodic deduplication" begin
    for Tree in (KDTree, BallTree), leafsize in (1, 3, 8), reorder in (false, true)
        data = reshape(collect(0.0:0.1:0.9), 1, :)
        tree = Tree(data; leafsize, reorder)
        @test inrangecount(tree, [0.5], 10.0) == 10
        @test inrangecount(tree, [0.5], 10.0, iseven) == 5
        @test inrangecount(tree, [0.5], 10.0, Returns(true)) == 0
        periodic = PeriodicTree(tree, [0.0], [1.0])
        @test inrangecount(periodic, [0.5], 10.0) == 10
        @test inrangecount(periodic, [0.5], 10.0, iseven) == 5
        for radius in (0.05, 0.25, 0.55)
            @test inrangecount(tree, [0.52], radius) == inrangecount(BruteTree(data), [0.52], radius)
        end
    end
end
end
