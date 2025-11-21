using Test
using StaticArrays
using NearestNeighbors
using NearestNeighbors: children, parent, _treeindex, HyperRectangle, get_leaf_range, getparent, getleft
using NearestNeighbors: treeroot_isbits, children_isbits, parent_isbits, treeregion_isbits, _isleaf_isbits
using NearestNeighbors: preorder_isbits, postorder_isbits, leaves_isbits, leafpoints_isbits, leaf_point_indices_isbits
using NearestNeighbors: preorder_custom, postorder_custom, leaves_custom
using NearestNeighbors: treeroot_indexnode, preorder_indexnode, leaves_indexnode, leafpoints_indexnode
using NearestNeighbors: leaf_point_indices_indexnode, treeregion_indexnode, _isleaf_indexnode
using AbstractTrees

@testset "Tree traversal - all implementations" begin
    data = [rand(SVector{2, Float64}) for _ in 1:24]
    tree = BallTree(data; leafsize = 4)

    # v1: TreeNode + AbstractTrees
    root_v1 = treeroot(tree)
    nodes_v1 = collect(PreOrderDFS(root_v1))
    leaves_v1 = collect(Leaves(root_v1))

    # v2: isbits + custom
    nodes_v2 = collect(preorder_isbits(tree))
    leaves_v2 = collect(leaves_isbits(tree))

    # v3: TreeNode + custom
    nodes_v3 = collect(preorder_custom(tree))
    leaves_v3 = collect(leaves_custom(tree))

    # v4: IndexNode + AbstractTrees
    nodes_v4 = collect(preorder_indexnode(tree))
    leaves_v4 = collect(leaves_indexnode(tree))

    # All should have same counts
    total_nodes = tree.tree_data.n_internal_nodes + tree.tree_data.n_leafs
    @test length(nodes_v1) == total_nodes
    @test length(nodes_v2) == total_nodes
    @test length(nodes_v3) == total_nodes
    @test length(nodes_v4) == total_nodes

    @test length(leaves_v1) == tree.tree_data.n_leafs
    @test length(leaves_v2) == tree.tree_data.n_leafs
    @test length(leaves_v3) == tree.tree_data.n_leafs
    @test length(leaves_v4) == tree.tree_data.n_leafs

    # Test leaf properties for v1
    @test all(node -> isempty(children(node)), leaves_v1)
    @test all(node -> _treeindex(node) > tree.tree_data.n_internal_nodes, leaves_v1)

    # Test leaf properties for v2
    @test all(node -> isempty(children_isbits(tree, node)), leaves_v2)
    @test all(node -> node.index > tree.tree_data.n_internal_nodes, leaves_v2)

    # Test leafpoints for v1
    first_leaf_v1 = first(leaves_v1)
    pts_v1 = collect(leafpoints(first_leaf_v1))
    @test !isempty(pts_v1)
    @test all(x -> x isa SVector{2, Float64}, pts_v1)

    # Test leafpoints for v2
    first_leaf_v2 = first(leaves_v2)
    pts_v2 = collect(leafpoints_isbits(tree, first_leaf_v2))
    @test !isempty(pts_v2)
    @test all(x -> x isa SVector{2, Float64}, pts_v2)

    # Test leafpoints for v4
    first_leaf_v4 = first(leaves_v4)
    pts_v4 = collect(leafpoints_indexnode(first_leaf_v4))
    @test !isempty(pts_v4)
    @test all(x -> x isa SVector{2, Float64}, pts_v4)

    # Test leaf point indices
    idxs_v1 = collect(leaf_point_indices(first_leaf_v1))
    idxs_v2 = collect(leaf_point_indices_isbits(tree, first_leaf_v2))
    idxs_v4 = collect(leaf_point_indices_indexnode(first_leaf_v4))
    @test length(idxs_v1) == length(pts_v1)
    @test length(idxs_v2) == length(pts_v2)
    @test length(idxs_v4) == length(pts_v4)

    # Regions match stored hyperspheres for v1
    @test treeregion(root_v1) === tree.hyper_spheres[1]

    # Regions match for v2
    root_v2 = treeroot_isbits(tree)
    @test treeregion_isbits(tree, root_v2) === tree.hyper_spheres[1]

    # Regions match for v4
    root_v4 = treeroot_indexnode(tree)
    @test treeregion_indexnode(root_v4) === tree.hyper_spheres[1]
end

@testset "Tree traversal with empty data" begin
    tree = KDTree(SVector{2, Float64}[])
    @test_throws ArgumentError treeroot(tree)
    @test_throws ArgumentError treeroot_isbits(tree)
    @test_throws ArgumentError treeroot_indexnode(tree)
end

@testset "KDTree regions - all implementations" begin
    data = [SVector(rand(), rand()) for _ in 1:16]
    tree = KDTree(data; leafsize = 2)

    # v1: TreeNode
    root_v1 = treeroot(tree)
    @test treeregion(root_v1) == tree.hyper_rec

    # v2: isbits
    root_v2 = treeroot_isbits(tree)
    @test treeregion_isbits(tree, root_v2) == tree.hyper_rec

    # v4: IndexNode
    root_v4 = treeroot_indexnode(tree)
    @test treeregion_indexnode(root_v4) == tree.hyper_rec

    # Test child regions for v1
    dim = Int(tree.split_dims[1])
    val = tree.split_vals[1]
    left_v1, right_v1 = children(root_v1)

    expected_left = HyperRectangle(
        tree.hyper_rec.mins,
        setindex(tree.hyper_rec.maxes, val, dim))
    expected_right = HyperRectangle(
        setindex(tree.hyper_rec.mins, val, dim),
        tree.hyper_rec.maxes)

    @test treeregion(left_v1) == expected_left
    @test treeregion(right_v1) == expected_right

    # Test child regions for v2
    left_v2, right_v2 = children_isbits(tree, root_v2)
    @test treeregion_isbits(tree, left_v2) == expected_left
    @test treeregion_isbits(tree, right_v2) == expected_right

    # Test leaf region reconstruction
    first_leaf_v2 = first(leaves_isbits(tree))
    leaf_region = treeregion_isbits(tree, first_leaf_v2)
    @test leaf_region isa HyperRectangle
    for i in eachindex(leaf_region.mins)
        @test leaf_region.mins[i] >= tree.hyper_rec.mins[i]
        @test leaf_region.maxes[i] <= tree.hyper_rec.maxes[i]
    end

    # Test indices from reordered storage
    first_leaf_v1 = first(Leaves(root_v1))
    @test leaf_point_indices(first_leaf_v1; original = false) ==
          get_leaf_range(tree.tree_data, _treeindex(first_leaf_v1))

    @test leaf_point_indices_isbits(tree, first_leaf_v2; original = false) ==
          get_leaf_range(tree.tree_data, first_leaf_v2.index)
end

@testset "Unsupported trees" begin
    data = [SVector(rand(), rand()) for _ in 1:4]
    brute = BruteTree(data)
    @test_throws ArgumentError treeroot(brute)
    @test_throws ArgumentError treeroot_isbits(brute)
end

@testset "KDTree parent traversal - all implementations" begin
    data = [SVector(rand(), rand()) for _ in 1:32]
    tree = KDTree(data; leafsize = 2)

    # v1: TreeNode
    root_v1 = treeroot(tree)
    @test parent(root_v1) === nothing
    @test isroot(root_v1)

    left_v1, right_v1 = children(root_v1)
    parent_from_left = parent(left_v1)
    @test _treeindex(parent_from_left) == 1
    @test treeregion(parent_from_left) == tree.hyper_rec

    # v2: isbits
    root_v2 = treeroot_isbits(tree)
    @test parent_isbits(tree, root_v2) === nothing

    left_v2, right_v2 = children_isbits(tree, root_v2)
    parent_from_left_v2 = parent_isbits(tree, left_v2)
    @test parent_from_left_v2.index == 1
    @test treeregion_isbits(tree, parent_from_left_v2) == tree.hyper_rec

    # Verify parent traversal for all internal nodes (v1)
    for node in PreOrderDFS(root_v1)
        if !isempty(children(node)) && _treeindex(node) != 1
            p = parent(node)
            @test _treeindex(p) == getparent(_treeindex(node))

            # Verify parent region contains child region
            preg = treeregion(p)
            creg = treeregion(node)
            for i in 1:length(creg.mins)
                @test creg.mins[i] >= preg.mins[i]
                @test creg.maxes[i] <= preg.maxes[i]
            end
        end
    end

    # Verify parent traversal for all internal nodes (v2)
    for node in preorder_isbits(tree)
        if !isempty(children_isbits(tree, node)) && node.index != 1
            p = parent_isbits(tree, node)
            @test p.index == getparent(node.index)

            preg = treeregion_isbits(tree, p)
            creg = treeregion_isbits(tree, node)
            for i in 1:length(creg.mins)
                @test creg.mins[i] >= preg.mins[i]
                @test creg.maxes[i] <= preg.maxes[i]
            end
        end
    end
end

@testset "BallTree parent traversal - all implementations" begin
    data = [SVector(rand(), rand()) for _ in 1:32]
    tree = BallTree(data; leafsize = 2)

    # v1: TreeNode
    root_v1 = treeroot(tree)
    @test parent(root_v1) === nothing

    left_v1, right_v1 = children(root_v1)
    parent_from_left = parent(left_v1)
    @test _treeindex(parent_from_left) == 1
    @test treeregion(parent_from_left) === tree.hyper_spheres[1]

    # v2: isbits
    root_v2 = treeroot_isbits(tree)
    @test parent_isbits(tree, root_v2) === nothing

    left_v2, right_v2 = children_isbits(tree, root_v2)
    parent_from_left_v2 = parent_isbits(tree, left_v2)
    @test parent_from_left_v2.index == 1
    @test treeregion_isbits(tree, parent_from_left_v2) === tree.hyper_spheres[1]

    # Verify parent traversal works for all internal nodes (v2)
    for node in preorder_isbits(tree)
        if !isempty(children_isbits(tree, node)) && node.index != 1
            p = parent_isbits(tree, node)
            @test p.index == getparent(node.index)
            @test treeregion_isbits(tree, p) === tree.hyper_spheres[p.index]
        end
    end
end

@testset "PostOrder traversal - isbits" begin
    data = [SVector(rand(), rand()) for _ in 1:32]
    tree = KDTree(data; leafsize = 2)

    postorder_nodes = collect(postorder_isbits(tree))
    preorder_nodes = collect(preorder_isbits(tree))

    @test length(postorder_nodes) == length(preorder_nodes)
    @test postorder_nodes[end].index == 1

    # All leaves should appear before their parents
    for node in postorder_nodes
        if !_isleaf_isbits(tree, node)
            left, right = children_isbits(tree, node)
            left_pos = findfirst(n -> n.index == left.index, postorder_nodes)
            right_pos = findfirst(n -> n.index == right.index, postorder_nodes)
            node_pos = findfirst(n -> n.index == node.index, postorder_nodes)
            @test left_pos < node_pos
            @test right_pos < node_pos
        end
    end
end

@testset "PostOrder traversal - custom" begin
    data = [SVector(rand(), rand()) for _ in 1:32]
    tree = KDTree(data; leafsize = 2)

    postorder_nodes = collect(postorder_custom(tree))
    preorder_nodes = collect(preorder_custom(tree))

    @test length(postorder_nodes) == length(preorder_nodes)
    @test _treeindex(postorder_nodes[end]) == 1

    # All leaves should appear before their parents
    for node in postorder_nodes
        if !isempty(children(node))
            left, right = children(node)
            left_pos = findfirst(n -> _treeindex(n) == _treeindex(left), postorder_nodes)
            right_pos = findfirst(n -> _treeindex(n) == _treeindex(right), postorder_nodes)
            node_pos = findfirst(n -> _treeindex(n) == _treeindex(node), postorder_nodes)
            @test left_pos < node_pos
            @test right_pos < node_pos
        end
    end
end

@testset "Iterator properties" begin
    data = [SVector(rand(), rand()) for _ in 1:64]
    tree = KDTree(data; leafsize = 4)

    expected_total = tree.tree_data.n_internal_nodes + tree.tree_data.n_leafs
    expected_leaves = tree.tree_data.n_leafs

    # v2: isbits
    preorder_walker = preorder_isbits(tree)
    @test length(preorder_walker) == expected_total
    @test Base.IteratorSize(typeof(preorder_walker)) == Base.HasLength()

    postorder_walker = postorder_isbits(tree)
    @test length(postorder_walker) == expected_total
    @test Base.IteratorSize(typeof(postorder_walker)) == Base.HasLength()

    leaf_walker = leaves_isbits(tree)
    @test length(leaf_walker) == expected_leaves
    @test Base.IteratorSize(typeof(leaf_walker)) == Base.HasLength()

    # v3: custom
    preorder_walker_v3 = preorder_custom(tree)
    @test length(preorder_walker_v3) == expected_total
    @test Base.IteratorSize(typeof(preorder_walker_v3)) == Base.HasLength()

    postorder_walker_v3 = postorder_custom(tree)
    @test length(postorder_walker_v3) == expected_total
    @test Base.IteratorSize(typeof(postorder_walker_v3)) == Base.HasLength()

    leaf_walker_v3 = leaves_custom(tree)
    @test length(leaf_walker_v3) == expected_leaves
    @test Base.IteratorSize(typeof(leaf_walker_v3)) == Base.HasLength()
end

@testset "Leaf region reconstruction" begin
    data = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(1.0, 1.0)]
    tree = KDTree(data; leafsize = 1)

    # All leaves should have valid regions (v2)
    for leaf in leaves_isbits(tree)
        region = treeregion_isbits(tree, leaf)
        @test region isa HyperRectangle
        for i in eachindex(region.mins)
            @test region.mins[i] <= region.maxes[i]
        end
    end
end

@testset "Traversal from subtree" begin
    data = [SVector(rand(), rand()) for _ in 1:64]
    tree = KDTree(data; leafsize = 2)

    # v1: TreeNode
    root = treeroot(tree)
    left_child, right_child = children(root)

    left_nodes = collect(PreOrderDFS(left_child))
    left_indices = Set(_treeindex(n) for n in left_nodes)

    all_nodes = collect(PreOrderDFS(root))
    all_indices = Set(_treeindex(n) for n in all_nodes)

    @test left_indices ⊆ all_indices
    @test length(left_indices) < length(all_indices)
    @test !(1 in left_indices)

    for idx in left_indices
        @test idx >= getleft(1)
    end

    left_leaves = collect(Leaves(left_child))
    @test all(node -> isempty(children(node)), left_leaves)
    @test length(left_leaves) > 0
end
