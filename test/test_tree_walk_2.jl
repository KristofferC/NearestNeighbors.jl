using Test
using StaticArrays
using NearestNeighbors
using NearestNeighbors: treeroot2, children2, parent2, treeregion2, _isleaf2
using NearestNeighbors: preorder2, postorder2, leaves2, leafpoints2, leaf_point_indices2
using NearestNeighbors: HyperRectangle, get_leaf_range, getparent, getleft

@testset "Tree traversal (v2)" begin
    data = [rand(SVector{2, Float64}) for _ in 1:24]
    tree = BallTree(data; leafsize = 4)
    root = treeroot2(tree)

    nodes = collect(preorder2(tree))
    total_nodes = tree.tree_data.n_internal_nodes + tree.tree_data.n_leafs
    @test length(nodes) == total_nodes

    leaves = collect(leaves2(tree))
    @test all(node -> isempty(children2(tree, node)), leaves)
    @test all(node -> node.index > tree.tree_data.n_internal_nodes, leaves)

    first_leaf = first(leaves)
    pts = collect(leafpoints2(tree, first_leaf))
    @test !isempty(pts)
    @test all(x -> x isa SVector{2, Float64}, pts)

    idxs = collect(leaf_point_indices2(tree, first_leaf))
    @test length(idxs) == length(pts)
    @test all(∈(1:length(tree.data)), idxs)

    # Internal-only traversal (filter leaves)
    internals = [node for node in preorder2(tree) if !isempty(children2(tree, node))]
    @test all(node -> !isempty(children2(tree, node)), internals)

    # Regions match stored hyperspheres
    @test treeregion2(tree, root) === tree.hyper_spheres[1]
end

@testset "Tree traversal with empty data (v2)" begin
    tree = KDTree(SVector{2, Float64}[])
    @test_throws ArgumentError treeroot2(tree)
end

@testset "KDTree regions (v2)" begin
    data = [SVector(rand(), rand()) for _ in 1:16]
    tree = KDTree(data; leafsize = 2)

    root = treeroot2(tree)
    @test treeregion2(tree, root) == tree.hyper_rec

    dim = Int(tree.split_dims[1])
    val = tree.split_vals[1]
    left_node, right_node = children2(tree, root)

    expected_left = HyperRectangle(
        tree.hyper_rec.mins,
        setindex(tree.hyper_rec.maxes, val, dim))
    expected_right = HyperRectangle(
        setindex(tree.hyper_rec.mins, val, dim),
        tree.hyper_rec.maxes)

    @test treeregion2(tree, left_node) == expected_left
    @test treeregion2(tree, right_node) == expected_right

    # Test that leaf regions can be reconstructed
    first_leaf = first(leaves2(tree))
    leaf_region = treeregion2(tree, first_leaf)
    @test leaf_region isa HyperRectangle
    # Leaf region should be contained within tree bounds
    for i in eachindex(leaf_region.mins)
        @test leaf_region.mins[i] >= tree.hyper_rec.mins[i]
        @test leaf_region.maxes[i] <= tree.hyper_rec.maxes[i]
    end

    # Indices from reordered storage
    @test leaf_point_indices2(tree, first_leaf; original = false) ==
          get_leaf_range(tree.tree_data, first_leaf.index)
end

@testset "Unsupported trees (v2)" begin
    data = [SVector(rand(), rand()) for _ in 1:4]
    brute = BruteTree(data)
    @test_throws ArgumentError treeroot2(brute)
end

@testset "KDTree parent traversal (v2)" begin
    data = [SVector(rand(), rand()) for _ in 1:32]
    tree = KDTree(data; leafsize = 2)

    root = treeroot2(tree)

    # Test that parent of root returns nothing
    @test parent2(tree, root) === nothing

    # Get children and verify parent reconstruction
    left_node, right_node = children2(tree, root)

    # Parent of left child should reconstruct root region
    parent_from_left = parent2(tree, left_node)
    @test parent_from_left.index == 1
    @test treeregion2(tree, parent_from_left) == tree.hyper_rec

    # Parent of right child should also reconstruct root region
    parent_from_right = parent2(tree, right_node)
    @test parent_from_right.index == 1
    @test treeregion2(tree, parent_from_right) == tree.hyper_rec

    # Test deeper in the tree
    if !isempty(children2(tree, left_node))
        left_left, left_right = children2(tree, left_node)
        parent_of_left_left = parent2(tree, left_left)
        @test parent_of_left_left.index == left_node.index
        @test treeregion2(tree, parent_of_left_left) == treeregion2(tree, left_node)

        parent_of_left_right = parent2(tree, left_right)
        @test parent_of_left_right.index == left_node.index
        @test treeregion2(tree, parent_of_left_right) == treeregion2(tree, left_node)
    end

    # Verify parent traversal works for all internal nodes
    for node in preorder2(tree)
        if !isempty(children2(tree, node)) && node.index != 1
            p = parent2(tree, node)
            @test p.index == getparent(node.index)

            # Verify that the parent's region contains the child's region
            preg = treeregion2(tree, p)
            creg = treeregion2(tree, node)
            for i in 1:length(creg.mins)
                @test creg.mins[i] >= preg.mins[i]
                @test creg.maxes[i] <= preg.maxes[i]
            end
        end
    end
end

@testset "BallTree parent traversal (v2)" begin
    data = [SVector(rand(), rand()) for _ in 1:32]
    tree = BallTree(data; leafsize = 2)

    root = treeroot2(tree)

    # Test that parent of root returns nothing
    @test parent2(tree, root) === nothing

    # Get children and verify parent lookup
    left_node, right_node = children2(tree, root)

    # Parent of left child should be root
    parent_from_left = parent2(tree, left_node)
    @test parent_from_left.index == 1
    @test treeregion2(tree, parent_from_left) === tree.hyper_spheres[1]

    # Parent of right child should also be root
    parent_from_right = parent2(tree, right_node)
    @test parent_from_right.index == 1
    @test treeregion2(tree, parent_from_right) === tree.hyper_spheres[1]

    # Verify parent traversal works for all internal nodes
    for node in preorder2(tree)
        if !isempty(children2(tree, node)) && node.index != 1
            p = parent2(tree, node)
            @test p.index == getparent(node.index)
            @test treeregion2(tree, p) === tree.hyper_spheres[p.index]
        end
    end
end

@testset "PostOrder traversal (v2)" begin
    data = [SVector(rand(), rand()) for _ in 1:32]
    tree = KDTree(data; leafsize = 2)

    # Collect nodes in post-order
    postorder_nodes = collect(postorder2(tree))
    preorder_nodes = collect(preorder2(tree))

    # Should visit same number of nodes
    @test length(postorder_nodes) == length(preorder_nodes)

    # Root should be last in post-order
    @test postorder_nodes[end].index == 1

    # All leaves should appear before their parents
    for node in postorder_nodes
        if !_isleaf2(tree, node)
            left, right = children2(tree, node)
            left_pos = findfirst(n -> n.index == left.index, postorder_nodes)
            right_pos = findfirst(n -> n.index == right.index, postorder_nodes)
            node_pos = findfirst(n -> n.index == node.index, postorder_nodes)
            @test left_pos < node_pos
            @test right_pos < node_pos
        end
    end
end

@testset "Iterator properties (v2)" begin
    data = [SVector(rand(), rand()) for _ in 1:64]
    tree = KDTree(data; leafsize = 4)

    # Test that iterators have correct length
    expected_total = tree.tree_data.n_internal_nodes + tree.tree_data.n_leafs
    expected_leaves = tree.tree_data.n_leafs

    preorder_walker = preorder2(tree)
    @test length(preorder_walker) == expected_total
    @test Base.IteratorSize(typeof(preorder_walker)) == Base.HasLength()

    postorder_walker = postorder2(tree)
    @test length(postorder_walker) == expected_total
    @test Base.IteratorSize(typeof(postorder_walker)) == Base.HasLength()

    leaf_walker = leaves2(tree)
    @test length(leaf_walker) == expected_leaves
    @test Base.IteratorSize(typeof(leaf_walker)) == Base.HasLength()
end

@testset "Leaf region reconstruction (v2)" begin
    # Test with small tree to verify leaf regions
    data = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(1.0, 1.0)]
    tree = KDTree(data; leafsize = 1)

    # All leaves should have valid regions
    for leaf in leaves2(tree)
        region = treeregion2(tree, leaf)
        @test region isa HyperRectangle
        # Region should not be degenerate
        for i in eachindex(region.mins)
            @test region.mins[i] <= region.maxes[i]
        end
    end
end
