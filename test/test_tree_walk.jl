using Test
using StaticArrays
using NearestNeighbors
using NearestNeighbors: children, parent, _treeindex, HyperRectangle, get_leaf_range, getparent, getleft
using AbstractTrees

@testset "Tree traversal" begin
    data = [rand(SVector{2, Float64}) for _ in 1:24]
    tree = BallTree(data; leafsize = 4)
    root = treeroot(tree)

    nodes = collect(PreOrderDFS(root))
    total_nodes = tree.tree_data.n_internal_nodes + tree.tree_data.n_leafs
    @test length(nodes) == total_nodes

    leaves = collect(Leaves(root))
    @test all(node -> isempty(children(node)), leaves)
    @test all(node -> _treeindex(node) > tree.tree_data.n_internal_nodes, leaves)

    first_leaf = first(leaves)
    pts = collect(leafpoints(first_leaf))
    @test !isempty(pts)
    @test all(x -> x isa SVector{2, Float64}, pts)

    idxs = collect(leaf_point_indices(first_leaf))
    @test length(idxs) == length(pts)
    @test all(∈(1:length(tree.data)), idxs)

    # Internal-only traversal (filter leaves)
    internals = [node for node in PreOrderDFS(root) if !isempty(children(node))]
    @test all(node -> !isempty(children(node)), internals)

    # Regions match stored hyperspheres
    @test treeregion(root) === tree.hyper_spheres[1]
end

@testset "Tree traversal with empty data" begin
    tree = KDTree(SVector{2, Float64}[])
    @test_throws ArgumentError treeroot(tree)
end

@testset "KDTree regions" begin
    data = [SVector(rand(), rand()) for _ in 1:16]
    tree = KDTree(data; leafsize = 2)

    root = treeroot(tree)
    @test treeregion(root) == tree.hyper_rec

    dim = Int(tree.split_dims[1])
    val = tree.split_vals[1]
    left_node, right_node = children(root)

    expected_left = HyperRectangle(
        tree.hyper_rec.mins,
        setindex(tree.hyper_rec.maxes, val, dim))
    expected_right = HyperRectangle(
        setindex(tree.hyper_rec.mins, val, dim),
        tree.hyper_rec.maxes)

    @test treeregion(left_node) == expected_left
    @test treeregion(right_node) == expected_right

    # Indices from reordered storage
    root = treeroot(tree)
    first_leaf = first(Leaves(root))
    @test leaf_point_indices(first_leaf; original = false) ==
          get_leaf_range(tree.tree_data, _treeindex(first_leaf))
end

@testset "Unsupported trees" begin
    data = [SVector(rand(), rand()) for _ in 1:4]
    brute = BruteTree(data)
    @test_throws ArgumentError treeroot(brute)
end

@testset "KDTree parent traversal" begin
    data = [SVector(rand(), rand()) for _ in 1:32]
    tree = KDTree(data; leafsize = 2)

    root = treeroot(tree)

    # Test that parent of root returns nothing
    @test parent(root) === nothing
    @test isroot(root)

    # Get children and verify parent reconstruction
    left_node, right_node = children(root)

    # Parent of left child should reconstruct root region
    parent_from_left = parent(left_node)
    @test _treeindex(parent_from_left) == 1
    @test treeregion(parent_from_left) == tree.hyper_rec

    # Parent of right child should also reconstruct root region
    parent_from_right = parent(right_node)
    @test _treeindex(parent_from_right) == 1
    @test treeregion(parent_from_right) == tree.hyper_rec

    # Test deeper in the tree
    if !isempty(children(left_node))
        left_left, left_right = children(left_node)
        parent_of_left_left = parent(left_left)
        @test _treeindex(parent_of_left_left) == _treeindex(left_node)
        @test treeregion(parent_of_left_left) == treeregion(left_node)

        parent_of_left_right = parent(left_right)
        @test _treeindex(parent_of_left_right) == _treeindex(left_node)
        @test treeregion(parent_of_left_right) == treeregion(left_node)
    end

    # Verify parent traversal works for all internal nodes
    for node in PreOrderDFS(root)
        if !isempty(children(node)) && _treeindex(node) != 1
            p = parent(node)
            @test _treeindex(p) == getparent(_treeindex(node))

            # Verify that the parent's region contains the child's region
            preg = treeregion(p)
            creg = treeregion(node)
            for i in 1:length(creg.mins)
                @test creg.mins[i] >= preg.mins[i]
                @test creg.maxes[i] <= preg.maxes[i]
            end
        end
    end
end

@testset "BallTree parent traversal" begin
    data = [SVector(rand(), rand()) for _ in 1:32]
    tree = BallTree(data; leafsize = 2)

    root = treeroot(tree)

    # Test that parent of root returns nothing
    @test parent(root) === nothing
    @test isroot(root)

    # Get children and verify parent lookup
    left_node, right_node = children(root)

    # Parent of left child should be root
    parent_from_left = parent(left_node)
    @test _treeindex(parent_from_left) == 1
    @test treeregion(parent_from_left) === tree.hyper_spheres[1]

    # Parent of right child should also be root
    parent_from_right = parent(right_node)
    @test _treeindex(parent_from_right) == 1
    @test treeregion(parent_from_right) === tree.hyper_spheres[1]

    # Verify parent traversal works for all internal nodes
    for node in PreOrderDFS(root)
        if !isempty(children(node)) && _treeindex(node) != 1
            p = parent(node)
            @test _treeindex(p) == getparent(_treeindex(node))
            @test treeregion(p) === tree.hyper_spheres[_treeindex(p)]
        end
    end
end

@testset "Traversal from subtree" begin
    data = [SVector(rand(), rand()) for _ in 1:64]
    tree = KDTree(data; leafsize = 2)

    root = treeroot(tree)
    left_child, right_child = children(root)

    # Walk from left subtree only
    left_nodes = collect(PreOrderDFS(left_child))
    left_indices = Set(_treeindex(n) for n in left_nodes)

    # Walk from entire tree
    all_nodes = collect(PreOrderDFS(root))
    all_indices = Set(_treeindex(n) for n in all_nodes)

    # Left subtree should be subset of all nodes
    @test left_indices ⊆ all_indices
    @test length(left_indices) < length(all_indices)

    # Root should not be in left subtree walk
    @test !(1 in left_indices)

    # All nodes in left walk should have indices >= getleft(1)
    for idx in left_indices
        # Should be in left subtree
        @test idx >= getleft(1)
    end

    # Test walking leaves from a subtree
    left_leaves = collect(Leaves(left_child))
    @test all(node -> isempty(children(node)), left_leaves)
    @test length(left_leaves) > 0
end
