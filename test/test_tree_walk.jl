module TestTreeWalk
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "TestSetup.jl"))

using Test
using StaticArrays
using NearestNeighbors
using NearestNeighbors: children, parent, _treeindex, HyperRectangle, get_leaf_range, getparent, getleft
using NearestNeighbors: preorder, postorder, leaves
using AbstractTrees

@testset "Tree traversal" begin
    data = [rand(SVector{2, Float64}) for _ in 1:24]
    tree = BallTree(data; leafsize = 4)

    root = treeroot(tree)

    # AbstractTrees traversal
    nodes_at = collect(PreOrderDFS(root))
    leaves_at = collect(Leaves(root))

    # Custom walkers
    nodes_custom = collect(preorder(tree))
    leaves_custom = collect(leaves(tree))

    # All should have same counts
    total_nodes = tree.tree_data.n_internal_nodes + tree.tree_data.n_leafs
    @test length(nodes_at) == total_nodes
    @test length(nodes_custom) == total_nodes

    @test length(leaves_at) == tree.tree_data.n_leafs
    @test length(leaves_custom) == tree.tree_data.n_leafs

    # Test leaf properties
    @test all(node -> isempty(children(node)), leaves_at)
    @test all(node -> _treeindex(node) > tree.tree_data.n_internal_nodes, leaves_at)

    # Test leafpoints
    first_leaf = first(leaves_at)
    pts = collect(leafpoints(first_leaf))
    @test !isempty(pts)
    @test all(x -> x isa SVector{2, Float64}, pts)

    # Test leaf point indices
    expected_idx_view = view(tree.indices, get_leaf_range(tree.tree_data, _treeindex(first_leaf)))
    idxs = collect(leaf_point_indices(first_leaf))
    @test length(idxs) == length(pts)
    @test leaf_point_indices(first_leaf) == expected_idx_view

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

    # Test child regions
    dim = Int(tree.split_dims[1])
    val = tree.split_vals[1]
    left, right = children(root)

    expected_left = HyperRectangle(
        tree.hyper_rec.mins,
        setindex(tree.hyper_rec.maxes, val, dim))
    expected_right = HyperRectangle(
        setindex(tree.hyper_rec.mins, val, dim),
        tree.hyper_rec.maxes)

    @test treeregion(left) == expected_left
    @test treeregion(right) == expected_right

    # Test leaf region reconstruction
    first_leaf = first(leaves(tree))
    leaf_region = treeregion(first_leaf)
    @test leaf_region isa HyperRectangle
    for i in eachindex(leaf_region.mins)
        @test leaf_region.mins[i] >= tree.hyper_rec.mins[i]
        @test leaf_region.maxes[i] <= tree.hyper_rec.maxes[i]
    end

    first_leaf_at = first(Leaves(root))
    expected_idx_view = view(tree.indices, get_leaf_range(tree.tree_data, _treeindex(first_leaf_at)))
    @test leaf_point_indices(first_leaf_at) == expected_idx_view
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
    @test parent(root) === nothing
    @test isroot(root)

    left, right = children(root)
    parent_from_left = parent(left)
    @test _treeindex(parent_from_left) == 1
    @test treeregion(parent_from_left) == tree.hyper_rec

    # Verify parent traversal for all internal nodes
    for node in PreOrderDFS(root)
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
end

@testset "BallTree parent traversal" begin
    data = [SVector(rand(), rand()) for _ in 1:32]
    tree = BallTree(data; leafsize = 2)

    root = treeroot(tree)
    @test parent(root) === nothing

    left, right = children(root)
    parent_from_left = parent(left)
    @test _treeindex(parent_from_left) == 1
    @test treeregion(parent_from_left) === tree.hyper_spheres[1]

    # Verify parent traversal works for all internal nodes
    for node in preorder(tree)
        if !isempty(children(node)) && _treeindex(node) != 1
            p = parent(node)
            @test _treeindex(p) == getparent(_treeindex(node))
            @test treeregion(p) === tree.hyper_spheres[_treeindex(p)]
        end
    end
end

@testset "PostOrder traversal" begin
    data = [SVector(rand(), rand()) for _ in 1:32]
    tree = KDTree(data; leafsize = 2)

    postorder_nodes = collect(postorder(tree))
    preorder_nodes = collect(preorder(tree))

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

    preorder_walker = preorder(tree)
    @test length(preorder_walker) == expected_total
    @test Base.IteratorSize(typeof(preorder_walker)) == Base.HasLength()

    postorder_walker = postorder(tree)
    @test length(postorder_walker) == expected_total
    @test Base.IteratorSize(typeof(postorder_walker)) == Base.HasLength()

    leaf_walker = leaves(tree)
    @test length(leaf_walker) == expected_leaves
    @test Base.IteratorSize(typeof(leaf_walker)) == Base.HasLength()
end

@testset "Leaf region reconstruction" begin
    data = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(1.0, 1.0)]
    tree = KDTree(data; leafsize = 1)

    # All leaves should have valid regions
    for leaf in leaves(tree)
        region = treeregion(leaf)
        @test region isa HyperRectangle
        for i in eachindex(region.mins)
            @test region.mins[i] <= region.maxes[i]
        end
    end
end

@testset "Traversal from subtree" begin
    data = [SVector(rand(), rand()) for _ in 1:64]
    tree = KDTree(data; leafsize = 2)

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

end # module
