using Test
using NearestNeighbors: TreeData, get_leaf_range

@testset "TreeData leaf coverage" begin
    leafsizes = (1, 2, 3, 4, 7, 16, 25)
    for leafsize in leafsizes, n_p in 0:128
        data = fill(0.0, n_p)
        td = TreeData(data, leafsize)
        if td.n_leafs == 0
            @test n_p == 0
            continue
        end
        first_leaf = td.n_internal_nodes + 1
        last_leaf = td.n_internal_nodes + td.n_leafs
        covered = Int[]
        for idx in first_leaf:last_leaf
            append!(covered, get_leaf_range(td, idx))
        end
        sort!(covered)
        @test covered == collect(1:n_p)
    end
end
