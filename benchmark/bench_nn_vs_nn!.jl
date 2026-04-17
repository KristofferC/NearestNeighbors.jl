"""
Benchmark: nn! (in-place, scratch buffers) vs nn (allocating, old path via _firsteach).

The concern is that the scratch-buffer loop iteration in nn! is slower despite
fewer allocations than the old knn→_firsteach path.
"""

using NearestNeighbors, BenchmarkTools, StaticArrays, Random

# ── old nn path (pre-PR) ────────────────────────────────────────────────────
# Extracted verbatim so we can benchmark the original path without reverting.
_old_firsteach(v::Tuple) = first.(first(v)), first.(last(v))
function old_nn(tree, points, skip=NearestNeighbors.always_false)
    _old_firsteach(knn(tree, points, 1, false, skip))
end

# ── setup ───────────────────────────────────────────────────────────────────
rng = MersenneTwister(42)
const DIM = 3
const N_TREE = 100_000
const N_QUERY = 1_000

data  = rand(rng, DIM, N_TREE)
query_mat = rand(rng, DIM, N_QUERY)
query_svecs = [SVector{DIM,Float64}(query_mat[:, i]) for i in 1:N_QUERY]

tree_kd   = KDTree(data)
tree_ball = BallTree(data)

# Pre-allocate output buffers for nn!
idxs_out  = Vector{Int}(undef, N_QUERY)
dists_out = Vector{Float64}(undef, N_QUERY)

# ── correctness check ────────────────────────────────────────────────────────
let i1, d1, i2, d2
    i1, d1 = old_nn(tree_kd, query_svecs)
    i2, d2 = nn(tree_kd, query_svecs)
    @assert i1 == i2 && d1 ≈ d2 "nn results differ from old_nn!"
    nn!(tree_kd, query_svecs, idxs_out, dists_out)
    @assert i1 == idxs_out && d1 ≈ dists_out "nn! results differ from old_nn!"
    println("Correctness check passed.")
end

println()
println("=" ^ 60)
println("KDTree, $(DIM)D, tree=$N_TREE pts, query=$N_QUERY SVecs")
println("=" ^ 60)

println("\n--- old nn (knn→_firsteach, allocates $N_QUERY inner vecs) ---")
@btime old_nn($tree_kd, $query_svecs)

println("\n--- new nn  (calls nn! + 2 output allocs) ---")
@btime nn($tree_kd, $query_svecs)

println("\n--- nn!  (caller owns output, 2 scratch allocs total) ---")
@btime nn!($tree_kd, $query_svecs, $idxs_out, $dists_out)

println()
println("=" ^ 60)
println("BallTree, $(DIM)D, tree=$N_TREE pts, query=$N_QUERY SVecs")
println("=" ^ 60)

println("\n--- old nn ---")
@btime old_nn($tree_ball, $query_svecs)

println("\n--- new nn ---")
@btime nn($tree_ball, $query_svecs)

println("\n--- nn! ---")
@btime nn!($tree_ball, $query_svecs, $idxs_out, $dists_out)

# ── single-point form ────────────────────────────────────────────────────────
println()
println("=" ^ 60)
println("Single point (KDTree)")
println("=" ^ 60)
q = query_svecs[1]
idx1  = Vector{Int}(undef, 1)
dist1 = Vector{Float64}(undef, 1)

println("\n--- nn (single point, allocates 2 vecs) ---")
@btime nn($tree_kd, $q)

println("\n--- nn! (single point, caller-owned bufs) ---")
@btime nn!($tree_kd, $q, $idx1, $dist1)
