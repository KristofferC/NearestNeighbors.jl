"""
Benchmark: three approaches to batch nearest-neighbour lookup.

  A. nn   — allocating: builds flat output vecs, delegates to nn!
  B. nn!  — in-place: caller-owned output, zero per-point allocations

nn! uses the scalar nn path internally (typed skip::F), so both KDTree
and BallTree produce zero per-point allocations.
"""

using NearestNeighbors, BenchmarkTools, StaticArrays, Random

# ── setup ───────────────────────────────────────────────────────────────
rng = MersenneTwister(42)
const DIM = 3
const N_TREE = 100_000
const N_QUERY = 1_000

data       = rand(rng, DIM, N_TREE)
query_mat  = rand(rng, DIM, N_QUERY)
query_svecs = [SVector{DIM,Float64}(query_mat[:, i]) for i in 1:N_QUERY]

tree_kd   = KDTree(data)
tree_ball = BallTree(data)

idxs_out  = Vector{Int}(undef, N_QUERY)
dists_out = Vector{Float64}(undef, N_QUERY)

# ── correctness check ────────────────────────────────────────────────────
let ref_i, ref_d
    ref_i, ref_d = nn(tree_kd, query_svecs)
    nn!(tree_kd, query_svecs, idxs_out, dists_out)
    @assert idxs_out == ref_i && dists_out ≈ ref_d "nn! disagrees with nn"
    println("Correctness check passed.")
end

for (label, tree) in (("KDTree", tree_kd), ("BallTree", tree_ball))
    println()
    println("=" ^ 60)
    println("$label, $(DIM)D, tree=$N_TREE pts, query=$N_QUERY SVecs")
    println("=" ^ 60)

    println("\n--- nn  (2 output allocs) ---")
    @btime nn($tree, $query_svecs)

    println("\n--- nn!  (0 allocs) ---")
    @btime nn!($tree, $query_svecs, $idxs_out, $dists_out)
end

# ── single-point form ────────────────────────────────────────────────────
println()
println("=" ^ 60)
println("Single point (KDTree)")
println("=" ^ 60)
q     = query_svecs[1]
idx1  = Vector{Int}(undef, 1)
dist1 = Vector{Float64}(undef, 1)

println("\n--- nn (scalar-tracking, no allocs) ---")
@btime nn($tree_kd, $q)

println("\n--- nn! (caller-owned bufs, no allocs) ---")
@btime nn!($tree_kd, $q, $idx1, $dist1)
