# NearestNeighbors.jl

[![Build Status](https://github.com/KristofferC/NearestNeighbors.jl/workflows/CI/badge.svg)](https://github.com/KristofferC/NearestNeighbors.jl/actions?query=workflows/CI)
[![codecov.io](https://codecov.io/github/KristofferC/NearestNeighbors.jl/coverage.svg?branch=master)](https://codecov.io/github/KristofferC/NearestNeighbors.jl?branch=master)
[![DOI](https://zenodo.org/badge/45321556.svg)](https://zenodo.org/badge/latestdoi/45321556)

`NearestNeighbors.jl` is a Julia package for performing nearest neighbor searches.

## Creating a Tree

There are currently three types of trees available:

* `BruteTree`: Not actually a tree. It linearly searches all points in a brute force manner. Works with any `Metric`.
* `KDTree`: Recursively splits points into groups using hyper-planes. Only works with axis-aligned metrics: `Euclidean`, `Chebyshev`, `Minkowski`, and `Cityblock`.
* `BallTree`: Recursively splits points into groups bounded by hyper-spheres. Works with any `Metric`.

These trees can be created using the following syntax:

```julia
BruteTree(data, metric; leafsize, reorder)
KDTree(data, metric; leafsize, reorder)
BallTree(data, metric; leafsize, reorder)
```

* `data`: The points to build the tree from, either as
    - A matrix of size `nd × np` where `nd` is the dimensionality and `np` is the number of points, or
    - A vector of vectors with fixed dimensionality `nd`, i.e., `data` should be a `Vector{V}` where `V` is a subtype of `AbstractVector` with defined `length(V)`. For example a `Vector{V}` where `V = SVector{3, Float64}` is ok because `length(V) = 3` is defined.
* `metric`: The metric to use, defaults to `Euclidean`. You can use metrics from the `Distances.jl` package or define your own by creating new types that are subtypes of `Metric`.
* `leafsize`: Determines the number of points (default 10) at which to stop splitting the tree. There is a trade-off between tree traversal and evaluating the metric for an increasing number of points.
* `reorder`: If `true` (default), during tree construction this rearranges points to improve cache locality during querying. This will create a copy of the original data.

All trees in `NearestNeighbors.jl` are static, meaning points cannot be added or removed after creation.

Example of creating trees:
```julia
using NearestNeighbors
data = rand(3, 10^4)

# Create trees
kdtree = KDTree(data; leafsize = 10)
balltree = BallTree(data, Minkowski(3.5); reorder = false)
brutetree = BruteTree(data)
```

## k-Nearest Neighbor (kNN) Searches

A kNN search finds the `k` nearest neighbors to a given point or points. This is done with the method:

```julia
knn(tree, points, k, skip = always_false) -> idxs, dists
```

* `tree`: The tree instance.
* `points`: A vector or matrix of points to find the `k` nearest neighbors for. A vector of numbers represents a single point; a matrix means the `k` nearest neighbors for each point (column) will be computed. `points` can also be a vector of vectors.
* `skip` (optional): A predicate to skip certain points, e.g., points already visited.


For the singlke closest neighbor, you can use `nn`:

```julia
nn(tree, points, skip = always_false) -> idxs, dists
```

Examples:

```julia
using NearestNeighbors
data = rand(3, 10^4)
k = 3
point = rand(3)

kdtree = KDTree(data)
idxs, dists = knn(kdtree, point, k, true)

idxs
# 3-element Array{Int64,1}:
#  4683
#  6119
#  3278

dists
# 3-element Array{Float64,1}:
#  0.039032201026256215
#  0.04134193711411951
#  0.042974090446474184

# Multiple points
points = rand(3, 4)
idxs, dists = knn(kdtree, points, k, true)

idxs
# 4-element Array{Array{Int64,1},1}:
#  [3330, 4072, 2696]
#  [1825, 7799, 8358]
#  [3497, 2169, 3737]
#  [1845, 9796, 2908]

# dists
# 4-element Array{Array{Float64,1},1}:
#  [0.0298932, 0.0327349, 0.0365979]
#  [0.0348751, 0.0498355, 0.0506802]
#  [0.0318547, 0.037291, 0.0421208]
#  [0.03321, 0.0360935, 0.0411951]

# Static vectors
v = @SVector[0.5, 0.3, 0.2];

idxs, dists = knn(kdtree, v, k, true)

idxs
# 3-element Array{Int64,1}:
#   842
#  3075
#  3046

dists
# 3-element Array{Float64,1}:
#  0.04178677766255837
#  0.04556078331418939
#  0.049967238112417205
```

## Range Searches

A range search finds all neighbors within the range `r` of given point(s). This is done with the method:

```julia
inrange(tree, points, r) -> idxs
```

Distances are not returned.

Example:

```julia
using NearestNeighbors
data = rand(3, 10^4)
r = 0.05
point = rand(3)

balltree = BallTree(data)
idxs = inrange(balltree, point, r, true)

# 4-element Array{Int64,1}:
#  317
#  983
# 4577
# 8675

neighborscount = inrangecount(balltree, point, r, true)  # counts points without allocating index arrays
```

## Using On-Disk Data Sets

By default, trees store a copy of the `data` provided during construction. For data sets larger than available memory, `DataFreeTree` can be used to strip a tree of its data field and re-link it later.

Example with a large on-disk data set:

```julia
using Mmap
ndim = 2
ndata = 10_000_000_000
data = Mmap.mmap(datafilename, Matrix{Float32}, (ndim, ndata))
data[:] = rand(Float32, ndim, ndata)  # create example data
dftree = DataFreeTree(KDTree, data)
```

`dftree` stores the indexing data structures. To perform look-ups, re-link the tree to the data:

```julia
tree = injectdata(dftree, data)  # yields a KDTree
knn(tree, data[:,1], 3)  # perform operations as usual
```

