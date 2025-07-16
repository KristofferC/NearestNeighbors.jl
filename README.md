# NearestNeighbors.jl

[![Build Status](https://github.com/KristofferC/NearestNeighbors.jl/workflows/CI/badge.svg)](https://github.com/KristofferC/NearestNeighbors.jl/actions?query=workflows/CI)
[![codecov.io](https://codecov.io/github/KristofferC/NearestNeighbors.jl/coverage.svg?branch=master)](https://codecov.io/github/KristofferC/NearestNeighbors.jl?branch=master)
[![DOI](https://zenodo.org/badge/45321556.svg)](https://zenodo.org/badge/latestdoi/45321556)

`NearestNeighbors.jl` is a Julia package for performing nearest neighbor searches.

## Creating a Tree

There are currently four types of trees available:

* `KDTree`: Recursively splits points into groups using hyper-planes. Best for low-dimensional data with axis-aligned metrics.
* `BallTree`: Recursively splits points into groups bounded by hyper-spheres. Suitable for high-dimensional data and arbitrary metrics.
* `BruteTree`: Not actually a tree. It linearly searches all points in a brute force manner. Useful as a baseline or for small datasets.
* `PeriodicTree`: Wraps one of the trees above to handle periodic boundary conditions. Essential for simulations with periodic domains.

These trees can be created using the following syntax:

```julia
KDTree(data, metric; leafsize, reorder)
BallTree(data, metric; leafsize, reorder)
BruteTree(data, metric; leafsize, reorder) # leafsize and reorder are unused for BruteTree
PeriodicTree(tree, bounds_min, bounds_max)
```

* `data`: The points to build the tree from, either as
    - A matrix of size `nd × np` where `nd` is the dimensionality and `np` is the number of points, or
    - A vector of vectors with fixed dimensionality `nd`, i.e., `data` should be a `Vector{V}` where `V` is a subtype of `AbstractVector` with defined `length(V)`. For example a `Vector{V}` where `V = SVector{3, Float64}` is ok because `length(V) = 3` is defined.
* `metric`: The `Metric` (from `Distances.jl`) to use, defaults to `Euclidean`. `KDTree` works with axis-aligned metrics: `Euclidean`, `Chebyshev`, `Minkowski`, and `Cityblock` while for `BallTree` and `BruteTree` other pre-defined `Metric`s can be used as well as custom metrics (that are subtypes of `Metric`).
* `leafsize`: Determines the number of points (default 25) at which to stop splitting the tree. There is a trade-off between tree traversal and evaluating the metric for an increasing number of points.
* `reorder`: If `true` (default), during tree construction this rearranges points to improve cache locality during querying. This will create a copy of the original data.
* `tree`: An existing tree (`KDTree`, `BallTree`, or `BruteTree`) built from your data.
* `bounds_min`, `bounds_max`: Vectors defining the periodic domain boundaries. Use `Inf` in `bounds_max` for non-periodic dimensions.

All trees in `NearestNeighbors.jl` are static, meaning points cannot be added or removed after creation.
Note that this package is not suitable for very high dimensional points due to high compilation time and inefficient queries on the trees.

Example of creating trees:
```julia
using NearestNeighbors
data = rand(3, 10^4)

# Create trees
kdtree = KDTree(data; leafsize = 25)
balltree = BallTree(data, Minkowski(3.5); reorder = false)
brutetree = BruteTree(data)
periodictree = PeriodicTree(kdtree, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
```

### Parallel Tree Building

`KDTree` and `BallTree` support parallel tree construction when multiple threads are available:

* `parallel`: Enable/disable parallel tree building (default: `Threads.nthreads() > 1`)

For large datasets, parallel construction can provide significant speedups (~4× faster with 8 threads). Start Julia with `julia --threads=N` to enable.

```julia
# Parallel by default when multiple threads available
kdtree = KDTree(data)
balltree = BallTree(data)

# Explicitly disable parallel building
kdtree_seq = KDTree(data; parallel=false)
```

## k-Nearest Neighbor (kNN) Searches

A kNN search finds the `k` nearest neighbors to a given point or points. This is done with the methods:

```julia
knn(tree, point[s], k [, skip=Returns(false)]) -> idxs, dists
knn!(idxs, dists, tree, point, k [, skip=Returns(false)])
```

* `tree`: The tree instance.
* `point[s]`: A vector or matrix of points to find the `k` nearest neighbors for. A vector of numbers represents a single point; a matrix means the `k` nearest neighbors for each point (column) will be computed. `points` can also be a vector of vectors.
* `k`: Number of nearest neighbors to find.
* `skip` (optional): A predicate function to skip certain points, e.g., points already visited.


For the single closest neighbor, you can use `nn`:

```julia
nn(tree, point[s] [, skip=Returns(false)]) -> idx, dist
```

Examples:

```julia
using NearestNeighbors
data = rand(3, 10^4)
k = 3
point = rand(3)

kdtree = KDTree(data)
idxs, dists = knn(kdtree, point, k)

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
idxs, dists = knn(kdtree, points, k)

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
using StaticArrays
v = @SVector[0.5, 0.3, 0.2];

idxs, dists = knn(kdtree, v, k)

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

# Preallocating input results
idxs, dists = zeros(Int32, k), zeros(Float32, k)
knn!(idxs, dists, kdtree, v, k)
```

## Range Searches

A range search finds all neighbors within the range `r` of given point(s). This is done with the methods:

```julia
inrange(tree, point[s], radius) -> idxs
inrange!(idxs, tree, point, radius)
```

* `tree`: The tree instance.
* `point[s]`: A vector or matrix of points to find neighbors for.
* `radius`: Search radius.

Note: Distances are not returned, only indices.

Example:

```julia
using NearestNeighbors
data = rand(3, 10^4)
r = 0.05
point = rand(3)

balltree = BallTree(data)
idxs = inrange(balltree, point, r)

# 4-element Array{Int64,1}:
#  317
#  983
# 4577
# 8675

# Updates `idxs`
idxs = Int32[]
inrange!(idxs, balltree, point, r)

# counts points without allocating index arrays
neighborscount = inrangecount(balltree, point, r)
```

## Periodic Boundary Conditions

The `PeriodicTree` provides nearest neighbor searches with periodic boundary conditions.

### Creating a PeriodicTree

A `PeriodicTree` wraps an existing tree (`KDTree`, `BallTree`, or `BruteTree`) and handles periodic boundary conditions:

```julia
PeriodicTree(tree, bounds_min, bounds_max)
```

* `tree`: An existing tree built from your data
* `bounds_min`: Vector of minimum bounds for each dimension
* `bounds_max`: Vector of maximum bounds for each dimension (use `Inf` for non-periodic dimensions)

### Examples

**Basic periodic boundaries:**
```julia
using NearestNeighbors, StaticArrays

# Create data in a 2D periodic domain
data = [SVector(0.1, 0.2), SVector(0.8, 0.9), SVector(0.5, 0.5)]
kdtree = KDTree(data)

# Create periodic tree with bounds [0,1] × [0,1]
ptree = PeriodicTree(kdtree, [0.0, 0.0], [1.0, 1.0])

# Query near boundary - finds neighbors through periodic wrapping
query_point = [0.05, 0.15]  # Near data[1] = [0.1, 0.2]
neighbor_point = [0.95, 0.85] # Near data[2] = [0.8, 0.9] via wrapping

idxs, dists = knn(ptree, query_point, 2)
# Finds both nearby points, including wrapped distances
```

**Mixed periodic/non-periodic dimensions:**
```julia
# 2D domain: x-periodic, y-infinite
data = [SVector(1.0, 2.0), SVector(9.0, 8.0)]
kdtree = KDTree(data)
ptree = PeriodicTree(kdtree, [0.0, 0.0], [10.0, Inf])

# Query near x-boundary finds wrapped neighbor
query = [0.5, 3.0]
idxs, dists = knn(ptree, query, 1)
# Finds data[1] with wrapped x-distance of 0.5 instead of 8.5
```

## Using On-Disk Data Sets

By default, trees store a copy of the `data` provided during construction. For data sets larger than available memory, `DataFreeTree` can be used to strip a tree of its data field and re-link it later.

Example with a large on-disk data set:

```julia
using Mmap
ndim = 2
ndata = 10_000_000_000
data = Mmap.mmap(datafilename, Matrix{Float32}, (ndim, ndata))
data[:] = rand(Float32, ndim, ndata) # create example data
dftree = DataFreeTree(KDTree, data)
```

`dftree` stores the indexing data structures. To perform look-ups, re-link the tree to the data:

```julia
tree = injectdata(dftree, data) # yields a KDTree
knn(tree, data[:,1], 3) # perform operations as usual
```
