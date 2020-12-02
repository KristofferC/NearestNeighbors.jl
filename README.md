# NearestNeighbors.jl

[![Build Status](https://github.com/KristofferC/NearestNeighbors.jl/workflows/CI/badge.svg)](https://github.com/KristofferC/NearestNeighbors.jl/actions?query=workflows/CI)
 [![codecov.io](https://codecov.io/github/KristofferC/NearestNeighbors.jl/coverage.svg?branch=master)](https://codecov.io/github/KristofferC/NearestNeighbors.jl?branch=master) [![DOI](https://zenodo.org/badge/45321556.svg)](https://zenodo.org/badge/latestdoi/45321556)



 `NearestNeighbors.jl` is a package written in Julia to perform high performance nearest neighbor searches in
 arbitrarily high dimensions.

-----------------------------


## Creating a tree

The abstract tree type that the trees in this package are a subtype of is called a `NNTree`. A `NNTree`
is created by the function:
```jl
NNTree(data, metric; leafsize, reorder)
```

* `data`: The data, i.e., the points to build up the tree from. It can either be 
    - a matrix of size `nd × np` with the points to insert in the tree where `nd` is the dimensionality of the points and `np` is the number of points
    - a vector of vectors with fixed dimensionality, `nd`, which must be part of the type. Specifically, `data` should be a `Vector{V}`, where `V` is itself a subtype of an `AbstractVector` and such that `eltype(V)` and `length(V)` are defined.
        (For example, with 3D points, `V = SVector{3, Float64}` works because `eltype(V) = Float64` and `length(V) = 3` are defined in `V`.)
* `metric`: The metric to use, defaults to `Euclidean`. This is one of the `Metric` types defined in the `Distances.jl` packages. It is possible to define your own metrics by simply creating new types that are subtypes of `Metric`.
* `leafsize` (keyword argument): Determines at what number of points to stop splitting the tree further. There is a trade-off between traversing the tree and having to evaluate the metric function for increasing number of points.
* `reorder` (keyword argument): While building the tree this will put points close in distance close in memory since this helps with cache locality. In this case, a copy of the original data will be made so that the original data is left unmodified. This can have a significant impact on performance and is by default set to `true`.

There are currently three types of trees available:

* `BruteTree`: Not actually a tree. It linearly searches all points in a brute force fashion. Works with any `Metric`.
* `KDTree`: In a kd tree the points are recursively split into groups using hyper-planes.
Therefore a `KDTree` only works with axis aligned metrics which are: `Euclidean`, `Chebyshev`, `Minkowski` and `Cityblock`.
* `BallTree`: Points are recursively split into groups bounded by hyper-spheres. Works with any `Metric`.

All trees in `NearestNeighbors.jl` are static which means that points can not be added or removed from an already created tree.

Here are a few examples of creating trees:
```jl
using NearestNeighbors
data = rand(3, 10^4)

# Create trees
kdtree = KDTree(data; leafsize = 10)
balltree = BallTree(data, Minkowski(3.5); reorder = false)
brutetree = BruteTree(data)
```

## k Nearest Neighbor (kNN) searches

A kNN search finds the `k` nearest neighbors to given point(s).
This is done with the method:

```jl
knn(tree, points, k, sortres = false, skip = always_false) -> idxs, dists
```

* `tree`: The tree instance
* `points`: A vector or matrix of points to find the `k` nearest neighbors to. If `points` is a vector of numbers then this represents a single point, if `points` is a matrix then the `k` nearest neighbors to each point (column) will be computed. `points` can also be a vector of other vectors where each element in the outer vector is considered a point.
* `sortres` (optional): Determines if the results should be sorted before returning.
In this case the results will be sorted in order of increasing distance to the point.
* `skip` (optional): A predicate to determine if a given point should be skipped, for
example if iterating over points and a point has already been visited.

It is generally better for performance to query once with a large number of points than to query multiple
times with one point per query.

As a convenience, if you only want the closest nearest neighbor, you can call `nn` instead for a cleaner result:

```jl
nn(tree, points, skip = always_false) -> idxs, dists
```

Some examples:

```jl
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
points = rand(3, 4);

idxs, dists = knn(kdtree, points, k, true);

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

idxs, dists = knn(kdtree, v, k, true);

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

## Range searches

A range search finds all neighbors within the range `r` of given point(s).
This is done with the method:

```jl
inrange(tree, points, r, sortres = false) -> idxs
```

Note that for performance reasons the distances are not returned. The arguments to `inrange` are the same as for `knn` except that `sortres` just sorts the returned index vector.

An example:

```jl
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
```

## Using on-disk data sets

By default, the trees store a copy of the `data` provided during construction. This is problematic in case you want to work on data sets which are larger than available memory, thus wanting to `mmap` the data or want to store the data in a different place, outside the tree.

`DataFreeTree` can be used to strip a constructed tree of its data field and re-link it with that data at a later stage. An example of using a large on-disk data set looks like this:

```jl
using Mmap
ndim = 2; ndata = 10_000_000_000
data = Mmap.mmap(datafilename, Matrix{Float32}, (ndim, ndata))
data[:] = rand(Float32, ndim, ndata)  # create example data
dftree = DataFreeTree(KDTree, data)
```

`dftree` now only stores the indexing data structures. It can be passed around, saved and reloaded independently of `data`.

To perform look-ups, `dftree` is re-linked to the underlying data:

```jl
tree = injectdata(dftree, data)  # yields a KDTree
knn(tree, data[:,1], 3)  # perform operations as usual
```


## Author

Kristoffer Carlsson -  @KristofferC - kristoffer.carlsson@chalmers.se
