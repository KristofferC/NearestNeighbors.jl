# NearestNeighbors.jl

[![Build Status](https://travis-ci.org/KristofferC/NearestNeighbors.jl.svg?branch=master)](https://travis-ci.org/KristofferC/NearestNeighbors.jl) [![Build status](https://ci.appveyor.com/api/projects/status/lj0lk3c0pgwn06xe?svg=true)](https://ci.appveyor.com/project/KristofferC/nearestneighbors-jl)
 [![codecov.io](https://codecov.io/github/KristofferC/NearestNeighbors.jl/coverage.svg?branch=master)](https://codecov.io/github/KristofferC/NearestNeighbors.jl?branch=master)

 `NearestNeighbors.jl` is a package written in Julia to perform high performance nearest neighbor searches in
 arbitrarily high dimensions.

-----------------------------


## Creating a tree

The abstract tree type that the trees in this package are a subtype of is called a `NNTree`. A `NNTree`
is created by the function:
```jl
NNTree(data, metric; leafsize, reorder)
```

* `data`: This parameter represents the points to build up the tree from. It can either be a matrix of size `nd × np` with the points to insert in the tree where `nd` is the dimensionality of the points, `np` is the number of points or it can be a `Vector{V}` where `V` is itself a subtype of an `AbstractVector` and such that `eltype(V)` and `length(V)` is defined.
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

An example:

```jl
using NearestNeighbors
data = rand(3, 10^4)
k = 5
point = rand(3)

kdtree = KDTree(data)
idxs, dists = knn(kdtree, point, k, true)

# Results in idxs and dists:

# idxs:
# 5-element Array{Int64,1}:
#  5667
#  7247
#  9277
#  5327
#  4449

# dists:
# 5-element Array{Float64,1}:
#  0.0314375
#  0.0345444
#  0.0492791
#  0.0512624
#  0.0559252
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

# Result in idxs:
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

In case you want to exploit the reordering feature, which can improve access times by placing data items close together in memory / on disk when they are close together according to the metric used, you can pass a custom `reorderbuffer`. This can be either in-memory or mmapped, as in the following example:

```jl
reorderbuffer = Mmap.mmap(reorderedfilename, Matrix{Float32}, (ndim, ndata))
dftree = DataFreeTree(KDTree, data, reorderbuffer = reorderbuffer)
# all future operations are indepented of 'data'
tree = injectdata(dftree, reorderbuffer)
```

## Debugging

There are some basic debugging/statistics functionality implemented. These are activated by setting the
`DEBUG` variable to `true` in the `NearestNeighbors.jl` file. For the debugging options, please see the `debugging.jl` file. Pull requests to enhance this are welcome.

## Author

Kristoffer Carlsson -  @KristofferC - kristoffer.carlsson@chalmers.se
