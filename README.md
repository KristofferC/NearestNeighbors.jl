# NearestNeighbors.jl

[![Build Status](https://travis-ci.org/KristofferC/NearestNeighbors.jl.svg?branch=master)](https://travis-ci.org/KristofferC/NearestNeighbors.jl) [![Build status](https://ci.appveyor.com/api/projects/status/lj0lk3c0pgwn06xe?svg=true)](https://ci.appveyor.com/project/KristofferC/nearestneighbors-jl)
 [![codecov.io](https://codecov.io/github/KristofferC/NearestNeighbors.jl/coverage.svg?branch=master)](https://codecov.io/github/KristofferC/NearestNeighbors.jl?branch=master)

 `NearestNeighbors.jl` is a package written in Julia to perform high performance nearest neighbor searches in
 arbitrarily high dimensions.

 **Note:** Currently the latest master of `Distances.jl` is required for this package. Get it by running `Pkg.checkout("Distances")`.

-----------------------------


## Creating a tree

The abstract tree type that the trees in this package is a subtype of is called a `NNTree`. A `NNTree`
is created by the function:
```jl
NNTree(data, metric; leafsize, reorder)
```

* `data`: A  matrix of size `nd × np` with the points to insert in the tree. `nd` is the dimensionality of the points, `np` is the number of points.
* `metric`: The metric to use, defaults to `Euclidean`. This is one of the `Metric` types defined in the `Distances.jl` packages.
* `leafsize` (keyword argument): Determines at what number of points to stop splitting the tree further. There is a trade-off between traversing the tree and evaluating the distance function for .
* `reorder` (keyword argument): While building the tree this will put points close in distance close in memory since this helps with cache locality. In this case, a copy of the original data will be made so that the original data is left unmodified. This can have a significant impact on performance and is by default set to `true`.

There are currently three types of trees available:

* `BruteTree`: Not actually a tree. It linearly searches all points in a brute force fashion. Works with any `Metric`.
* `KDTree`: In a kd tree the points recursively split into groups using hyper-planes.
Therefore a `KDTree` only work axis aligned metrics which are: `Euclidean`, `Chebyshev`, `Minkowski` and `Cityblock`.
* `BallTree`, points are recursively split into groups using hyper-spheres. Works with any `Metric`.

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

A kNN search is the method of finding the `k` nearest neighbors to given point(s).
This is done with the method:

```jl
knn(tree, points, k, sortres = false) -> idxs, dists
```

* `tree`: The tree instance
* `points`: A vector or matrix of points to find the `k` nearest neighbors to. If `points` is a vector then this represents a single point, if `points` is a matrix then the `k` nearest neighbors to each point (column) will be computed.
* `sortres` (optional): Determines if the results should be sorted before returning.
In this case the results will be sorted in order of increasing distance to the point.

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

A range search is the method of finding all neighbors within the range `r` of given point(s).
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
idxs = inrange(kdtree, point, r, true)

# Result in idxs:
# 4-element Array{Int64,1}:
#  317
#  983
# 4577
# 8675
```

## Debugging

There are some basic debugging/statistics functionality implemented. These are activated by setting the
`DEBUG` variable to `true` in the `NearestNeighbors.jl` file. For the debugging options, please see the `debugging.jl` file. Pull requests to enhance this are welcome.

## Author

Kristoffer Carlsson -  @KristofferC - kristoffer.carlsson@chalmers.se
