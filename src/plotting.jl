
using KDTrees
import KDTrees.HyperRectangle
using NearestNeighbors
import NearestNeighbors.HyperSphere


function plot2d(hr::HyperRectangle)
    x = [hr.mins[1], hr.maxes[1], hr.maxes[1], hr.mins[1], hr.mins[1]]
    y = [hr.mins[2], hr.mins[2], hr.maxes[2], hr.maxes[2], hr.mins[2]]
    plot(x,y)
end

#a = rand(2, 100)
#tree = KDTree(a; leafsize=1)

#plot(vec(tree.data[1,:]), vec(tree.data[2,:]), "*")

#for hr in reverse(tree.hyper_recs[1:100])
#    plot2d(hr)
#end

using PyPlot
using PyCall
@pyimport matplotlib.patches as patch
import NearestNeighbors.HyperSphere



cfig = figure()
ax = cfig[:add_subplot](1,1,1)
ax[:set_aspect]("equal")
axis((-3,3,-3,3))

function add_sphere(ax, hs::HyperSphere)
    ell = patch.Circle(hs.center, radius = hs.r, facecolor="none")
    ax[:add_artist](ell)
end

for hr in reverse(tree.hyper_spheres[150:end])
    add_sphere(ax, hr)
end

workspace()
using NearestNeighbors
a = rand(2, 100)

a = rand(5, 10^4)
p = rand(5)
kdtree = KDTree(a, Euclidean())
brutetree = BruteTree(a, Euclidean())
balltree = BallTree(a, Euclidean())


inrange(kdtree, p, 0.2, true)

NearestNeighbors.print_stats()
NearestNeighbors.reset_stats()

inrange(balltree, p, 0.2, true)

NearestNeighbors.print_stats()
NearestNeighbors.reset_stats()

inrange(brutetree, p, 0.2, true)

NearestNeighbors.print_stats()
NearestNeighbors.reset_stats()



knn(kdtree, p, 5, true)
knn(brutetree, p, 5, true)





@time for i = 1:10^4 inrange(balltree, rand(5), 0.2, true) end
@time for i = 1:10^4 inrange(brutetree, rand(5), 0.2, true) end


inrange(brutetree, p, 0.2, true)
