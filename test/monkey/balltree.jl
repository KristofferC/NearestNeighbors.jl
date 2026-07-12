module TestMonkeyBallTree
# Randomized tests for BallTree with the standard metrics; the driver lives in
# TestSetup.monkey_tests. The custom metrics run in balltree_custom.jl and
# Hamming is not supported (the bounding spheres need a vector-space metric).
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "..", "TestSetup.jl"))
using ..Main.TestSetup: monkey_tests, metrics
using NearestNeighbors
using Test

monkey_tests(BallTree, metrics)

end # module
