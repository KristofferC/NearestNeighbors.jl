module TestMonkeyKDTree
# Randomized tests for KDTree; the driver lives in TestSetup.monkey_tests.
# KDTree only supports Minkowski metrics.
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "..", "TestSetup.jl"))
using ..Main.TestSetup: monkey_tests, fullmetrics
using NearestNeighbors
using Test
using Distances: MinkowskiMetric

monkey_tests(KDTree, filter(m -> m isa MinkowskiMetric, fullmetrics))

end # module
