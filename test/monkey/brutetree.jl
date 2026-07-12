module TestMonkeyBruteTree
# Randomized tests for BruteTree; the driver lives in TestSetup.monkey_tests.
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "..", "TestSetup.jl"))
using ..Main.TestSetup: monkey_tests, fullmetrics
using NearestNeighbors
using Test

monkey_tests(BruteTree, fullmetrics)

end # module
