module TestMonkeyBallTreeCustom
# Randomized tests for BallTree with the custom test metrics (the slowest
# metrics to evaluate, hence their own test file); the driver lives in
# TestSetup.monkey_tests.
isdefined(Main, :TestSetup) || @eval Main include(joinpath(@__DIR__, "..", "TestSetup.jl"))
using ..Main.TestSetup: monkey_tests, CustomMetric1, CustomMetric2
using NearestNeighbors
using Test

monkey_tests(BallTree, [CustomMetric1(), CustomMetric2()])

end # module
