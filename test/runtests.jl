
using NearestNeighbors
using ParallelTestRunner

testsuite = find_tests(@__DIR__)
# TestSetup.jl provides shared utilities but is not a test file itself
haskey(testsuite, "TestSetup") && delete!(testsuite, "TestSetup")

# Auto CPU thread count detection in ParallelTestRunner is bad
push!(ARGS, "--jobs=$(Sys.CPU_THREADS)")
runtests(NearestNeighbors, ARGS; testsuite)
