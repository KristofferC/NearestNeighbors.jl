#!/usr/bin/env julia

using NearestNeighbors
using StableRNGs
using BenchmarkTools
using Printf

function benchmark_build()
    # Setup data
    rng = StableRNG(1)
    data_small = rand(rng, 3, 10^4)   # Small dataset
    data_medium = rand(rng, 3, 10^5)  # Medium dataset
    data_large = rand(rng, 3, 10^6)   # Large dataset

    datasets = [
        ("Small (10k points)", data_small),
        ("Medium (100k points)", data_medium),
        ("Large (1M points)", data_large)
    ]

    current_threads = Threads.nthreads()

    for (dataset_name, data) in datasets
        # Benchmark BallTree with parallel=true
        ball_time_parallel = @benchmark BallTree($data, parallel=true) samples=5 evals=1
        ball_minimum_parallel = minimum(ball_time_parallel).time / 1e6  # Convert to ms

        # Benchmark BallTree with parallel=false
        ball_time_sequential = @benchmark BallTree($data, parallel=false) samples=5 evals=1
        ball_minimum_sequential = minimum(ball_time_sequential).time / 1e6  # Convert to ms

        # Benchmark KDTree with parallel=true
        kd_time_parallel = @benchmark KDTree($data, parallel=true) samples=5 evals=1
        kd_minimum_parallel = minimum(kd_time_parallel).time / 1e6  # Convert to ms

        # Benchmark KDTree with parallel=false
        kd_time_sequential = @benchmark KDTree($data, parallel=false) samples=5 evals=1
        kd_minimum_sequential = minimum(kd_time_sequential).time / 1e6  # Convert to ms

        println(@sprintf("%-20s %-8d %-12.2f %-12.2f %-12.2f %-12.2f",
            dataset_name, current_threads, ball_minimum_parallel, ball_minimum_sequential, kd_minimum_parallel, kd_minimum_sequential))
    end
end

# Run the benchmark
benchmark_build()
