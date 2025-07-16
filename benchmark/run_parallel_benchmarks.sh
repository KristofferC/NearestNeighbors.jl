#!/bin/bash

echo "Running parallel tree building benchmarks..."
echo "==========================================="
echo ""
echo "$(printf "%-20s %-8s %-12s %-12s %-12s %-12s" "Dataset" "Threads" "Ball-Par" "Ball-Seq" "KD-Par" "KD-Seq")"
echo "$(printf "%s" "$(printf -- '-%.0s' {1..80})")"

# Test with different thread counts
for threads in 1 2 4 8; do
    julia --threads=$threads --project benchmark_parallel.jl
done

echo ""
echo "All benchmarks completed!"