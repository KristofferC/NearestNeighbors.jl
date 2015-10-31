# Does not test leaf_size
# Does not test different metrics
import Distances.evaluate

facts("NearestNeighbors") do

context("NearestNeighbors.knn") do

for metric in [Euclidean()]
    for TreeType in [BruteTree, BallTree, KDTree]
        #println("TreeType: $TreeType, metric: $metric")
        # 8 node rectangle
        data = [0.0 0.0 0.0 0.5 0.5 1.0 1.0 1.0;
                0.0 0.5 1.0 0.0 1.0 0.0 0.5 1.0]
        if BruteTree == TreeType
            tree = TreeType(data, metric)
        else
            tree = TreeType(data, metric; leafsize=2)
        end

        idxs, dists = knn(tree, [0.8, 0.8], 1)
        @fact idxs[1] --> 8 # Should be closest to top right corner
        @fact evaluate(metric, [0.2, 0.2], zeros(2)) --> roughly(dists[1])

        idxs, dists = knn(tree, [0.1, 0.8], 3)
        @fact idxs --> [5, 2, 3]

        idxs, dists = knn(tree, [1//10, 8//10], 3)
        @fact idxs --> [5, 2, 3]

        @fact_throws ArgumentError knn(tree, [0.1, 0.8], 10) # k > n_points
        @fact_throws ArgumentError knn(tree, [0.1], 10) # n_dim != trees dim
    end
end

end  #context

end # facts
