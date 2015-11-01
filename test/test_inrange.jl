# Does not test leaf_size

facts("NearestNeighbors") do

context("NearestNeighbors.inrange") do

for metric in [Euclidean()]
    for TreeType in trees_with_brute
        data = [0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0;
                0.0 0.0 1.0 1.0 0.0 0.0 1.0 1.0;
                0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0] # 8 node cube

        if BruteTree == TreeType
            tree = TreeType(data, metric)
        else
            tree = TreeType(data, metric; leafsize=2)
        end
        dosort = true

        idxs = inrange(tree, [1.1, 1.1, 1.1], 0.2, dosort)
        @fact idxs --> [8] # Only corner 8 at least 0.2 distance away from [1.1, 1.1, 1.1]

        idxs = inrange(tree, [0.0, 0.0, 0.5], 0.6, dosort)
        @fact idxs --> [1, 2] # Corner 1 and 2 at least 0.6 distance away from [0.0, 0.0, 0.5]

        idxs = inrange(tree, [0, 0, 0], 0.6, dosort)
        @fact idxs --> [1]

        idxs = inrange(tree, [0.33333333333, 0.33333333333, 0.33333333333], 1, dosort)
        @fact idxs --> [1, 2, 3, 5]

        idxs = inrange(tree, [0.5, 0.5, 0.5], 0.2, dosort)
        @fact idxs --> []

        idxs = inrange(tree, [0.5, 0.5, 0.5], 1.0, dosort)
        @fact idxs --> [1, 2, 3, 4, 5, 6, 7, 8]

        @fact_throws ArgumentError inrange(tree, rand(3), -0.1)
        @fact_throws ArgumentError inrange(tree, rand(5), 1.0)
    end
end
end # context

end # facts
