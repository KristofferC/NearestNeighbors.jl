# Only tests BruteTree as other trees need Static Vectors to use length(Type)
import Distances.evaluate
using Base.Test

m = 10
n = 100

spdata = sprand(n,m,0.3)
data = full(spdata)
# metric = Distances.Euclidean()
vs = [spdata[:,j] for j=1:m]

@testset "sparse knn" begin
    @testset "metric" for metric in metrics
        @testset "tree type" for TreeType in trees_with_brute[1:1]

            @testset "compare distances" begin
                evals = (Distances.evaluate(metric, data[:,i], data[:,j]) for i=1:m, j=1:m)
                spevals = (Distances.evaluate(metric, spdata[:,i], spdata[:,j]) for i=1:m, j=1:m)
                spstate = start(spevals)
                for e in evals
                    spe, spstate = next(spevals,spstate)
                    @test e ≈ spe
                end
            end

            @testset "compare knn" begin
                tree = TreeType(data, metric; leafsize=2)
                sptree = TreeType(vs, metric; leafsize=2)

                for i=1:length(vs)
                    idxs, dists = knn(tree, full(vs[i]), 1)
                    spidxs, spdists = knn(sptree, vs[i], 1)

                    @test idxs == spidxs
                    @test dists == spdists
                end
            end
        end
    end
end
