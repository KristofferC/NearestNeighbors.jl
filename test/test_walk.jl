using StableRNGs, GeometryBasics
@testset "tree type" for TreeType in trees_with_brute
    @testset "type" for T in (Float32, Float64)
        @testset "tree walking" begin 
            allpts = rand(StableRNG(1), Point2{T},  1000)
            tree = TreeType(allpts)

            function _find_in_node!(node, indices, pts)
                if isleaf(node)
                    for (point, index) in zip(points(node), points_indices(node))
                        @test point == allpts[index] 
                    end 

                    for point in points(node) 
                        push!(pts, point)
                    end 
                    for index in points_indices(node)
                        push!(indices, index)
                    end 
                else  
                    left, right = children(node)
                    _find_in_node!( left, indices, pts)
                    _find_in_node!(right, indices, pts)
                end 
            end 
            function find_all_points(root, allpts)
                # walk to find all points 
                indices = Set{Int64}() 
                pts = Set{Point2{T}}() 
                _find_in_node!(root, indices, pts)
                @test indices == Set(eachindex(allpts))
                @test pts == Set(allpts)
            end 

            find_all_points(root(tree), allpts) 
            find_all_points(skip_regions(root(tree)), allpts) 
        end 
        @testset "region containment" begin 
            allpts = rand(StableRNG(2), Point2{T},  1000)
            tree = TreeType(allpts)

            function _contains(pt, region::NearestNeighbors.HyperRectangle)
                for i in eachindex(pt)
                    if pt[i] < region.mins[i] || pt[i] > region.maxes[i]
                        return false
                    end
                end
                return true
            end
            function _contains(pt, region::NearestNeighbors.HyperSphere)
                center = region.center
                radius = region.r
                dist = norm(pt - center)
                return dist <= radius
            end

            function _contains(subregion::NearestNeighbors.HyperSphere, region::NearestNeighbors.HyperSphere)
                # check that the center of the subregion is in the region
                return _contains(subregion.center, region) 
            end

            function _contains(subregion::NearestNeighbors.HyperRectangle, region::NearestNeighbors.HyperRectangle)
                return _contains(subregion.mins, region) && 
                       _contains(subregion.maxes, region) 
            end 
            
            function check_containment(node)
                r = region(node) 
                if isleaf(node)
                    for point in points(node)
                        @test _contains(point, r)
                    end
                else
                    left, right = children(node)
                    # check 
                    @test _contains(region(left), r)
                    @test _contains(region(right), r)
                end
            end

            check_containment(root(tree))
        end 
    end
end

