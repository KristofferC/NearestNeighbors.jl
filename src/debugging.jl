const NODES_VISITED = Ref{Int}(0)
const POINTS_VISITED = Ref{Int}(0)
const POINTS_VISITED_UNCHECKED = Ref{Int}(0)

add_node_visited(n::Int) = NODES_VISITED[] += n
add_point_visited(n::Int) = POINTS_VISITED[] += n
function add_point_visited_unchecked(n::Int)
    POINTS_VISITED[] += n
    POINTS_VISITED_UNCHECKED[] += n
end

function reset_stats()
    NODES_VISITED[] = 0
    POINTS_VISITED[] = 0
    return
end

function print_stats()
    println("Nodes visited: $(NODES_VISITED[])")
    println("Points visited: $(POINTS_VISITED[]), out of these: $(POINTS_VISITED_UNCHECKED[]) unchecked.")
end

macro POINT(n) end
macro POINT_UNCHECKED(n) end
macro NODE(n) end

if DEBUG
    println("Debugging for NearestNeighbors is active")
    @eval begin
        macro POINT(n)
            :(add_point_visited($n))
        end

        macro NODE(n)
            :(add_node_visited($n))
        end

        macro POINT_UNCHECK(n)
            :(add_point_visited_unchecked($n))
        end
    end
end
