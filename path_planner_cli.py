import argparse
from src.graph import GridGraph, Cell
from src.graph_search import a_star_search, breadth_first_search, depth_first_search
from src.utils import generate_plan_file

# Argument parasar: Flags for map, start and end position, and the search algorithm
# Used when running terminal command

def parse_args():
    parser = argparse.ArgumentParser(description="HelloRob Path Planning Client.")
    parser.add_argument("-m", "--map", type=str, required=True, help="Path to the map file.")
    parser.add_argument("--start", type=int, nargs=2, required=True, help="Start cell.")
    parser.add_argument("--goal", type=int, nargs=2, required=True, help="Goal cell.")
    parser.add_argument("--algo", type=str, default="bfs", choices=["bfs", "dfs", "astar"],
                        help="Algorithm to use.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    # Construct the graph.
    # Converts the map into a grid the search algorithm can use.
    graph = GridGraph(args.map)

    # Construct the start and goal cells.
    # package the start and goal positions the user gave into cells so the search algorithm can use it
    start, goal = Cell(*args.start), Cell(*args.goal)

    # Run the alg chosen by user
    if args.algo == "astar":
        path = a_star_search(graph, start, goal)
    elif args.algo == "bfs":
        path = breadth_first_search(graph, start, goal)
    elif args.algo == "dfs":
        path = depth_first_search(graph, start, goal)
    else:
        print("Invalid option:", args.algo)
        exit()

    # Output the planning file for visualization.
    # - And then all of this is outputted as the planning file (out.planner) which we can load into the nav app. 
    generate_plan_file(graph, start, goal, path)
