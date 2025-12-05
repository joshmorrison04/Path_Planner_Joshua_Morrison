import argparse
from mbot_bridge.api import MBot

from src.graph import GridGraph, Cell
from src.graph_search import a_star_search, breadth_first_search, depth_first_search
from src.utils import generate_plan_file


def cells_to_poses(path, g):
    """Convert a list of Cell objects into [x, y, theta] poses in meters."""
    return [[*g.cell_to_pos(c.i, c.j), 0] for c in path]


def parse_args():
    parser = argparse.ArgumentParser(description="HelloRob Path Planning on the Robot.")
    parser.add_argument(
        "-m",
        "--map",
        type=str,
        default="/home/mbot/current.map",
        help="Path to the map file."
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=2,
        default=[0, 0],
        help="Goal position (x y) in meters."
    )
    parser.add_argument(
        "-r",
        "--collision-radius",
        type=float,
        default=15,
        help="Collision radius (meters)."
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["bfs", "dfs", "astar"],
        default="bfs",
        help="Planning algorithm to use."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # 1. Create the GridGraph using the robot's map.
    graph = GridGraph(args.map, collision_radius=args.collision_radius)

    # 2. Convert the goal position in meters to a goal cell in the grid.
    goal = graph.pos_to_cell(*args.goal)

    # 3. Initialize the MBot object.
    robot = MBot()

    # 4. Get the current SLAM pose [x, y, theta] and convert it to a start cell.
    start_pose = robot.read_slam_pose()
    start = graph.pos_to_cell(*start_pose[:2])

    # 5. Call the selected graph search function to compute a path.
    if args.algo == "bfs":
        path = breadth_first_search(graph, start, goal)
    elif args.algo == "dfs":
        path = depth_first_search(graph, start, goal)
    elif args.algo == "astar":
        path = a_star_search(graph, start, goal)
    else:
        # Should never hit this because of choices=...
        path = []

    # 6. Check if we actually found a path.
    if not path:
        print(f"[{args.algo}] No path found from start to goal. Not sending a path to the robot.")
    else:
        print(f"[{args.algo}] Found path of length {len(path)}. Driving to the goal!")

        # 7. Send the path to the robot as a list of [x, y, theta] poses.
        robot.drive_path(cells_to_poses(path, graph))

    # 8. Generate the planner file for later visualization in the nav app.
    generate_plan_file(graph, start, goal, path, algo=args.algo)

