import numpy as np
from collections import deque
from .graph import Cell
from .utils import trace_path

"""
General graph search instructions:

- The graph.init_graph() call resets parent, visited, and distance info.
- Use graph.find_neighbors(i, j) to get valid neighbors.
- Use graph.visited_cells.append(Cell(i, j)) to record cells for visualization.
- Use trace_path(goal_cell, graph) to return the final path.
"""


def depth_first_search(graph, start, goal):
    """Depth First Search (DFS) algorithm. Optional for P3."""
    graph.init_graph()
    # TODO (optional): implement DFS
    return []


def breadth_first_search(graph, start, goal):
    """Breadth First Search (BFS) algorithm."""
    graph.init_graph()

    # Extract start & goal indices
    si, sj = start.i, start.j
    gi, gj = goal.i, goal.j

    # Queue for BFS (stores (i, j) tuples)
    q = deque()
    q.append((si, sj))

    # Initialize start node
    graph.visited[sj, si] = True
    graph.distances[sj, si] = 0
    graph.parents[sj, si] = None   # Start has no parent

    # BFS loop
    while q:
        ci, cj = q.popleft()

        # Add to list for visualization on web-app
        graph.visited_cells.append(Cell(ci, cj))

        # Check for goal
        if ci == gi and cj == gj:
            return trace_path(Cell(gi, gj), graph)

        # Explore neighbors
        for (ni, nj) in graph.find_neighbors(ci, cj):
            if not graph.visited[nj, ni]:
                graph.visited[nj, ni] = True
                graph.distances[nj, ni] = graph.distances[cj, ci] + 1
                graph.parents[nj, ni] = Cell(ci, cj)

                q.append((ni, nj))

    # If no path found
    return []


def a_star_search(graph, start, goal):
    """A* Search algorithm. Optional for P3."""
    graph.init_graph()
    # TODO (optional): implement A*
    return []
