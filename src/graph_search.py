import numpy as np
from collections import deque
import heapq

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

    si, sj = start.i, start.j
    gi, gj = goal.i, goal.j

    # Stack for DFS (LIFO), store (i, j)
    stack = [(si, sj)]

    # Initialize start node
    graph.visited[sj, si] = True
    graph.distances[sj, si] = 0
    graph.parents[sj, si] = None

    while stack:
        ci, cj = stack.pop()

        # Record for visualization
        graph.visited_cells.append(Cell(ci, cj))

        # Goal check
        if ci == gi and cj == gj:
            return trace_path(Cell(gi, gj), graph)

        # Explore neighbors (order affects traversal, but not correctness)
        for (ni, nj) in graph.find_neighbors(ci, cj):
            if not graph.visited[nj, ni]:
                graph.visited[nj, ni] = True
                graph.distances[nj, ni] = graph.distances[cj, ci] + 1
                graph.parents[nj, ni] = Cell(ci, cj)
                stack.append((ni, nj))

    # No path found
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
    """A* Search algorithm."""
    graph.init_graph()

    si, sj = start.i, start.j
    gi, gj = goal.i, goal.j

    # Heuristic: Manhattan distance on grid indices
    def heuristic(i, j):
        return abs(i - gi) + abs(j - gj)

    height, width = graph.distances.shape

    # g_cost: cost from start to this cell
    g_cost = np.full((height, width), np.inf, dtype=float)

    # Priority queue elements: (f_cost, g_cost, i, j)
    pq = []

    # Initialize start node
    g_cost[sj, si] = 0.0
    f_start = heuristic(si, sj)
    heapq.heappush(pq, (f_start, 0.0, si, sj))

    graph.visited[sj, si] = False  # we’ll use visited as a "closed set" once expanded
    graph.distances[sj, si] = 0.0
    graph.parents[sj, si] = None

    while pq:
        f_current, g_current, ci, cj = heapq.heappop(pq)

        # If we've already closed this node, skip
        if graph.visited[cj, ci]:
            continue

        # Mark as closed/expanded
        graph.visited[cj, ci] = True

        # Record for visualization
        graph.visited_cells.append(Cell(ci, cj))

        # Goal check
        if ci == gi and cj == gj:
            # distances can store the final g-cost if desired
            graph.distances[cj, ci] = g_current
            return trace_path(Cell(gi, gj), graph)

        # Explore neighbors
        for (ni, nj) in graph.find_neighbors(ci, cj):
            if graph.visited[nj, ni]:
                # Already expanded (in closed set)
                continue

            tentative_g = g_current + 1.0  # uniform cost: each step = 1

            if tentative_g < g_cost[nj, ni]:
                # Found a better path to this neighbor
                g_cost[nj, ni] = tentative_g
                graph.distances[nj, ni] = tentative_g
                graph.parents[nj, ni] = Cell(ci, cj)

                f_neighbor = tentative_g + heuristic(ni, nj)
                heapq.heappush(pq, (f_neighbor, tentative_g, ni, nj))

    # No path found
    return []
