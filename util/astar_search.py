import heapq
from collections import defaultdict
from typing import Optional

from util.maze_helpers import GridTile, check_in_bounds, find_unique_item


def astar_pathfinding(maze: list[list[int]]) -> list[tuple[int, int]]:
    """Finds the shortest path in a maze using the A* pathfinding algorithm.

    Args:
        maze (list[list[int]]): A 2D list representing the maze, where different integers
                                 represent different types of tiles (e.g., walls, paths).

    Returns:
        list[tuple[int, int]]: A list of coordinates representing the path from the start
                                to the end, or an empty list if no path is found.
    """
    start_x, start_y = find_unique_item(maze, GridTile.ROBOT.value)
    end_x, end_y = find_unique_item(maze, GridTile.TARGET.value)

    parent = _search(maze)
    ret = _backtrack(parent, (start_x, start_y), (end_x, end_y))
    return ret


def _search(maze: list[list[int]]) -> list[list[Optional[tuple[int, int]]]]:
    """Performs the A* search algorithm to find the shortest path in the maze.

    Args:
        maze (list[list[int]]): A 2D list representing the maze, where different integers
                                 represent different types of tiles (e.g., walls, paths).

    Returns:
        list[list[tuple[int | None, int | None]]]: A 2D list where each cell contains the
                                                    coordinates of the parent node for each
                                                    position in the maze, or (None, None) if
                                                    there is no parent.
    """
    to_search = set()
    visited = set()
    g_score: defaultdict[tuple[int, int], float] = defaultdict(lambda: 1e10)
    h_score: defaultdict[tuple[int, int], float] = defaultdict(lambda: 1e10)
    parent: list[list[tuple[int, int] | None]] = [
        [None for _ in range(len(maze[0]))] for _ in range(len(maze))
    ]

    start_x, start_y = find_unique_item(maze, GridTile.ROBOT.value)
    to_search.add((start_x, start_y))
    g_score[(start_x, start_y)] = 0
    h_score[(start_x, start_y)] = _get_h_value((start_x, start_y), maze)

    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while len(to_search):
        x, y = -1, -1
        f_score = 1e12
        for qx, qy in to_search:
            new_f_score = h_score[(qx, qy)] + g_score[(qx, qy)]
            if new_f_score < f_score or (
                new_f_score == f_score and h_score[(qx, qy)] < h_score[(x, y)]
            ):
                x, y = qx, qy
                f_score = new_f_score
        assert f_score != 1e12
        assert x != -1 and y != -1

        visited.add((x, y))
        to_search.remove((x, y))

        for dir in dirs:
            nx, ny = x + dir[0], y + dir[1]
            if (
                not check_in_bounds((nx, ny), (len(maze), len(maze[0])))
                or maze[nx][ny] == GridTile.WALL.value
                or (nx, ny) in visited
            ):
                continue

            in_search = (nx, ny) in to_search
            cost_to_neighbor = g_score[(x, y)] + (10 if (dir[0] * dir[1] == 0) else 14)

            if not in_search or cost_to_neighbor < g_score[(nx, ny)]:
                g_score[(nx, ny)] = cost_to_neighbor
                parent[nx][ny] = (x, y)

                if not in_search:
                    h_score[(nx, ny)] = _get_h_value((nx, ny), maze)
                    to_search.add((nx, ny))
    return parent


def _backtrack(
    parent: list[list[Optional[tuple[int, int]]]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Reconstructs the path from the start node to the end node using the parent pointers.

    Args:
        parent (list[list[Optional[tuple[int, int]]]]): A 2D list where each element contains the parent
            coordinates of the corresponding node in the maze.
        start (tuple[int, int]): The starting coordinates of the path.
        end (tuple[int, int]): The ending coordinates of the path.

    Returns:
        list[tuple[int, int]]: A list of coordinates representing the path from start to end, in order.
    """
    path = []
    current = end
    while current != start:
        if current is None:
            break
        path.append(current)
        current = parent[current[0]][current[1]]  # type: ignore
    path.append(start)
    return path[::-1]


def _get_h_value(node: tuple[int, int], maze: list[list[int]]) -> float:
    """
    Calculates the heuristic value (H score) for a given node based on its distance to the target.

    Args:
        node (tuple[int, int]): The coordinates of the current node.
        maze (list[list[int]]): The maze represented as a 2D list.

    Returns:
        float: The heuristic value representing the estimated cost to reach the target from the current node.
    """
    x, y = node
    ix, iy = find_unique_item(maze, 2)
    dx = abs(x - ix)
    dy = abs(y - iy)
    return 10 * (dx + dy) + 4 * min(dx, dy)


def _dijkstras(maze: list[list[int]]) -> float:
    """
    Uses Dijkstra's algorithm to find the shortest path from the robot to the target in a maze.

    Args:
        maze (list[list[int]]): A 2D list representing the maze.

    Returns:
        float: The shortest distance to the target, or a large number if unreachable.
    """
    to_search: list[tuple[float, tuple[int, int]]] = []
    dist: defaultdict[tuple[int, int], float] = defaultdict(lambda: 1e10)

    start_x, start_y = find_unique_item(maze, GridTile.ROBOT.value)
    heapq.heappush(to_search, (0, (start_x, start_y)))
    dist[(start_x, start_y)] = 0

    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while len(to_search):
        last_dist, (x, y) = heapq.heappop(to_search)
        if last_dist > dist[(x, y)]:
            continue

        for dir in dirs:
            nx, ny = x + dir[0], y + dir[1]
            edge = 10 if dir[0] * dir[1] == 0 else 14
            if (
                not check_in_bounds((nx, ny), (len(maze), len(maze[0])))
                or maze[nx][ny] == GridTile.WALL.value
                or last_dist + edge >= dist[(nx, ny)]
            ):
                continue

            dist[(nx, ny)] = last_dist + edge
            heapq.heappush(to_search, (dist[(nx, ny)], (nx, ny)))

    end_x, end_y = find_unique_item(maze, GridTile.TARGET.value)
    return dist[(end_x, end_y)]
