from collections import defaultdict
import math
from util.maze_helpers import GridTile, find_unique_item, check_in_bounds



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


def _search(maze: list[list[int]]) -> list[list[tuple[int, int] | None]]:
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
    parent: list[list[tuple[int, int] | None]] = [[None for _ in range(len(maze[0]))] for _ in range(len(maze))]

    start_x, start_y = find_unique_item(maze, GridTile.ROBOT.value)
    to_search.add((start_x, start_y))

    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while len(to_search):
        x, y = -1, -1
        f_score = 1e12
        for qx, qy in to_search:
            h_value = _get_h_value((qx, qy), maze)
            new_f_score = h_value + g_score[(qx, qy)] 
            if new_f_score < f_score or (new_f_score == new_f_score and h_value < _get_h_value((x, y), maze)):
                x, y = qx, qy
                f_score = new_f_score

        to_search.remove((x, y))
        visited.add((x, y))
        for dir in dirs:
            nx, ny = x + dir[0], y + dir[1]
            if (
                not check_in_bounds((nx, ny), (len(maze), len(maze[0])))
                or maze[nx][ny] == GridTile.WALL.value
                or (nx, ny) in visited
            ):
                continue

            in_search = (nx, ny) in visited
            cost_to_neighbor = g_score[(x, y)] + (1 if dir[0] * dir[1] == 0 else 1.4)

            if not in_search or cost_to_neighbor < g_score[(nx, ny)]:
                g_score[(nx, ny)] = cost_to_neighbor
                parent[nx][ny] = (x, y)

                if not in_search:
                    to_search.add((nx, ny))
    return parent


def _backtrack(parent: list[list[tuple[int | None, int | None]]], start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    path = []
    current = end
    while current != start:
        if current is None:
            break
        path.append(current)
        current = parent[current[0]][current[1]]
    path.append(start)
    return path[::-1]


def _get_h_value(node: tuple[int, int], maze: list[list[int]]) -> float:
    x, y = node
    ix, iy = find_unique_item(maze, 2)
    return math.sqrt((x - ix) ** 2 + (y - iy) ** 2)