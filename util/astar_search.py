from collections import defaultdict
import math


def astar_pathfinding(maze: list[list[int]]) -> list[tuple[int, int]]:
    start_x, start_y = _find_item(1, maze)
    end_x, end_y = _find_item(2, maze)
    parent = _search(maze)
    return _backtrack(parent, (start_x, start_y), (end_x, end_y))


def _search(maze: list[list[int]]) -> list[list[tuple[int, int]]]:
    # G is the cost to get to the node
    # H is the heuristic cost to get to the goal

    def _in_bounds(x: int, y: int) -> bool:
        return 0 <= x < len(maze) and 0 <= y < len(maze[0])

    def _is_walkable(x: int, y: int) -> bool:
        return maze[x][y] != 3

    to_search = set()
    visited = set()
    g_score = defaultdict(lambda: 1e10)
    parent = [[None for _ in range(len(maze[0]))] for _ in range(len(maze))]

    start_x, start_y = _find_item(1, maze)
    to_search.add((start_x, start_y))

    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while len(to_search):
        x, y = None, None
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
                not _in_bounds(nx, ny)
                or not _is_walkable(nx, ny)
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


def _backtrack(parent: list[list[tuple[int, int]]], start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
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
    ix, iy = _find_item(2, maze)
    return math.sqrt((x - ix) ** 2 + (y - iy) ** 2)


def _find_item(value: int, maze: list[list[int]]) -> tuple[int, int]:
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == value:
                return (i, j)
    raise AssertionError("Item Not Found")