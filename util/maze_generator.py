import random
import bisect
from discrete_env.robot_util_class import GridTile


def maze_generator(
    dims: tuple[int, int], min_dist: int = 3, depth: int = 3, smoothness: int = 1
) -> list[list[int]]:
    maze = _maze_generator_attempt(dims, min_dist, depth, smoothness)
    while _verify_maze(maze):
        maze = _maze_generator_attempt(dims, min_dist, depth, smoothness)
    return maze


def _is_in_bounds(coords: tuple[int, int], bounds: tuple[int, int]) -> bool:
    cx, cy = coords
    x, y = bounds
    return 0 <= cx < x and 0 <= cy < y


def _verify_maze(maze: list[list[int]]) -> bool:
    nmaze = [row[:] for row in maze]
    dirs: list[tuple] = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def _dfs(coords: tuple[int, int]) -> None:
        curx, cury = coords
        if nmaze[curx][cury] == GridTile.TARGET.value:
            return True
        nmaze[curx][cury] = GridTile.WALL.value
        for nx, ny in dirs:
            is_possible = False
            if (
                _is_in_bounds((curx + nx, cury + ny), (len(maze), len(maze[0])))
                and nmaze[curx + nx][cury + ny] != GridTile.WALL.value
            ):
                is_possible |= _dfs((curx + nx, cury + ny))
        return is_possible

    def _find_item(value: int) -> tuple[int, int]:
        for i in range(len(nmaze)):
            for j in range(len(nmaze[i])):
                if nmaze[i][j] == value:
                    return (i, j)
        raise AssertionError("Item Not Found")

    return _dfs(_find_item(GridTile.ROBOT.value))


def _maze_generator_attempt(
    dims: tuple[int, int], min_dist: int = 3, depth: int = 3, smoothness: int = 1
) -> list[list[int]]:
    x, y = dims
    maze: list[list[int]] = [[GridTile.WALL.value] * y for _ in range(x)]
    dirs: list[tuple] = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def _dfs(coords: tuple[int, int]) -> None:
        nonlocal depth
        if depth <= 0:
            return

        curx, cury = coords
        valid_dirs: list[dict] = []
        for nx, ny in dirs:
            min_indx: int = bisect.bisect(
                [
                    not _is_in_bounds((curx + i * nx, cury + i * ny), (x, y))
                    for i in range(1, min_dist)
                ],
                False,
            )

            if all(
                maze[curx + i * nx][cury + i * ny] == GridTile.WALL.value
                for i in range(min_indx)
            ):
                valid_dirs.append({"mndist": min_indx, "coords": (nx, ny)})

        random.shuffle(valid_dirs)
        for dir in valid_dirs:
            nx, ny = dir["coords"]
            rngdist = dir["mndist"] - random.randint(0, 1)
            for i in range(rngdist):
                maze[curx + i * nx][cury + i * ny] = GridTile.FLOOR.value
            _dfs((curx + rngdist * nx, cury + rngdist * ny))
            depth -= 1

    rx, ry = random.randint(0, x - 1), random.randint(0, y - 1)
    _dfs((rx, ry))
    maze = _maze_smoothing(maze, smoothness=smoothness)

    def _place_random_tile(tile_value: int) -> None:
        rx, ry = random.randint(0, x - 1), random.randint(0, y - 1)
        while maze[rx][ry] != GridTile.FLOOR.value:
            rx, ry = random.randint(0, x - 1), random.randint(0, y - 1)
        maze[rx][ry] = tile_value

    _place_random_tile(GridTile.ROBOT.value)
    _place_random_tile(GridTile.TARGET.value)
    return maze


def _maze_smoothing(
    maze: list[list[int]], smoothness: int, neighs: int = 8
) -> list[list[int]]:
    for _ in range(smoothness):
        new_grid = [row.copy() for row in maze]
        for x in range(1, len(maze) - 1):
            for y in range(1, len(maze[0]) - 1):
                neighbors_2d = [maze[i][y - 1 : y + 2] for i in range(x - 1, x + 2)]
                floor_count = sum(
                    row.count(GridTile.FLOOR.value) for row in neighbors_2d
                )
                new_grid[x][y] = (
                    GridTile.FLOOR.value if floor_count >= neighs else maze[x][y]
                )
        maze = new_grid
    return maze
