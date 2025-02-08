import random
import bisect
from discrete_env.robot_util_class import GridTile
from typing import Any
from util.maze_helpers import find_unique_item, check_in_bounds
import copy


DIRS: list[tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def maze_generator(
    dims: tuple[int, int], min_dist: int = 3, depth: int = 3, smoothness: int = 1
) -> list[list[int]]:
    """Generates a maze based on the specified dimensions and parameters.

    Args:
        dims (tuple[int, int]): The dimensions of the maze as (width, height).
        min_dist (int, optional): The minimum distance between paths. Defaults to 3.
        depth (int, optional): The depth of the maze generation algorithm. Defaults to 3.
        smoothness (int, optional): The smoothness of the maze paths. Defaults to 1.

    Returns:
        list[list[int]]: A 2D list representing the generated maze, where 
                          different integers represent different types of tiles.
    """
    maze = _maze_generator_attempt(dims, min_dist, depth, smoothness)
    while _verify_maze(maze):
        maze = _maze_generator_attempt(dims, min_dist, depth, smoothness)
    return maze


def _verify_maze(maze: list[list[int]]) -> bool:
    """Verifies if the maze has a valid path from the robot to the target.

    Args:
        maze (list[list[int]]): A 2D list representing the maze, where 
                                 different integers represent different types of tiles.

    Returns:
        bool: True if there is a valid path from the robot to the target, 
              False otherwise.
    """
    temp_maze = copy.deepcopy(maze)

    def _dfs(coords: tuple[int, int]) -> bool:
        """Performs a depth-first search to find a path to the target.

        Args:
            coords (tuple[int, int]): The current coordinates in the maze.

        Returns:
            bool: True if the target is reached, False otherwise.
        """
        curr_x, curr_y = coords
        if temp_maze[curr_x][curr_y] == GridTile.TARGET.value:
            return True
        temp_maze[curr_x][curr_y] = GridTile.WALL.value
        for dx, dy in DIRS:
            is_possible = False
            if (
                check_in_bounds((curr_x + dx, curr_y + dy), (len(maze), len(maze[0])))
                and temp_maze[curr_x + dx][curr_y + dy] != GridTile.WALL.value
            ):
                is_possible |= _dfs((curr_x + dx, curr_y + dy))
        return is_possible

    item_position = find_unique_item(temp_maze, GridTile.ROBOT.value)
    if item_position is None:
        return False

    return _dfs(item_position)


def _maze_generator_attempt(
    dims: tuple[int, int], min_dist: int = 5, depth: int = 5, smoothness: int = 1
) -> list[list[int]]:
    """Generates a maze using a depth-first search algorithm.

    Args:
        dims (tuple[int, int]): Dimensions of the maze (width, height).
        min_dist (int): Minimum distance between paths.
        depth (int): Depth of the search.
        smoothness (int): Number of smoothing iterations.

    Returns:
        list[list[int]]: 2D list representing the generated maze.
    """
    x, y = dims
    maze: list[list[int]] = [[GridTile.WALL.value] * y for _ in range(x)]

    def _dfs(coords: tuple[int, int]) -> None:
        """Performs depth-first search to carve paths in the maze."""
        nonlocal depth
        if depth <= 0:
            return

        curx, cury = coords
        valid_dirs: list[dict[str, Any]] = []
        for nx, ny in DIRS:
            min_indx: int = bisect.bisect(
                [
                    not check_in_bounds((curx + i * nx, cury + i * ny), (x, y))
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
        """Places a random tile of the specified value in the maze."""
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
    """
    Smooths the given maze by applying a specified number of smoothing iterations.

    The smoothing process involves checking the neighboring tiles of each tile in the maze.
    If the number of neighboring tiles that are marked as FLOOR is greater than or equal to
    the specified threshold (neighs), the current tile is set to FLOOR. Otherwise, it retains
    its original value.

    Parameters:
    - maze (list[list[int]]): A 2D list representing the maze, where each tile is an integer
      corresponding to its type (e.g., WALL or FLOOR).
    - smoothness (int): The number of smoothing iterations to apply to the maze.
    - neighs (int, optional): The threshold for the number of FLOOR neighbors required to
      change a tile to FLOOR. Default is 8.

    Returns:
    - list[list[int]]: The smoothed maze as a 2D list.
    """
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
