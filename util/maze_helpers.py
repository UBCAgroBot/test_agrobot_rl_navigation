from enum import Enum


class GridTile(Enum):
    FLOOR = 0
    ROBOT = 1
    TARGET = 2
    WALL = 3



def find_unique_item(maze_layout: list[list[int]], item_to_find: int) -> tuple[int, int]:
    """Locate the coordinates of a specified item within the maze.

    Raises:
        ValueError: If the item is not found in the maze.
    """
    for i, row in enumerate(maze_layout):
        for j, element in enumerate(row):
            if element == item_to_find:
                return (i, j)
    raise ValueError(f"Item {item_to_find} not found in maze.")


def check_in_bounds(coords: tuple[int, int], bounds: tuple[int, int]) -> bool:
    """Checks if the given coordinates are within the specified bounds.

    Args:
        coords (tuple[int, int]): The coordinates to check.
        bounds (tuple[int, int]): The bounds to check against as (width, height).

    Returns:
        bool: True if the coordinates are within bounds, False otherwise.
    """
    cx, cy = coords
    x, y = bounds
    return 0 <= cx < x and 0 <= cy < y