from enum import Enum
import random
import numpy as np
from util.maze_generator import maze_generator
from robot_util_class import (
    _import_envs,
    _action_space_to_tuple_vec,
    RobotActionSpace,
    GridTile,
)


class GridEnv:
    def __init__(self, size: int) -> None:
        self.env_orig = _import_envs(maze_generator((size, size)))
        self.size = size
        self.env_space: np.ndarray
        self.robot_pos: tuple
        self.target_pos: tuple
        self.reset()

    def move(self, action: int) -> bool:
        new_robot_pos: tuple = tuple(
            a + b for a, b in zip(self.robot_pos, _action_space_to_tuple_vec(action))
        )
        if (
            self._is_in_bounds(new_robot_pos)
            and self.env_space[new_robot_pos] != GridTile.WALL
        ):
            self.env_space[self.robot_pos] = GridTile.FLOOR
            self.env_space[new_robot_pos] = GridTile.ROBOT
            self.robot_pos = new_robot_pos
            return True
        return False

    def reset(self) -> None:
        self.env_space = _import_envs(maze_generator((self.size, self.size)))
        self.robot_pos = self._find_item(GridTile.ROBOT)
        self.target_pos = self._find_item(GridTile.TARGET)

    def render(self) -> None:
        for row in self.env_space:
            print("".join([str(tile.value) for tile in row]))

    def is_won(self) -> bool:
        return GridTile.TARGET not in self.env_space

    def _is_in_bounds(self, point: tuple[int, int]) -> bool:
        x, y = point
        return 0 <= x < self.size and 0 <= y < self.size

    def _find_item(self, item: GridTile) -> tuple[int, int]:
        pos = np.argwhere(self.env_space == item)
        if len(pos) > 0:
            return tuple(pos[0])
        raise ValueError("Item Not Found")


if __name__ == "__main__":
    env = GridEnv(env=_import_envs(EXAMPLE_ENV))
    env.render()

    for i in range(25):
        rand_action = random.choice(list(RobotActionSpace))
        print("\n", rand_action)

        env.move(rand_action)
        env.render()
