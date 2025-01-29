from enum import Enum
import random
from envs import EXAMPLE_ENV
import numpy as np


class RobotActionSpace(Enum):
    NORTH = (-1, 0)
    EAST = (0, 1)
    SOUTH = (1, 0)
    WEST = (0, -1)


class GridTile(Enum):
    FLOOR = 0
    ROBOT = 1
    TARGET = 2
    WALL = 3


def _import_envs(env: list[list[int]]) -> np.ndarray:
    new_env: list[list[GridTile]] = [[] for _ in range(len(env))]
    for i in range(len(env)):
        new_env[i] = [GridTile(j) for j in env[i]]
    return np.array(new_env)


class GridEnv:
    def __init__(self, env: np.ndarray = _import_envs(EXAMPLE_ENV)) -> None:
        self.env_orig = env
        self.length = len(env)
        self.width = len(env[0])
        self.env_space: np.ndarray
        self.robot_pos: tuple
        self.target_pos: tuple
        self.reset()

    def move(self, action: RobotActionSpace) -> bool:
        new_robot_pos: tuple = tuple(
            a + b for a, b in zip(self.robot_pos, action.value)
        )
        if (
            self._is_in_bounds(new_robot_pos)
            and self.env_space[new_robot_pos] == GridTile.FLOOR
        ):
            self.env_space[new_robot_pos] = GridTile.ROBOT
            self.env_space[self.robot_pos] = GridTile.FLOOR
            self.robot_pos = new_robot_pos
            return True
        return False

    def reset(self) -> None:
        self.env_space = self.env_orig
        self.robot_pos = self._find_item(GridTile.ROBOT)
        self.target_pos = self._find_item(GridTile.TARGET)

    def render(self) -> None:
        for row in self.env_space:
            print("".join([str(tile.value) for tile in row]))

    def is_won(self) -> bool:
        return GridTile.TARGET not in self.env_space

    def _is_in_bounds(self, point: tuple[int, int]) -> bool:
        x, y = point
        return 0 <= x < self.length and 0 <= y < self.width

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
