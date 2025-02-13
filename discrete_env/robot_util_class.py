from enum import Enum

import numpy as np


class RobotActionSpace(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class GridTile(Enum):
    FLOOR = 0
    ROBOT = 1
    TARGET = 2
    WALL = 3


_action_space_to_tuple_dict: dict[int, tuple[int, int]] = {
    RobotActionSpace.NORTH.value: (-1, 0),
    RobotActionSpace.EAST.value: (0, 1),
    RobotActionSpace.SOUTH.value: (1, 0),
    RobotActionSpace.WEST.value: (0, -1),
}


def _action_space_to_tuple(x: int) -> tuple[int, int]:
    return _action_space_to_tuple_dict[x]


_action_space_to_tuple_vec = np.vectorize(_action_space_to_tuple)


def _import_envs(env: list[list[int]]) -> np.ndarray:
    new_env: list[list[GridTile]] = [[] for _ in range(len(env))]
    for i in range(len(env)):
        new_env[i] = [GridTile(j) for j in env[i]]
    return np.array(new_env)
