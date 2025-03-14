from typing import Any, Optional

import gymnasium as gym
import numpy as np
from robot_obstacles_class import GridEnv
from robot_util_class import GridTile, RobotActionSpace

gym.register(
    id="RobotObstacleEnv-v0",
    entry_point="robot_obstacles_env:RobotObstacleEnv",
)


class RobotObstacleEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self, discount: float = 0.9995, render_mode: Optional[str] = None
    ) -> None:
        self.robot_grid_env = GridEnv(size=6)
        self.render_mode = render_mode
        self.steps: int = 0
        self.discount: float = discount

        self.action_space = gym.spaces.Discrete(len(RobotActionSpace))
        self.observation_space = gym.spaces.Dict(
            {
                "space": gym.spaces.Box(
                    low=0,
                    high=3,
                    shape=(self.robot_grid_env.size, self.robot_grid_env.size),
                    dtype=np.uint8,
                ),
                "agent": gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array(
                        [self.robot_grid_env.size - 1, self.robot_grid_env.size - 1]
                    ),
                    shape=(2,),
                    dtype=np.uint8,
                ),
            }
        )

    def _get_obs(self) -> dict[str, Any]:
        def _from_grid_tile(x: GridTile) -> np.uint8:
            return np.uint8(x.value)

        _from_grid_tile_vec = np.vectorize(_from_grid_tile)
        return {
            "space": _from_grid_tile_vec(self.robot_grid_env.env_space),
            "agent": np.array(self.robot_grid_env.robot_pos, dtype=np.uint8),
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.robot_grid_env.reset()
        self.steps = 0

        obs: dict[str, Any] = self._get_obs()
        info: dict[str, Any] = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.robot_grid_env.move(action=action)

        reward: float = 0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        obs: dict[str, Any] = self._get_obs()

        self.steps += 1
        if self.robot_grid_env.is_won():
            reward = float(np.pow(self.discount, self.steps))
            terminated = True

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        self.robot_grid_env.render()


_action_to_directions = {
    0: RobotActionSpace.NORTH,
    1: RobotActionSpace.EAST,
    2: RobotActionSpace.SOUTH,
    3: RobotActionSpace.WEST,
}
