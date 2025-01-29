import numpy as np
import gymnasium as gym
from typing import Optional
from robot_obstacles_class import RobotActionSpace, GridEnv, GridTile

gym.register(
    id="RobotObstacleEnv-v0",
    entry_point="robot_obstacles_env:RobotObstacleEnv",
)


class RobotObstacleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, discount=0.95, render_mode=None) -> None:
        self.robot_grid_env = GridEnv()
        self.render_mode = render_mode
        self.steps: int = 0
        self.discount: float = discount

        self.action_space = gym.spaces.Discrete(len(RobotActionSpace))
        self.observation_space = gym.spaces.Dict(
            {
                "space": gym.spaces.Box(
                    low=0,
                    high=3,
                    shape=(self.robot_grid_env.length, self.robot_grid_env.width),
                    dtype=np.uint8,
                ),
                "agent": gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array(
                        [self.robot_grid_env.length, self.robot_grid_env.width]
                    ),
                    shape=(2,),
                    dtype=np.uint8,
                ),
            }
        )

    def _get_obs(self) -> dict:
        def _from_grid_tile(x: GridTile) -> np.uint8:
            return np.uint8(x.value)

        _from_grid_tile_vec = np.vectorize(_from_grid_tile)
        return {
            "space": _from_grid_tile_vec(self.robot_grid_env.env_space),
            "agent": np.array(self.robot_grid_env.robot_pos, dtype=np.uint8),
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.robot_grid_env.reset()
        self.steps = 0

        obs: dict = self._get_obs()
        info: dict = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: RobotActionSpace) -> tuple[dict, float, bool, bool, dict]:
        self.robot_grid_env.move(action=action)

        reward = 0
        terminated = False
        truncated = False
        info: dict = {}
        obs: dict = self._get_obs()

        self.steps += 1
        if self.robot_grid_env.is_won():
            reward = float(np.pow(self.discount, self.steps))
            terminated = True

        if self.render_mode == "human":
            print(action)
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

if __name__ == "__main__":
    env = gym.make("RobotObstacleEnv-v0", render_mode="human")
    env.reset()

    for tstep in range(20):
        # rand_action = env.action_space.sample()
        rand_action = int(input())

        obs, reward, terminated, _, _ = env.step(_action_to_directions[rand_action])
        env.render()

        if terminated:
            obs, _ = env.reset()
