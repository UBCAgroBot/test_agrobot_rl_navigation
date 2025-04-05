from typing import Any, Optional

import numpy as np
import pygame
from gymnasium import spaces
from numpy.typing import NDArray

from continuous_env.robot_obstacles import RobotObstacles


class DiscreteRobotObstacles(RobotObstacles):
    """Robot obstacles environment with discrete action space"""

    def __init__(
        self,
        render_mode: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(render_mode=render_mode, verbose=verbose)

        self.action_space = spaces.Discrete(6)
        self.action_map = {
            0: np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # steer left
            1: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # steer right
            2: np.array([0.0, 1.0, 0.2, 0.0], dtype=np.float32),  # accelerate
            3: np.array([0.0, 0.0, 0.8, 0.0], dtype=np.float32),  # brake
            4: np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32),  # go back
            5: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # do nothing
        }

    def step(
        self, action: Optional[int] = None
    ) -> tuple[NDArray[np.float32], int, bool, bool, dict[str, Any]]:
        """Override step to convert discrete action to continuous action"""
        if action is not None:
            continuous_action = self.action_map[action]
            return super().step(continuous_action)
        return super().step(None)


def main() -> None:
    action = 5
    quit: bool = False
    restart: bool = False

    def register_input() -> None:
        nonlocal quit, restart, action
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0 or event.key == pygame.K_KP0:
                    action = 0  # steer left
                if event.key == pygame.K_1 or event.key == pygame.K_KP1:
                    action = 1  # steer right
                if event.key == pygame.K_2 or event.key == pygame.K_KP2:
                    action = 2  # accelerate
                if event.key == pygame.K_3 or event.key == pygame.K_KP3:
                    action = 3  # brake
                if event.key == pygame.K_4 or event.key == pygame.K_KP5:
                    action = 4  # go back
                if event.key == pygame.K_5 or event.key == pygame.K_KP4:
                    action = 5  # do nothing

                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.QUIT:
                quit = True

    env = DiscreteRobotObstacles(render_mode="human")

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(action)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                action_description = list(env.action_map.keys())[action]
                continuous_action = env.action_map[action]
                print(f"\naction {action}: {[f'{x:+0.2f}' for x in continuous_action]}")
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                quit = True
                break
    env.close()


if __name__ == "__main__":
    main()
