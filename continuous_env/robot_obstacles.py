from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import Box2D
import gymnasium as gym
import numpy as np
import pygame
from Box2D.b2 import contactListener, fixtureDef, polygonShape
from Box2D.Box2D import b2Body, b2Contact
from gymnasium import spaces
from numpy import ndarray
from numpy.typing import NDArray
from pygame import gfxdraw
from pygame.surface import Surface

from continuous_env.robot_dynamics import Robot
from util.astar_search import astar_pathfinding
from util.maze_generator import maze_generator
from util.maze_helpers import find_unique_item

"""
Car racing environment adapted with maze generation and A*, originally from
OpenAI Gymnasium, heavily modified 
"""

STATE_W = 96
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800
TILE_DIMS = 20  # 20 works well

SCALE = 10.0  # 10 works well
PLAYFIELD = 2000 / SCALE
FPS = 50
ZOOM = 0.3
MAX_SHAPE_DIM = TILE_DIMS * math.sqrt(2) * ZOOM * SCALE


class FrictionDetector(contactListener):  # type: ignore
    def __init__(self, env: RobotObstacles) -> None:
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact: b2Contact) -> None:
        ret = self._identify_contact_objs(contact)
        if ret:
            obj, tile = ret
            obj.tiles.add(tile)
            if tile.is_end:
                self.env.reward += 1000.0
                self.env.reached_reward = True

    def EndContact(self, contact: b2Contact) -> None:
        ret = self._identify_contact_objs(contact)
        if ret:
            obj, tile = ret
            obj.tiles.remove(tile)

    def _identify_contact_objs(
        self, contact: b2Contact
    ) -> Optional[tuple[b2Body, b2Body]]:
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if not u1 or not u2:
            return None

        # road friction property is tile, tiles property is car
        if "road_friction" in u1.__dict__ and "tiles" in u2.__dict__:
            tile = u1
            obj = u2
        elif "road_friction" in u2.__dict__ and "tiles" in u1.__dict__:
            tile = u2
            obj = u1
        else:
            return None
        return obj, tile


class RobotObstacles(gym.Env[NDArray[np.uint8], NDArray[np.float32]]):  # type: ignore
    metadata: dict[str, Union[list[str], int]] = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.render_mode: Optional[str] = render_mode
        self.verbose: bool = verbose
        self.reward: float = 0.0
        self.robot: Optional[Robot] = None
        self.obstacles: list[Box2D.b2Body] = []
        self.target: Optional[Box2D.b2Body] = None
        self.is_open: bool = True
        self.clock: Optional[pygame.time.Clock] = None
        self.surf: Optional[pygame.Surface] = None
        self.steps: Optional[int] = None
        self.maze: Optional[list[list[int]]] = None
        self.maze_updated: bool = False
        self.path: list[tuple[int, int]] = []
        self.screen: Optional[pygame.Surface] = None

        self.world = Box2D.b2World((0, 0), contactListener=FrictionDetector(self))
        self.reached_reward: bool = False
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
        )  # steer, gas, brake, reverse
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )
        self._init_colors()

    def _destroy(self) -> None:
        if not self.obstacles:
            return
        for t in self.obstacles:
            self.world.DestroyBody(t)
        self.world.DestroyBody(self.target)
        self.obstacles = []
        self.target = None
        assert self.robot is not None
        self.robot.destroy()

    def _init_colors(self) -> None:
        self.obs_color = np.array([102, 102, 102])
        self.end_color = np.array([20, 20, 192])
        self.bg_color = np.array([102, 204, 102])
        self.grass_color = np.array([102, 230, 102])
        self.path_color = np.array([230, 230, 230])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[NDArray[np.uint8], Dict[Any, Any]]:
        super().reset(seed=seed)
        self._destroy()
        self._initialize_contact_listener()
        self._reset_environment()

        if self.render_mode == "human":
            self.render()
        return self.step()[0], {}

    def _initialize_contact_listener(self) -> None:
        self.world.contactListener_bug_workaround = FrictionDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround

    def _reset_environment(self) -> None:
        self.maze = maze_generator(
            (2 * int(PLAYFIELD / TILE_DIMS), 2 * int(PLAYFIELD / TILE_DIMS))
        )
        self.maze_updated = True

        self.reward = 0.0
        self.t = 0.0
        self.steps = 0
        self.reached_reward = False
        self.path = []
        self.obstacles_poly = []
        self.obstacles = []

        for x in range(len(self.maze)):
            for y in range(len(self.maze[0])):
                xcoord, ycoord = (
                    int(x * TILE_DIMS + TILE_DIMS / 2 - PLAYFIELD),
                    int(y * TILE_DIMS + TILE_DIMS / 2 - PLAYFIELD),
                )
                if self.maze[x][y] == 1:
                    self.robot = Robot(self.world, 0, xcoord, ycoord)
                if self.maze[x][y] == 2:
                    end, end_poly = self._get_tile(xcoord, ycoord, is_end=True)
                    self.target = end
                    self.obstacles_poly.append(end_poly)
                if self.maze[x][y] == 3:
                    obj, obj_poly = self._get_tile(xcoord, ycoord, is_end=False)
                    self.obstacles.append(obj)
                    self.obstacles_poly.append(obj_poly)

    def _get_tile(
        self, x: int, y: int, is_end: bool = False
    ) -> tuple[Box2D.Box2D.b2Body, tuple[list[tuple[float, float]], NDArray[np.uint]]]:
        t = self.world.CreateStaticBody(position=(x, y))
        t.userData = t
        t.is_end = is_end
        t.road_friction = 2.0
        t.color = self.end_color if is_end else self.obs_color
        t.CreateFixture(
            fixtureDef(
                shape=polygonShape(box=(int(TILE_DIMS / 2), int(TILE_DIMS / 2))),
                isSensor=is_end,
            )
        )

        vertices = [
            (x - TILE_DIMS / 2, y - TILE_DIMS / 2),
            (x + TILE_DIMS / 2, y - TILE_DIMS / 2),
            (x + TILE_DIMS / 2, y + TILE_DIMS / 2),
            (x - TILE_DIMS / 2, y + TILE_DIMS / 2),
        ]
        poly_info = (vertices, t.color)
        return t, poly_info

    def step(
        self, action: Optional[NDArray[np.float32]] = None
    ) -> tuple[NDArray[np.uint8], int, bool, bool, dict[str, Any]]:
        assert self.robot is not None
        assert self.maze is not None
        assert self.steps is not None

        if action is not None:
            action = action.astype(np.float64)
            self.robot.steer(-action[0])
            self.robot.gas(action[1])
            self.robot.brake(action[2])
            self.robot.reverse(action[3])

        self.robot.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        self.state = self._render("state_pixels")
        self.steps += 1

        x, y = self.robot.hull.position
        nx = int((x + PLAYFIELD - TILE_DIMS / 2) / TILE_DIMS)
        ny = int((y + PLAYFIELD - TILE_DIMS / 2) / TILE_DIMS)
        ox, oy = find_unique_item(self.maze, 1)
        self.maze[ox][oy] = 0
        self.maze[nx][ny] = 1
        if nx != ox or ny != oy:
            self.maze_updated = True

        step_reward = 0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            self.robot.fuel_spent = 0.0
            if self.reached_reward:
                step_reward += int(100000 * pow(0.99999995, self.steps))
                terminated = True
            x, y = self.robot.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return None
        else:
            self._render(self.render_mode)
            return None

    def _render(self, mode: str) -> Optional[NDArray[np.uint8]]:
        assert mode in self.metadata["render_modes"]  # type: ignore

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return None  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.robot is not None
        # computing transformations
        angle = -self.robot.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.robot.hull.position[0]) * zoom
        scroll_y = -(self.robot.hull.position[1]) * zoom
        trans_vec = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans_vec[0], WINDOW_H / 4 + trans_vec[1])

        self._render_items(zoom, trans, angle)
        try:
            self._render_pathfinding(zoom, trans, angle)
        except Exception as e:
            print("Note: exception workaround, ", e)
        self.robot.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, int(WINDOW_H - WINDOW_H * 2.5 / 40.0))
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))

        return None

    def _render_pathfinding(
        self, zoom: float, translation: Tuple[float, float], angle: float
    ) -> None:
        assert self.maze is not None
        assert self.surf is not None
        assert self.robot is not None

        def _fix_coords(coords: tuple[int, int]) -> tuple[int, int]:
            coords_vec = pygame.math.Vector2(coords).rotate_rad(angle)
            coords = (
                int(coords_vec[0] * zoom + translation[0]),
                int(coords_vec[1] * zoom + translation[1]),
            )
            return coords[0], coords[1]

        def _get_center(x: int, y: int) -> tuple[int, int]:
            xcoord, ycoord = (
                int(x * TILE_DIMS + TILE_DIMS / 2) - PLAYFIELD,
                int(y * TILE_DIMS + TILE_DIMS / 2) - PLAYFIELD,
            )
            return int(xcoord), int(ycoord)

        if self.maze_updated:
            self.maze_updated = False
            self.path = astar_pathfinding(self.maze)
        render_path = [_fix_coords(_get_center(*x)) for x in self.path]

        robot_x, robot_y = _fix_coords(
            (int(self.robot.hull.position[0]), (self.robot.hull.position[1]))
        )
        prev = (robot_x, robot_y)
        for curr in render_path:
            px, py = prev
            cx, cy = curr
            gfxdraw.line(self.surf, px, py, cx, cy, self.path_color)
            prev = curr

    def _render_items(
        self, zoom: float, translation: tuple[float, float], angle: float
    ) -> None:
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        for poly, color in self.obstacles_poly:
            ret_poly = [(float(p[0]), float(p[1])) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(
                self.surf, ret_poly, color, zoom, translation, angle
            )

    def _render_indicators(self, W: int, H: int) -> None:
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place: float, val: float) -> list[tuple[float, float]]:
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place: float, val: float) -> list[tuple[float, float]]:
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.robot is not None
        true_speed = np.sqrt(
            np.square(self.robot.hull.linearVelocity[0])
            + np.square(self.robot.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(
            value: float, points: list[tuple[float, float]], color: tuple[int, int, int]
        ) -> None:
            assert self.surf is not None
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.robot.wheels[0].omega,
            vertical_ind(7, 0.01 * self.robot.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.robot.wheels[1].omega,
            vertical_ind(8, 0.01 * self.robot.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.robot.wheels[2].omega,
            vertical_ind(9, 0.01 * self.robot.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.robot.wheels[3].omega,
            vertical_ind(10, 0.01 * self.robot.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.robot.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.robot.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.robot.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.robot.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self,
        surface: Optional[Surface],
        poly: list[tuple[float, float]],
        color: Union[ndarray, list[int]],
        zoom: float,
        translation: tuple[float, float],
        angle: float,
        clip: bool = True,
    ) -> None:
        assert self.surf is not None
        poly_vec = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1])
            for c in poly_vec
        ]
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen: Surface, size: Tuple[int, int]) -> ndarray:
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self) -> None:
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


def main() -> None:
    a = np.array([0.0, 0.0, 0.0, 0.0])
    quit: bool = False
    restart: bool = False

    def register_input() -> None:
        nonlocal quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_LSHIFT:
                    a[2] = +0.8
                if event.key == pygame.K_DOWN:
                    a[3] = +0.3
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_LSHIFT:
                    a[2] = 0
                if event.key == pygame.K_DOWN:
                    a[3] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = RobotObstacles(render_mode="human")

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                quit = True
                break
    env.close()


if __name__ == "__main__":
    main()
    # cProfile.run("main()", "profile_output")

    # stats = pstats.Stats("profile_output")
    # stats.sort_stats("tottime").print_stats()
