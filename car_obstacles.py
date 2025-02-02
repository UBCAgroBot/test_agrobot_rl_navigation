__credits__ = ["Andrea PIERRÉ"]

import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from car_dynamics import Car

import Box2D
from Box2D.b2 import contactListener, fixtureDef, polygonShape
import pygame
from pygame import gfxdraw


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800
TILE_DIMS = 5

SCALE = 20.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 0.4  # Camera zoom


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        ret = self._identify_contact_objs(contact)
        if ret:
            obj, tile = ret
            obj.tiles.add(tile)
            if tile.is_end:
                self.env.reward += 1000.0
                self.env.reached_reward = True

    def EndContact(self, contact):
        ret = self._identify_contact_objs(contact)
        if ret:
            obj, tile = ret
            obj.tiles.remove(tile)

    def _identify_contact_objs(self, contact):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if not u1 or not u2:
            return False

        if "road_friction" in u1.__dict__ and "tiles" in u2.__dict__:
            tile = u1
            obj = u2
        elif "road_friction" in u2.__dict__ and "tiles" in u1.__dict__:
            tile = u2
            obj = u1
        else:
            return False
        return obj, tile


class CarRacing(gym.Env):
    metadata = {
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
    ):
        self.render_mode = render_mode
        self.verbose = verbose
        self.reward = 0.0
        self.robot: Optional[Car] = None
        self.obstacles = []
        self.target = None
        self.is_open = True
        self.clock = None
        self.surf = None
        self.screen: Optional[pygame.Surface] = None

        self.world = Box2D.b2World((0, 0), contactListener=FrictionDetector(self))
        self.reached_reward = False
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
        )  # steer, gas, brake
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )
        self._init_colors()

    def _destroy(self):
        if not self.obstacles:
            return
        for t in self.obstacles:
            self.world.DestroyBody(t)
        self.world.DestroyBody(self.target)
        self.obstacles = []
        self.target = None
        assert self.robot is not None
        self.robot.destroy()

    def _init_colors(self):
        self.obs_color = np.array([102, 102, 102])
        self.end_color = np.array([20, 20, 192])
        self.bg_color = np.array([102, 204, 102])
        self.grass_color = np.array([102, 230, 102])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self._initialize_contact_listener()
        self._reset_environment()

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def _initialize_contact_listener(self):
        self.world.contactListener_bug_workaround = FrictionDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround

    def _reset_environment(self):
        self.reward = 0.0
        self.t = 0.0
        self.reached_reward = False
        self.obstacles_poly = []
        self.obstacles = []

        obj, obj_poly = self._get_tile(55, 80)
        self.obstacles.append(obj)
        self.obstacles_poly.append(obj_poly)

        end, end_poly = self._get_tile(20, 20, is_end=True)
        self.target = end
        self.obstacles_poly.append(end_poly)
        self.robot = Car(self.world, 0, 0, 0)

    def _get_tile(self, x: int, y: int, is_end: bool = False):
        t = self.world.CreateStaticBody(position=(x, y))
        t.userData = t
        t.is_end = is_end
        t.road_friction = 1.0
        t.color = self.end_color if is_end else self.obs_color
        t.CreateFixture(
            fixtureDef(shape=polygonShape(box=(TILE_DIMS, TILE_DIMS)), isSensor=is_end)
        )

        vertices = [
            (x - TILE_DIMS, y - TILE_DIMS),
            (x + TILE_DIMS, y - TILE_DIMS),
            (x + TILE_DIMS, y + TILE_DIMS),
            (x - TILE_DIMS, y + TILE_DIMS),
        ]
        poly_info = (vertices, t.color)
        return t, poly_info

    def step(self, action: Union[np.ndarray, int]):
        assert self.robot is not None
        if action is not None:
            action = action.astype(np.float64)
            self.robot.steer(-action[0])
            self.robot.gas(action[1])
            self.robot.brake(action[2])

        self.robot.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        self.state = self._render("state_pixels")

        step_reward = 0
        terminated = False
        truncated = False
        info = {}
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.robot.fuel_spent = 0.0
            if self.reached_reward:
                terminated = True
            x, y = self.robot.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.robot is not None
        # computing transformations
        angle = -self.robot.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.robot.hull.position[0]) * zoom
        scroll_y = -(self.robot.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
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
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
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
        else:
            return self.isopen

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw road
        for poly, color in self.obstacles_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
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
        def render_if_min(value, points, color):
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
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
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
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = CarRacing(render_mode="human")

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
                break
    env.close()
