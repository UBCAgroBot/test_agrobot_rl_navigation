# Car Racing Environment with Maze Generation and A\*

This repository contains a car racing environment adapted from OpenAI Gymnasium, enhanced with maze generation and A\* pathfinding. The environment simulates top-down car dynamics with realistic physics, including wheel rotation, friction, and skidding.

Credits: the continuous environment includes heavily modified files originally from OpenAI Gymnasium CarRacing Environment

## Features

- **Maze Generation**: Randomized maze environments with walls and obstacles.
- **Car Dynamics**: Realistic physics simulation for car movement, including acceleration, braking, and steering.
- **Pathfinding**: A\* algorithm for pathfinding in the maze.
- **Rendering**: Visual rendering of the environment using Pygame.

## Key Files

- **`car_obstacles.py`**: Main environment class with maze generation and car dynamics.
- **`car_dynamics.py`**: Car physics simulation, including wheel rotation and skidding.
- **`maze_generator.py`**: Maze generation logic.

## Example

```python
env = CarRacing(render_mode="human")
env.reset()
while True:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()
```
