# steps:

for environment, bulid up step by step:

- [x] step 1: have 2d box environment of grid (import from dict), with agent being able to be at any point and move around
- [x] step 2: have the agent train on PPO to hit some target
- [x] step 3: implement randomized environments and walls -- REVISIT
- [x] step 4: implement physics, agent velocity, etc.
- [x] step 5: implement display
- [x] step 6: implement A\* pathfinding

# next steps:

- [ ] refactoring + mypy + docstrings
- [ ] need to figure out how to render path without accounting for updating the car position
- [ ] need to figure out how to continuously update car position
- [ ] need to pixelize the map more -- make blocks smaller
- [ ] bug: fix A\* pathfinding, sometimes will be diagonal when straight is better
- [ ] bug: fix A\* pathfinding, sometimes will suggest going through wall
- [ ] bug: fix A\* rendering, sometimes it will disappear
- [ ] improvement: make map generation better
