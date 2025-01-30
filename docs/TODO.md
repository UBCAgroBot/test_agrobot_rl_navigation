# steps:

for environment, bulid up step by step:

- [x] step 1: have 2d box environment of grid (import from dict), with agent being able to be at any point and move around
- [x] step 2: have the agent train on PPO to hit some target
- [x] step 3: implement randomized environments and walls -- REVISIT
- [ ] step 4: implement physics, agent velocity, etc.
- [ ] step 5: implement display
- [ ] step 6: implement limited visual space

## note for physics:

- [ ] figure out how the pygame stuff works and how to adapt to simple environment
- [ ] figure out how to add boxes to the environment
- [ ] figure out how to make boxes inpenetrable

### notes:

- need to modify:
- create_track
- step
- reset function
- **init** function

- step 2: modify the reward function
