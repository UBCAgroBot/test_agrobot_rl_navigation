# notes:

needs to be 2D, assume I can see the whole state
uses A\* and uses that as part of the state  
need some way to generate maze and moving objects
use openAI gym to create environment

- figure out how to create environment in gym
- figure out how to use PPO
- figure out how implement A\*

# steps:

for environment, bulid up step by step:

- [x] step 1: have 2d box environment of grid (import from dict), with agent being able to be at any point and move around
- [x] step 2: have the agent train on PPO to hit some target
- [ ] step 3: implement randomized environments and walls
- [ ] step 4: implement physics, agent velocity, etc.
- [ ] step 5: implement display
- [ ] step 6: implement limited visual space
