import os
import gym
import numpy as np
from BlokusEnv_gym import BlokusEnv

env = BlokusEnv()
env.reset()

for _ in range(1):
    # generate an action using RL agent 
    action = {1:1}

    # advance the game
    # pass action into step()
    observation, reward, done, info = env.step(action) 

    # render the game
    env.render()

    if done:
        env.reset()
    
# env.close()
