import gym
import numpy as np
from BlokusEnv_gym import BlokusEnv

env = BlokusEnv()
env.reset()

for _ in range(1):
    # generate an action using RL agent 
    action = {1:1}

    # advance the game
    observation, reward, done, info = env.step(action) # pass action into step()
    env.render()

    if done:
        env.reset()
    
env.close()
