import gym
import random
import numpy as np
from BlokusEnv_gym import BlokusEnv

env = BlokusEnv()
env.reset()

NUM_GAMES = 2
while NUM_GAMES > 0:
    # generate an action using RL agent 
    action = {1:1}
    random.seed()

    # advance the game
    # pass action into step()
    observation, reward, done, info = env.step(action) 

    # render the game
    env.render()

    if done:
        NUM_GAMES -= 1
        env.reset()

env.pause()
env.close()
