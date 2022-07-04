import gym
import numpy as np
from BlokusEnv_gym import BlokusEnv



env = BlokusEnv()
env.reset()

for _ in range(1):
    observation, reward, done, info = env.step(''' Put action here ''')
    env.render()
    if done:
        env.reset()
    
env.close()
