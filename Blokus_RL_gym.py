import gym
import random
import numpy as np

from BlokusEnv_gym import BlokusEnv
from MCTS import MCTS

args = {
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,
}

env = BlokusEnv()
env.reset()
mcts = MCTS(env.game, None, args)

NUM_GAMES = 1
while NUM_GAMES > 0:
    random.seed()

    # generate an action using RL agent 
    action = env.action_space.sample()
    # action = mcts.selectAction(env.game.getCanonicalBoard())

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
