import gym
import numpy as np
from gym import spaces

from Blokus import Blokus
from DisplayEngine import DisplayEngine

class BlokusEnv(gym.Env):
    def __init__(self):
        # representation of board
        self.observation_space = spaces.Box(low=0, high=3, shape=(20,20), dtype=int)

        # representation of RL agent's possible moves
        self.action_space = spaces.Tuple((
            spaces.Discrete(4),  # player/color number
            spaces.Discrete(21), # piece number
            spaces.Discrete(8),  # piece orientation 
            spaces.Tuple((spaces.Discrete(20), spaces.Discrete(20))), # piece location
        ))

        self.display = DisplayEngine()

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        # initialize new game
        self.game = Blokus(5,4,20)

    def step(self, action=None):
        observation = 0
        reward = 0
        info = 0

        # Run one iteration of the game
        done = self.game.step(action)

        return observation, reward, done, info

    def pause(self):
        self.display.pause()

    def render(self):
        self.display.update(self.game.game_board)

    def display_score(self):
        scores = self.game.score()
        bgscore = scores[1] + scores[3]
        ryscore = scores[0] + scores[2]
        bgleft = 178-bgscore
        ryleft = 178-ryscore
        print('Blue & Green: {bg} ({bgleft} left)\nRed & Yellow: {ry} ({ryleft} left)\n'
                .format(bg=bgscore, ry=ryscore, bgleft=bgleft, ryleft=ryleft))

    def close(self):
        self.display.end()

