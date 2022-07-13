import gym
import pygame
import numpy as np
from gym import spaces

from Game import Game

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

        # display variables
        self.window = None
        self.window_size = 512

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        # initialize new game
        self.game = Game(5,4,20)

    def step(self, action):
        observation = 0
        reward = 0
        info = 0

        # Run one iteration of the game
        done = self.game.step()

        return observation, reward, done, info

    def pause(self):
        pygame.time.wait(10000)

    def render(self):
        self.game.game_board.display_pygame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

