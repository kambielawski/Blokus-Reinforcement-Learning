import gym
import pygame
import numpy as np
from gym import spaces


class BlokusEnv(gym.Env):
    def __init__(self):
        # representation of board
        self.observation_space = spaces.Box(low=0, high=3, shape=(20,20), dtype=int)

        # representation of RL agent's possible moves
        '''
        self.action_space = spaces.Dict({
            "acting_player": spaces.MultiBinary(1), # 0 or 1 to identify player
            "piece": ,
            "position": ,
            # orientation? 
        })
        '''
        self.action_space = spaces.Tuple(
            spaces.Discrete(1),  # ? 
            spaces.Discrete(21), # piece number
            spaces.Discrete(4),  # piece orientation 
            spaces.Tuple(spaces.Discrete(3), spaces.Discrete(3)), # piece flip axes
        )

        # display variables
        self.window = None

    def step(self, action):
        pass

    def display(self):
        pass

    def reset(self):
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

