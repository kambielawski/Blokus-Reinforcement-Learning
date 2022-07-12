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
        '''
        self.action_space = spaces.Dict({
            "acting_player": spaces.MultiBinary(1), # 0 or 1 to identify player
            "piece": ,
            "position": ,
            # orientation? 
        })
        '''
        # self-play wrapper only handles Discrete action spaces;
        # could either fork the self-play wrapper and implement Tuple action spaces
        # or convert below into a Discrete action space
        self.action_space = spaces.Tuple((
            spaces.Discrete(1),  # ? 
            spaces.Discrete(21), # piece number
            spaces.Discrete(4),  # piece orientation 
            spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3))), # piece flip axes
        ))

        # variables for self-play wrapper
        self.n_players = 2
        self.current_player_num = 1

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

    '''
    The observation function returns a numpy array that can be fed as input 
    to the PPO policy network. It should return a numeric representation of 
    the current game state, from the perspective of the current player, 
    where each element of the array is in the range [-1,1].
    '''
    def observation(self):
        pass

    '''
    The legal_actions function returns a numpy vector of the same length 
    as the action space, where 1 indicates that the action is valid and 0 
    indicates that the action is invalid.
    '''
    def legal_actions(self):
        pass

    def pause(self):
        pygame.time.wait(10000)

    def render(self):
        self.game.game_board.display_pygame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

