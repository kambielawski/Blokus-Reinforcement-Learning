import random
import numpy as np

from BlokusEnv_gym import BlokusEnv
from MCTS import MCTS

args = {
    'numMCTSSims': 10,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,
}

env = BlokusEnv()
env.reset()
mcts = MCTS(env.game, None, args)

NUM_GAMES = 1
while NUM_GAMES > 0:
    random.seed()

    valid_moves = env.game.get_valid_moves()
    if len(valid_moves) == 0:
        observation, reward, done, info = env.step()
    else:
        # PURE HEURISTIC APPROACH
        action_probs, _ = mcts.heuristic(valid_moves)
        action = np.random.choice(len(action_probs), p=action_probs)

        # generate an action using MCTS
        # action = mcts.selectAction(env.game)

        # advance the game
        # pass action into step()
        observation, reward, done, info = env.step(valid_moves[action]) 

        # render the board
        env.render()

    if done:
        NUM_GAMES -= 1
        env.display_score()
        env.reset()
        mcts = MCTS(env.game, None, args) # reset search tree

env.pause()
env.close()
