import logging
import math
import copy

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def selectAction(self, game):
        action_probs = self.getActionProb(game)
        # action = action_probs.index(max(action_probs))
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def getActionProb(self, game, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        print("Searching for move...")
        for _ in range(self.args['numMCTSSims']):
            self.search(game)

        s = np.array2string(game.get_canonical_board())
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(game.num_possible_moves())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            # choose among equally best actions
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    # returns: prior action probabilities (between 0 and size(action_space) - 1)
    #          value (value of the action)
    def heuristic(self, valid_moves):
        # randomly choose an action that maximizes the size of the piece placed
        piece_sizes = [self.game.piece_sizes[a[1]] for a in valid_moves]
        max_pieces = [1 if a == max(piece_sizes) else 0 for a in piece_sizes]
        # normalize
        prior_action_probs = [a / sum(max_pieces) for a in max_pieces]

        return prior_action_probs, max(piece_sizes)

    def search(self, game):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = np.array2string(game.get_canonical_board())

        # Check if game has ended given this board
        if s not in self.Es:
            self.Es[s] = game.get_game_ended()

        # Return -1 if P1&P3 has won, 1 if P2&P4 has won
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        # leaf node (non-terminal for now) (EXPAND)       
        if s not in self.Ps:
            valids = game.get_valid_moves()
            if len(valids) == 0:
                self.Vs[s] = []
                self.Ns[s] = 0
                return 0
            self.Ps[s], v = self.heuristic(valids) # self.nnet.predict(canonicalBoard)
            
            # eventually this will pass all the board states after the valid moves 
            # into a neural net to have it return a prior probability for each of 
            # the moves. 

            # Pretty sure we don't have to worry about this because self.heuristic()
            # is always valid and normalized
            '''
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            '''

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        if len(valids) == 0:
            return 0
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(len(valids)):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)])
            else:
                u = self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        game_after_move = copy.deepcopy(game)
        game_after_move.make_move(valids[a])

        # next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        # next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(game_after_move)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
