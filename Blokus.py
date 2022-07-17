from Piece import Piece
from Board import Board
from Player import Player
import numpy as np
import pickle
import random
import copy

# Game class- manages players, turns, and board
# self.game_board - Board object representing current game state
# self.pieces - list of lists, where each sublist represents all valid orientations 
    # for one piece, stored as a list of Piece objects (common to all players, not modified during play)
# self.player_list - list of 4 Player objects, where entry i corresponds to player (i+1)
# self.turn - integer from 1 to number of players, keeping track of which player is next to go
class Blokus():
    
    # Constructor
    # piece_size - int, maximum number of squares occupied by one piece (5)
    # num_players - int
    # board_size - int (generally 14 for 2 player game, 20 for 4 player game)
    def __init__(self,piece_size,num_players,board_size):
        self.game_board = Board(board_size)
        self.turns_since_last_move = 0
        self.n_players = num_players
        
        #loads piece shapes from file  --- Need to create a new pickle since class was redefined
        f = open('blokus_pieces_lim_5.pkl', 'rb')
        all_pieces = pickle.load(f)
        f.close()
        
        self.pieces = []
        for piece in all_pieces:
            # TODO: Piece size other than 5 doesn't work                                                        
            if piece.size <= piece_size:
                # self.pieces includes all orientations
                self.pieces.append(piece.get_orientations())
        
        self.piece_sizes = [1,2,3,4,5,3,4,5,5,5,5,5,5,5,5,5,5,5,4,4,4]
        self.player_list = []
        for i in range (1,num_players+1):
            # Note from Mike: I don't think piece_size is a necessary attribute for the Player object
            new_player = Player(i,piece_size,self.game_board,self.pieces)
            self.player_list.append(new_player)
            
        self.turn = 1
    
    def step(self, action=None):
        # select player to play
        current_player = self.player_list[self.turn-1]
        
        if action:
            self.make_move(action)
        else: 
            # ask player to make move (this function updates player and self.board)
            current_player.make_move(self.game_board,self.pieces, 'random')
            
            # eventually, log each move
            
            # end game if a player has played all pieces
            for player in self.player_list:
                if np.prod(player.played) == 1:
                    done = True
                    return done
        
            # change to next player
            self.turn = self.turn % len(self.player_list) + 1

        done = all([len(player.valid_moves) == 0 for player in self.player_list])

        return done

    # run() - runs a whole game on board
    # verbose - specifies whether the board and piecelists are printed before each turn
    def run(self, verbose = True):
        # Ends game if no player can make a move
        done = False
        while not done:
            done = self.step()
            if verbose:
                print(self.turn)
                #self.game_board.display2()
                print(self.get_canonical_board())
                print("\n\n")
                self.game_board.display_pygame()

        return self.score()
    
    # score() - scores game based on number of squares occupied on board by players
    def score(self):
        scores= []
        for i in range(0,len(self.player_list)):
            scores.append(sum(sum(self.game_board.board == i+1)))    
        return scores
    
    # returns true if all players are out of possible moves
    def get_game_ended(self):
        if all([self.num_possible_moves(i) == 0 for i in range(self.n_players)]):
            scores = self.score()
            p1_score = scores[0] + scores[2]
            p2_score = scores[1] + scores[3]
            if p1_score > p2_score:
                return -1
            # TODO: tie goes to p2... ?
            else:
                return 1
        return 0

    def num_possible_moves(self, playerNum=None):
        if not playerNum:
            playerNum = self.turn
        player = self.player_list[playerNum-1]
        moves = player.make_move(self.game_board,self.pieces,"random",return_all = True)
        return len(moves)

    # return all valid moves for given player number
    def get_valid_moves(self, playerNum=None):
        if not playerNum:
            playerNum = self.turn
        player = self.player_list[playerNum-1]
        # TODO: might be optimizable, we don't need to update moves before returning them
        moves = player.make_move(self.game_board,self.pieces,"random",return_all = True)
        return moves

    def enumerate_current_moves(self):
        """
        Returns current player idx and all possible moves (as board states) for that player
        """
        # get all valid moves
        player = self.player_list[self.turn-1]
        moves = player.make_move(self.game_board,self.pieces,"random",return_all = True)
        
        all_moves = []
        
        for move in moves:
            temp_board = copy.deepcopy(self.game_board)
            temp_piece = copy.deepcopy(self.pieces[move[1]][move[2]][0])
            temp_piece.translate(move[3])
            temp_board.play_piece(self.turn,temp_piece)            
            all_moves.append(temp_board)
            
        return all_moves,moves
     
    def make_move(self,move):
        """
        Manually makes a move, updating played pieces for that player,turn, and board_state
        """
        turn = self.turn
        player = self.player_list[turn-1]
        move_return = player.make_move(self.game_board,self.pieces, 'manual',input_move = move)
        
        if move_return == False:
            print ("error")
        
        # change to next player
        self.turn = self.turn % len(self.player_list) + 1

    def get_canonical_board(self):
        cboard = self.game_board.board
        cboard = np.where(cboard == 1, 1, cboard)
        cboard = np.where(cboard == 2, -0.5, cboard)
        cboard = np.where(cboard == 3, 0.5, cboard)
        cboard = np.where(cboard == 4, -1, cboard)
        if self.turn == 1 or self.turn == 3:
            return cboard
        else:
            return -cboard

    def get_string_representation(self):
        return np.array2string(self.game_board.board)

def num_to_player(num):
    if num == 1:
        return 'blue'
    elif num == 2:
        return 'red'
    elif num == 3:
        return 'green'
    else:
        return 'yellow'

if __name__ == "__main__":
    # start of body text, used to verify code was working        
    random.seed(3)
    #import os
    #os.chdir("C:/Users/Mike/Documents/Coding Projects/Blokus/Dereks/Blokus-Reinforcement-Learning")
    game = Blokus(5,4,20)
    final_score = game.run()
    print(final_score)
    print(num_to_player(final_score.index(max(final_score))), ' wins!')

