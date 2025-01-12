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
class Game():
    
    # Constructor
    # piece_size - int, maximum number of squares occupied by one piece (5)
    # num_players - int
    # board_size - int (generally 14 for 2 player game, 20 for 4 player game)
    def __init__(self,piece_size,num_players,board_size):
        self.game_board = Board(board_size)
        
        #loads piece shapes from file  --- Need to create a new pickle since class was redefined
        f = open('blokus_pieces_lim_5.pkl', 'rb')
        all_pieces = pickle.load(f)
        f.close()
        print(len(all_pieces))
        
        self.pieces = []
        for piece in all_pieces:
            # TODO: Piece size other than 5 doesn't work                                                        
            if piece.size <= piece_size:
                # self.pieces includes all orientations
                self.pieces.append(piece.get_orientations())
               
        self.player_list = []
        for i in range (1,num_players+1):
            # Note from Mike: I don't think piece_size is a necessary attribute for the Player object
            new_player = Player(i,piece_size,self.game_board,self.pieces)
            self.player_list.append(new_player)
            
        self.turn = 1
        
    # run() - runs a whole game on board
    # verbose - specifies whether the board and piecelists are printed before each turn
    def run(self, verbose = True):

        all_moves, moves = self.enumerate_current_moves()
        print('all moves: ', all_moves, 'moves: ', moves)
        
        # Ends game if no player can make a move
        turns_since_last_move = 0
        while turns_since_last_move <= len(self.player_list):
            if verbose:
                print(self.turn)
                #self.game_board.display2()
                print(self.player_list[0].played)
                print(self.player_list[1].played)
                #print(self.player_list[2].played)
                #print(self.player_list[3].played)
                print("\n\n")
                self.game_board.display_pygame()
                # self.game_board.display2()
            
            # select player to play
            current_player = self.player_list[self.turn-1]
            
            # ask player to make move (this function updates player and self.board)
            if self.turn == 1:
                move = current_player.make_move(self.game_board,self.pieces, 'random')
            elif self.turn == 2:
                move = current_player.make_move(self.game_board,self.pieces, 'random')
            else:
                move = current_player.make_move(self.game_board,self.pieces, 'random')
            
            # if no move could be made, increment counter by 1
            # else reset counter to 0
            if  move == False: #no move available
                turns_since_last_move = turns_since_last_move + 1
            else:
                turns_since_last_move = 0
            
            # eventually, log each move
            
            # end game if a player has played all pieces
            for player in self.player_list:
                if np.prod(player.played) == 1:
                    turns_since_last_move = len(self.player_list)
            
            # change to next player
            self.turn = self.turn % len(self.player_list) + 1

        # self.game_board.display2()
        return self.score()
    
    # score() - scores game based on number of squares occupied on board by players
    def score(self):
        scores= []
        for i in range(0,len(self.player_list)):
            scores.append(sum(sum(self.game_board.board == i+1)))    
        return scores

    def enumerate_current_moves(self):
        """
        Returns current player idx and all possible moves (as board states) for that player
        """
        # get all valid moves
        player = self.player_list[self.turn-1]
        print(self.pieces)
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

def num_to_player(num):
    if num == 0:
        return 'blue'
    elif num == 1:
        return 'red'
    elif num == 2:
        return 'green'
    else:
        return 'yellow'

if __name__ == "__main__":
    # start of body text, used to verify code was working        
    random.seed(3)
    #import os
    #os.chdir("C:/Users/Mike/Documents/Coding Projects/Blokus/Dereks/Blokus-Reinforcement-Learning")
    game = Game(5,4,20)
    final_score = game.run()
    print(final_score)
    print(final_score.index(max(final_score))+1, ' wins!')

