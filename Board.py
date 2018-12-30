from Piece import Piece
from Player import Player
from Game import Game
import numpy as np
import copy

# Board class has board and size

class Board:
    def __init__(self,size):
        self.board = np.zeros([size,size])
        self.size = size

# moves are expressed as a translated (player,piece_num,orientation,translation (x,y))
    def check_valid_move(self, player, piece_num,orientation,shift):
        #make copy of piece by querying player's piecelist and translating
        test_piece = [] #change later
        
        #verify each corner falls within bounds
        #verify at least one corner adjacent belongs to player or first move
        corner_adj = False
        for point in test_piece.corners:
            if point[0] < 0 or point[1] < 0 or point[0] >= self.size or point[1] >= self.size:
                return False
            else:
                if self.board[point[0],point[1]] == player or (point[0] == 0 and point[1] == 0):
                    corner_adj = True
        if corner_adj == False: return False
        
        #verify no adjacents occupied by player
        for point in test_piece.adjacents:
            if self.board[point[0],point[1]] == player:
                return False
            
        #verify no occupied spaces already occupied
        for point in test_piece.occupied:
            if self.board[point[0],point[1]] != 0:
                return False
        
        return True
       
    # display board
    def display(self):
        print(self.board)

    # display fancy
    def display_fance(self):
        print("Sorry Dude, that function doesn't exist yet.")
        
    # play_piece
    def play_piece(self,player,piece_num,orientation,shift):
        #get piece
        piece = []
        for point in piece.occupied:
            self.board[point[0],point[1]] = player
    