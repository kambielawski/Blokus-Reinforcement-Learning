from Piece import Piece
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pygame

# Board - maintains representation of game board
# self.board- a numpy array representation of the board, with a 0 representing
    # an unclaimed square and an integer representing a square claimed by that player
# self.size represents board dimensions

class Board:
    # Constructor
    # size - int representing length and width of board
    def __init__(self,size):
        self.board = np.zeros([size,size])
        self.size = size
        self.num_to_color = { 0: (255,255,255),
                              1: (255,255,0),
                              2: (0,0,255),
                              3: (255,0,0),
                              4: (0,255,0) }
        # pygame render parameters
        self.clock = None
        self.window = None
        self.window_size = 512

    #check_valid_move() - returns True if move is valid for current board state, False otherwise
    # player - int from 1 to number of players
    # piece - translated Piece object
    # verbose - if True, outputs reason why move is not valid
    def check_valid_move(self, player,test_piece,verbose = False):                
       
        #verify each corner falls within bounds
        corner_adj = False
        for point in test_piece.corners:
            if point[0] < 0 or point[1] < 0 or point[0] >= self.size or point[1] >= self.size:
                if verbose:
                    print("A corner is out of bounds: {}".format(point))
                return False
            
        #verify at least one corner adjacent belongs to player or first move    
        for point in test_piece.diag_adjacents:
            #diagonal adjacency
            if (point[0] >= 0 and point[1] >= 0 and point[0] < self.size and point[1] < self.size):
                if self.board[point[0],point[1]] == player:
                    corner_adj = True
                    break
            # first move
            elif point in [(-1,-1),(-1,self.size),(self.size,-1),(self.size,self.size)]:
                corner_adj = True
                break
        if corner_adj == False: 
            if verbose:
                    print("No adjacent corners.")
            return False
        
        #verify no adjacents occupied by player
        for point in test_piece.adjacents:
            if (point[0] >= 0 and point[1] >= 0 and point[0] < self.size and point[1] < self.size):
                if self.board[point[0],point[1]] == player:
                    if verbose:
                        print("Adjacent to an existing piece: {}".format(point))
                    return False
            
        #verify no occupied spaces already occupied
        for point in test_piece.occupied:
            if self.board[point[0],point[1]] != 0:
                if verbose:
                    print("Point is already occupied: {}".format(point))
                return False
        
        return True
       
    # display() - display board as numpy array
    def display(self):
        print(self.board)

    # display2() - displays a slightly nicer representation of board
    def display2(self):
        #plt.figure()
        sns.heatmap(self.board,cmap = 'Accent', linewidths = 1, square = True,cbar = False)
        plt.show()
        plt.pause(0.01)
        
    def display_pygame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255,255,255))
        square_size = (self.window_size / self.size)
        
        # draw gridlines
        for x in range(self.size):
            pygame.draw.line(
                canvas,
                0,
                (0, square_size * x),
                (self.window_size, square_size * x),
                width=3
            )
            pygame.draw.line(
                canvas,
                0,
                (square_size * x, 0),
                (square_size * x, self.window_size),
                width=3
            )

        # draw pieces
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] > 0:
                    pygame.draw.rect(
                        canvas,
                        self.num_to_color[self.board[i][j]],
                        pygame.Rect(
                            (square_size * i, square_size * j),
                            (square_size, square_size)
                        )
                    )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(30)

    # play_piece() - update board with player's piece if valid
    # player - int from 1 to number of players
    # piece - translated Piece object
    def play_piece(self,player,piece):
        if self.check_valid_move(player,piece,verbose = True):
            for point in piece.occupied:
                self.board[point[0],point[1]] = player
            return True
        else:
            print('invalid move')
            print(piece.occupied)
            return False
        
        #self.display()
