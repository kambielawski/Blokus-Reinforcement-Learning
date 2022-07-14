import pygame

class DisplayEngine():
    def __init__(self, window_size=512):
        self.clock = None
        # display variables
        self.window = None
        self.window_size = window_size
        self.num_to_color = { 0: (255,255,255),
                              1: (255,255,0),
                              2: (0,0,255),
                              3: (255,0,0),
                              4: (0,255,0) }

    def update(self, board):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255,255,255))
        square_size = (self.window_size / board.size)
        
        # draw gridlines
        for x in range(board.size):
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
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] > 0:
                    pygame.draw.rect(
                        canvas,
                        self.num_to_color[board.board[i][j]],
                        pygame.Rect(
                            (square_size * i, square_size * j),
                            (square_size, square_size)
                        )
                    )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(30)

    def end(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def pause(self):
        pygame.time.wait(10000)