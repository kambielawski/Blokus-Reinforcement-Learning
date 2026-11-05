"""The dual-mode board symmetry, for doubling training data.

With turn order TL -> BR -> TR -> BL, the only symmetry that keeps the turn cycle
is the left-right mirror combined with relabeling every color c as c + 2. Because
network inputs are relative to the player to move, the relabeling cancels out: only
the board planes and the actions are mirrored. Vertical flips and rotations reverse
the turn order, so they are not symmetries of this game.
"""

from typing import Optional

import numpy as np

from blokus.engine.game_state import (
    ACTION_SPACE_SIZE, BOARD_SIZE, GameState, encode_action, load_pieces,
)

_MIRROR_ACTIONS: Optional[np.ndarray] = None


def mirror_action_table() -> np.ndarray:
    """Array mapping each action index to its mirror image (-1 if not a placement)."""
    global _MIRROR_ACTIONS
    if _MIRROR_ACTIONS is None:
        table = np.full(ACTION_SPACE_SIZE, -1, dtype=np.int32)
        for pid, piece in enumerate(load_pieces()):
            shapes = {tuple(sorted(o.occupied)): i for i, o in enumerate(piece.orientations)}
            for oid, ori in enumerate(piece.orientations):
                mirrored = tuple(sorted((r, ori.max_col - c) for r, c in ori.occupied))
                other = shapes[mirrored]
                for row in range(BOARD_SIZE - ori.max_row):
                    for col in range(BOARD_SIZE - ori.max_col):
                        table[encode_action(pid, oid, row, col)] = encode_action(
                            pid, other, row, BOARD_SIZE - 1 - ori.max_col - col)
        _MIRROR_ACTIONS = table
    return _MIRROR_ACTIONS


def mirror_state(state: GameState) -> GameState:
    """The same position mirrored left to right with colors relabeled by +2."""
    board = state.board[:, ::-1].copy()
    occupied = board != 0
    board[occupied] = (board[occupied] - 1 + 2) % 4 + 1
    remaining = tuple(state.pieces_remaining[(c + 2) % 4] for c in range(4))
    has_played = tuple(state._has_played[(c + 2) % 4] for c in range(4))
    return GameState(board, remaining, (state.current_color + 2) % 4,
                     state.consecutive_passes, state.game_mode, state.pieces, has_played)
