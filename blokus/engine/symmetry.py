"""The dual-mode board symmetries, for multiplying training data.

Colors sit clockwise in turn order, so turning the board a quarter turn clockwise and
relabeling every color c as c + 1 gives an equivalent position. Network inputs are
relative to the player to move, so the relabeling cancels out and only the board
planes and actions turn. That gives 4 symmetries. Reflections reverse the direction
of play, so they are not symmetries of this game.
"""

from typing import List, Optional

import numpy as np

from blokus.engine.game_state import (
    ACTION_SPACE_SIZE, BOARD_SIZE, GameState, encode_action, load_pieces,
)

_TABLES: Optional[List[np.ndarray]] = None


def rotation_action_tables() -> List[np.ndarray]:
    """tables[k] maps each action to the same move after k clockwise quarter turns.

    Entries that are not real placements are -1.
    """
    global _TABLES
    if _TABLES is None:
        one = np.full(ACTION_SPACE_SIZE, -1, dtype=np.int32)
        for pid, piece in enumerate(load_pieces()):
            shapes = {tuple(sorted(o.occupied)): i for i, o in enumerate(piece.orientations)}
            for oid, ori in enumerate(piece.orientations):
                # Cell (r, c) of a board turned clockwise moves to (c, 19 - r)
                turned = tuple(sorted((c, ori.max_row - r) for r, c in ori.occupied))
                other = shapes[turned]
                for row in range(BOARD_SIZE - ori.max_row):
                    for col in range(BOARD_SIZE - ori.max_col):
                        one[encode_action(pid, oid, row, col)] = encode_action(
                            pid, other, col, BOARD_SIZE - 1 - ori.max_row - row)
        tables = [np.where(one >= 0, np.arange(ACTION_SPACE_SIZE), -1).astype(np.int32), one]
        for _ in range(2):
            previous = tables[-1]
            tables.append(np.where(previous >= 0, one[previous.clip(0)], -1).astype(np.int32))
        _TABLES = tables
    return _TABLES


def rotate_state(state: GameState, k: int = 1) -> GameState:
    """The same position after k clockwise quarter turns, colors relabeled by +k."""
    board = np.rot90(state.board, k=-k).copy()
    occupied = board != 0
    board[occupied] = (board[occupied] - 1 + k) % 4 + 1
    order = [(c - k) % 4 for c in range(4)]
    return GameState(
        board, tuple(state.pieces_remaining[c] for c in order),
        (state.current_color + k) % 4, state.consecutive_passes, state.game_mode,
        state.pieces, tuple(state._has_played[c] for c in order),
        tuple(state._last_piece[c] for c in order))
