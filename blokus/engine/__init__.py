"""Blokus game engine — board, pieces, and game state management."""

from blokus.engine.game_state import (
    GameState,
    encode_action,
    decode_action,
    load_pieces,
    play_random_game,
    BOARD_SIZE,
    NUM_PIECES,
    MAX_ORIENTATIONS,
    NUM_COLORS,
    ACTION_SPACE_SIZE,
    COLOR_CORNERS,
    COLOR_NAMES,
    COLOR_RGB,
)

__all__ = [
    'GameState',
    'encode_action',
    'decode_action',
    'load_pieces',
    'play_random_game',
    'BOARD_SIZE',
    'NUM_PIECES',
    'MAX_ORIENTATIONS',
    'NUM_COLORS',
    'ACTION_SPACE_SIZE',
    'COLOR_CORNERS',
    'COLOR_NAMES',
    'COLOR_RGB',
]
