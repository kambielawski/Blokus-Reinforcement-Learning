"""
Blokus game engine — clean, immutable GameState for AlphaZero-style RL.

Supports two game modes:
- 'standard': 4 agents, each controlling one color (1v1v1v1)
- 'dual': 2 agents, each controlling two colors (1v1 dual-color)

Both modes use a 20x20 board with 4 colors and 21 pieces per color.
Turn order always cycles: color 0 -> 1 -> 2 -> 3 -> 0 -> ...
"""

import numpy as np
import pickle
import os
import random as _random
from typing import List, Tuple, Dict, Optional, Set

# Import Piece class (used for loading/unpickling piece definitions)
from blokus.engine.piece import Piece

# Repo root for resolving data paths
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _PieceUnpickler(pickle.Unpickler):
    """Custom unpickler that redirects old 'Piece.Piece' references
    to the new package location 'blokus.engine.piece.Piece'."""
    def find_class(self, module: str, name: str):
        if module == 'Piece' and name == 'Piece':
            return Piece
        return super().find_class(module, name)

# =============================================================================
# Constants
# =============================================================================

BOARD_SIZE = 20
NUM_PIECES = 21
MAX_ORIENTATIONS = 8
NUM_COLORS = 4

# Total action space: piece_id * (8 * 20 * 20) + ori_id * (20 * 20) + row * 20 + col
ACTION_SPACE_SIZE = NUM_PIECES * MAX_ORIENTATIONS * BOARD_SIZE * BOARD_SIZE  # 67,200

# Board corner assignments for each color index (0-3) -> (row, col)
COLOR_CORNERS = {
    0: (0, 0),
    1: (BOARD_SIZE - 1, BOARD_SIZE - 1),
    2: (0, BOARD_SIZE - 1),
    3: (BOARD_SIZE - 1, 0),
}

# Color names for display
COLOR_NAMES = {0: 'Yellow', 1: 'Blue', 2: 'Red', 3: 'Green'}

# RGB colors for rendering
COLOR_RGB = {
    0: (240, 240, 240),  # empty: light gray
    1: (255, 220, 0),    # yellow
    2: (0, 100, 255),    # blue
    3: (255, 50, 50),    # red
    4: (50, 200, 50),    # green
}


# =============================================================================
# Piece data structures and loading
# =============================================================================

class OrientationData:
    """Pre-computed data for one piece orientation.

    All coordinates are relative to (0, 0) origin (after shift_min).
    """
    __slots__ = ['occupied', 'max_row', 'max_col']

    def __init__(self, occupied: tuple, max_row: int, max_col: int):
        self.occupied = occupied    # tuple of (row, col) tuples
        self.max_row = max_row
        self.max_col = max_col


class PieceInfo:
    """All orientations for one piece."""
    __slots__ = ['orientations', 'num_orientations', 'num_cells']

    def __init__(self, orientations: List[OrientationData], num_cells: int):
        self.orientations = orientations
        self.num_orientations = len(orientations)
        self.num_cells = num_cells


# Module-level cache
_PIECES_CACHE: Optional[List[PieceInfo]] = None


def load_pieces(pickle_path: Optional[str] = None) -> List[PieceInfo]:
    """Load piece definitions from pickle and pre-compute orientation data.

    Returns a list of 21 PieceInfo objects, one per standard Blokus piece.
    Each PieceInfo has up to 8 unique orientations (rotations + flips).
    """
    global _PIECES_CACHE
    if _PIECES_CACHE is not None:
        return _PIECES_CACHE

    if pickle_path is None:
        pickle_path = os.path.join(_REPO_ROOT, 'data', 'blokus_pieces.pkl')

    with open(pickle_path, 'rb') as f:
        all_pieces = _PieceUnpickler(f).load()

    pieces: List[PieceInfo] = []
    for piece in all_pieces:
        if piece.size <= 5:
            raw_orientations = piece.get_orientations()
            orientations: List[OrientationData] = []
            num_cells = 0
            for (p, _is_flipped, _rot) in raw_orientations:
                occupied = tuple(sorted(p.occupied))
                num_cells = len(occupied)
                max_row = max(r for r, c in occupied)
                max_col = max(c for r, c in occupied)
                orientations.append(OrientationData(occupied, max_row, max_col))
            pieces.append(PieceInfo(orientations, num_cells))

    assert len(pieces) == NUM_PIECES, f"Expected {NUM_PIECES} pieces, got {len(pieces)}"
    _PIECES_CACHE = pieces
    return pieces


def clear_piece_cache():
    """Clear the global piece cache (useful for testing)."""
    global _PIECES_CACHE
    _PIECES_CACHE = None


# =============================================================================
# Action encoding / decoding
# =============================================================================

def encode_action(piece_id: int, orientation_id: int, row: int, col: int) -> int:
    """Encode a move as a flat integer.

    action = piece_id * (8 * 20 * 20) + orientation_id * (20 * 20) + row * 20 + col
    """
    return (piece_id * (MAX_ORIENTATIONS * BOARD_SIZE * BOARD_SIZE)
            + orientation_id * (BOARD_SIZE * BOARD_SIZE)
            + row * BOARD_SIZE
            + col)


def decode_action(action: int) -> Tuple[int, int, int, int]:
    """Decode a flat action integer to (piece_id, orientation_id, row, col)."""
    col = action % BOARD_SIZE
    action //= BOARD_SIZE
    row = action % BOARD_SIZE
    action //= BOARD_SIZE
    orientation_id = action % MAX_ORIENTATIONS
    piece_id = action // MAX_ORIENTATIONS
    return piece_id, orientation_id, row, col


# =============================================================================
# GameState
# =============================================================================

class GameState:
    """Immutable Blokus game state.

    The board uses values 0 (empty) and 1-4 for the four colors.
    Color index i corresponds to board value i+1.

    Calling apply_action() or pass_turn() returns a *new* GameState;
    the original is never mutated.
    """

    __slots__ = ['board', 'pieces_remaining', 'current_color',
                 'consecutive_passes', 'game_mode', 'pieces', '_has_played']

    def __init__(self, board: np.ndarray, pieces_remaining: tuple,
                 current_color: int, consecutive_passes: int,
                 game_mode: str, pieces: List[PieceInfo], has_played: tuple):
        self.board = board
        self.pieces_remaining = pieces_remaining  # tuple of 4 frozensets
        self.current_color = current_color        # 0-3
        self.consecutive_passes = consecutive_passes
        self.game_mode = game_mode                # 'standard' or 'dual'
        self.pieces = pieces                      # shared reference
        self._has_played = has_played             # tuple of 4 bools

    # ---- Factory ----

    @staticmethod
    def new_game(game_mode: str = 'standard',
                 pickle_path: Optional[str] = None) -> 'GameState':
        """Create a fresh game state.

        Args:
            game_mode: 'standard' (4 agents) or 'dual' (2 agents, each plays 2 colors).
            pickle_path: Path to piece definitions pickle. Auto-detected if None.
        """
        assert game_mode in ('standard', 'dual'), f"Invalid game_mode: {game_mode}"
        pieces = load_pieces(pickle_path)
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        all_ids = frozenset(range(NUM_PIECES))
        pieces_remaining = (all_ids, all_ids, all_ids, all_ids)
        has_played = (False, False, False, False)
        return GameState(board, pieces_remaining, 0, 0, game_mode, pieces, has_played)

    def copy(self) -> 'GameState':
        """Shallow copy with new board array."""
        return GameState(
            self.board.copy(), self.pieces_remaining, self.current_color,
            self.consecutive_passes, self.game_mode, self.pieces, self._has_played)

    # ---- Query methods ----

    def get_current_player(self) -> int:
        """Current color index (0-3)."""
        return self.current_color

    def get_current_agent(self) -> int:
        """Current agent index.

        Standard mode: agent == color (0-3).
        Dual mode: agent 0 controls colors 0 and 2; agent 1 controls colors 1 and 3.
        """
        if self.game_mode == 'standard':
            return self.current_color
        else:
            return self.current_color % 2

    def get_num_agents(self) -> int:
        """Number of agents in this game mode."""
        return 4 if self.game_mode == 'standard' else 2

    def is_terminal(self) -> bool:
        """True if the game is over.

        Terminal conditions:
        1. All 4 colors passed consecutively (no one can move).
        2. Any color has placed all 21 pieces.
        """
        if self.consecutive_passes >= NUM_COLORS:
            return True
        for remaining in self.pieces_remaining:
            if len(remaining) == 0:
                return True
        return False

    def get_scores(self) -> Dict[int, int]:
        """Score per color index: number of squares occupied on the board."""
        scores = {}
        for ci in range(NUM_COLORS):
            scores[ci] = int(np.sum(self.board == (ci + 1)))
        return scores

    def get_rewards(self) -> Dict[int, float]:
        """Reward per *agent* in [-1, 1].

        Standard mode (4 agents): normalized score relative to best/worst.
        Dual mode (2 agents): difference of combined scores, normalized.
        """
        scores = self.get_scores()

        if self.game_mode == 'standard':
            vals = list(scores.values())
            max_s, min_s = max(vals), min(vals)
            span = max_s - min_s
            if span == 0:
                return {i: 0.0 for i in range(4)}
            return {i: (scores[i] - min_s) / span * 2 - 1 for i in range(4)}
        else:
            # Agent 0: colors 0+2, Agent 1: colors 1+3
            s0 = scores[0] + scores[2]
            s1 = scores[1] + scores[3]
            total = s0 + s1
            if total == 0:
                return {0: 0.0, 1: 0.0}
            diff = (s0 - s1) / total
            return {0: diff, 1: -diff}

    # ---- Legal action generation ----

    def get_legal_actions(self) -> List[int]:
        """All legal action indices for the current color.

        Returns an empty list when the current color has no moves (must pass).
        """
        ci = self.current_color
        cv = ci + 1  # board value for this color
        remaining = self.pieces_remaining[ci]
        if not remaining:
            return []

        board = self.board
        bs = BOARD_SIZE

        # ---- Compute anchor points ----
        # Anchor = empty cell where a piece cell can go to satisfy diagonal adjacency.
        if not self._has_played[ci]:
            # First move: piece must cover the assigned corner cell
            anchor_set = {COLOR_CORNERS[ci]}
        else:
            color_mask = (board == cv)
            occ_mask = (board != 0)
            anchor = np.zeros((bs, bs), dtype=bool)
            anchor[1:, 1:]   |= color_mask[:-1, :-1]
            anchor[1:, :-1]  |= color_mask[:-1, 1:]
            anchor[:-1, 1:]  |= color_mask[1:, :-1]
            anchor[:-1, :-1] |= color_mask[1:, 1:]
            anchor &= ~occ_mask
            anchor_set = set(zip(*np.where(anchor)))

        if not anchor_set:
            return []

        # ---- Compute forbidden mask (ortho-adjacent to same color) ----
        color_mask = (board == cv)
        forbidden = np.zeros((bs, bs), dtype=bool)
        forbidden[1:, :]  |= color_mask[:-1, :]
        forbidden[:-1, :] |= color_mask[1:, :]
        forbidden[:, 1:]  |= color_mask[:, :-1]
        forbidden[:, :-1] |= color_mask[:, 1:]

        occ_mask = (board != 0)

        # ---- Enumerate legal placements ----
        legal: List[int] = []
        seen: Set[Tuple[int, int, int, int]] = set()

        for pid in remaining:
            pinfo = self.pieces[pid]
            for oid in range(pinfo.num_orientations):
                ori = pinfo.orientations[oid]
                occ = ori.occupied
                mr, mc = ori.max_row, ori.max_col

                for (ar, ac) in anchor_set:
                    for (pr, pc) in occ:
                        tx = ar - pr
                        ty = ac - pc

                        # Dedup
                        key = (pid, oid, tx, ty)
                        if key in seen:
                            continue
                        seen.add(key)

                        # Quick bounds check
                        if tx < 0 or ty < 0 or tx + mr >= bs or ty + mc >= bs:
                            continue

                        # Validate each cell
                        valid = True
                        for (cr, cc) in occ:
                            r = cr + tx
                            c = cc + ty
                            if occ_mask[r, c] or forbidden[r, c]:
                                valid = False
                                break

                        if valid:
                            legal.append(encode_action(pid, oid, tx, ty))

        return legal

    def get_legal_actions_mask(self) -> np.ndarray:
        """Binary mask over the full action space (ACTION_SPACE_SIZE,).

        1 where the action is legal, 0 otherwise.
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for a in self.get_legal_actions():
            mask[a] = 1.0
        return mask

    # ---- State transitions ----

    def apply_action(self, action: int) -> 'GameState':
        """Place a piece and advance to the next color.

        Resets consecutive_passes to 0.
        """
        pid, oid, tx, ty = decode_action(action)
        ci = self.current_color
        cv = ci + 1

        ori = self.pieces[pid].orientations[oid]

        # New board
        new_board = self.board.copy()
        for (cr, cc) in ori.occupied:
            new_board[cr + tx, cc + ty] = cv

        # New pieces remaining
        pr = list(self.pieces_remaining)
        pr[ci] = pr[ci] - {pid}
        new_pr = tuple(pr)

        # New has_played
        hp = list(self._has_played)
        hp[ci] = True
        new_hp = tuple(hp)

        next_color = (ci + 1) % NUM_COLORS
        return GameState(new_board, new_pr, next_color, 0,
                         self.game_mode, self.pieces, new_hp)

    def pass_turn(self) -> 'GameState':
        """Current color passes (no legal moves). Advances turn."""
        next_color = (self.current_color + 1) % NUM_COLORS
        return GameState(self.board, self.pieces_remaining, next_color,
                         self.consecutive_passes + 1, self.game_mode,
                         self.pieces, self._has_played)

    # ---- Neural network state ----

    def get_nn_state(self) -> np.ndarray:
        """State tensor for neural network input.

        Shape: (NUM_COLORS + 1, BOARD_SIZE, BOARD_SIZE) float32
          Channel 0: current color's pieces
          Channels 1-3: other colors in turn order
          Channel 4: empty squares
        """
        state = np.zeros((NUM_COLORS + 1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        state[0] = (self.board == (self.current_color + 1))
        for i in range(1, NUM_COLORS):
            other = (self.current_color + i) % NUM_COLORS
            state[i] = (self.board == (other + 1))
        state[NUM_COLORS] = (self.board == 0)
        return state

    # ---- Rendering ----

    def render_text(self) -> str:
        """ASCII board representation."""
        syms = {0: '.', 1: 'Y', 2: 'B', 3: 'R', 4: 'G'}
        lines = []
        for r in range(BOARD_SIZE):
            lines.append(' '.join(syms[int(self.board[r, c])]
                                  for c in range(BOARD_SIZE)))
        return '\n'.join(lines)

    def render_image(self, cell_size: int = 24) -> np.ndarray:
        """Render board as an RGB numpy array (H, W, 3) uint8."""
        bs = BOARD_SIZE
        img_h = bs * cell_size + 1
        img_w = bs * cell_size + 1
        img = np.full((img_h, img_w, 3), 180, dtype=np.uint8)

        for r in range(bs):
            for c in range(bs):
                rgb = COLOR_RGB[int(self.board[r, c])]
                y0 = r * cell_size + 1
                x0 = c * cell_size + 1
                img[y0:y0 + cell_size - 1, x0:x0 + cell_size - 1] = rgb

        # Grid lines
        for i in range(bs + 1):
            img[i * cell_size, :] = (80, 80, 80)
            img[:, i * cell_size] = (80, 80, 80)

        return img

    def __repr__(self) -> str:
        scores = self.get_scores()
        score_str = ', '.join(f"{COLOR_NAMES[i]}={scores[i]}" for i in range(4))
        return (f"GameState(turn={COLOR_NAMES[self.current_color]}, "
                f"mode={self.game_mode}, passes={self.consecutive_passes}, "
                f"scores=[{score_str}])")


# =============================================================================
# Utility: play a random game
# =============================================================================

def play_random_game(game_mode: str = 'standard', seed: int = 42,
                     verbose: bool = False) -> Tuple['GameState', List['GameState']]:
    """Play a full game with random agents.

    Returns (final_state, history) where history is a list of GameState
    snapshots after each action/pass.
    """
    rng = _random.Random(seed)
    state = GameState.new_game(game_mode)
    history = [state]

    move_num = 0
    while not state.is_terminal():
        legal = state.get_legal_actions()
        if not legal:
            state = state.pass_turn()
        else:
            action = rng.choice(legal)
            state = state.apply_action(action)
            move_num += 1

        history.append(state)

        if verbose and move_num % 10 == 0:
            scores = state.get_scores()
            print(f"  Move {move_num}: {scores}")

    if verbose:
        print(f"Game over after {move_num} moves. Final scores: {state.get_scores()}")

    return state, history


if __name__ == '__main__':
    import time

    print("Playing a standard 4-player random game...")
    t0 = time.time()
    final, hist = play_random_game('standard', seed=42, verbose=True)
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s, {len(hist)} states\n")
    print(final.render_text())
    print(f"\nRewards: {final.get_rewards()}")

    print("\n\nPlaying a dual-color 1v1 random game...")
    t0 = time.time()
    final2, hist2 = play_random_game('dual', seed=123, verbose=True)
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s, {len(hist2)} states\n")
    print(final2.render_text())
    print(f"\nRewards: {final2.get_rewards()}")
