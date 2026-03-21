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

# Try to import Cython-accelerated legal move generation
try:
    from blokus.engine._legal_moves import fast_legal_actions as _cython_legal_actions
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False

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
    global _PIECES_CACHE, _FAST_LEGAL_CACHE
    _PIECES_CACHE = None
    _FAST_LEGAL_CACHE = None


# =============================================================================
# Fast legal move generation (vectorized numpy)
# =============================================================================

# Maximum piece size in standard Blokus (pentominoes)
_MAX_PIECE_CELLS = 5

# Pre-computed placement data cache
_FAST_LEGAL_CACHE = None


class _FastLegalData:
    """Pre-computed placement data for anchor-driven legal move generation.

    Instead of checking all ~30K placements, we use a reverse index from
    board cells to placements. At query time we only check placements that
    overlap an anchor cell, reducing the candidate set to ~2-5K.

    Data structures:
    - flat_indices: (TOTAL, 5) int32 — flat cell indices per placement
    - actions: (TOTAL,) int32 — action encoding per placement
    - pid_masks: list of 21 bool arrays — which placements belong to each piece
    - csr_offsets/csr_data: CSR reverse index from flat cell → placement indices
    """

    def __init__(self, pieces: List[PieceInfo]):
        pad = BOARD_SIZE  # sentinel coordinate
        stride = BOARD_SIZE + 1

        all_flat: list = []
        all_acts: list = []
        all_pids: list = []

        for pid in range(len(pieces)):
            pinfo = pieces[pid]
            for oid in range(pinfo.num_orientations):
                ori = pinfo.orientations[oid]
                mr, mc = ori.max_row, ori.max_col
                occ = ori.occupied

                for row in range(BOARD_SIZE - mr):
                    for col in range(BOARD_SIZE - mc):
                        flat_cells = []
                        for cr, cc in occ:
                            flat_cells.append((cr + row) * stride + (cc + col))
                        sentinel = pad * stride + pad
                        while len(flat_cells) < _MAX_PIECE_CELLS:
                            flat_cells.append(sentinel)
                        all_flat.append(flat_cells)
                        all_acts.append(encode_action(pid, oid, row, col))
                        all_pids.append(pid)

        self.flat_indices = np.array(all_flat, dtype=np.int32)  # (TOTAL, 5)
        self.actions = np.array(all_acts, dtype=np.int32)       # (TOTAL,)
        piece_ids = np.array(all_pids, dtype=np.int8)           # (TOTAL,)
        self.total = len(all_acts)

        # Pre-compute boolean masks for each piece_id
        self.pid_masks: List[np.ndarray] = []
        for pid in range(NUM_PIECES):
            self.pid_masks.append(piece_ids == pid)

        # Build CSR reverse index: flat_cell → placement indices
        total_cells = stride * stride
        sentinel = pad * stride + pad
        cell_lists: List[list] = [[] for _ in range(total_cells)]
        fi = self.flat_indices
        for i in range(self.total):
            for j in range(_MAX_PIECE_CELLS):
                c = int(fi[i, j])
                if c != sentinel:
                    cell_lists[c].append(i)

        csr_all: list = []
        self.csr_offsets = np.zeros(total_cells + 1, dtype=np.int32)
        for c in range(total_cells):
            csr_all.extend(cell_lists[c])
            self.csr_offsets[c + 1] = self.csr_offsets[c] + len(cell_lists[c])
        self.csr_data = np.array(csr_all, dtype=np.int32) if csr_all else np.empty(0, dtype=np.int32)


def _get_fast_legal_data(pieces: List[PieceInfo]) -> _FastLegalData:
    """Get or create the pre-computed placement data (cached globally)."""
    global _FAST_LEGAL_CACHE
    if _FAST_LEGAL_CACHE is None:
        _FAST_LEGAL_CACHE = _FastLegalData(pieces)
    return _FAST_LEGAL_CACHE


def _fast_legal_actions(board: np.ndarray, pieces_remaining: tuple,
                        current_color: int, has_played: tuple,
                        pieces: List[PieceInfo]) -> List[int]:
    """Anchor-driven legal action generation using pre-computed placements.

    Instead of checking all ~30K placements, finds anchor cells (diagonal
    adjacencies to same-color pieces) and uses a CSR reverse index to gather
    only the ~2-5K placements that overlap an anchor. Then validates just
    those candidates against the reject mask and remaining pieces.
    """
    ci = current_color
    cv = ci + 1
    remaining = pieces_remaining[ci]
    if not remaining:
        return []

    bs = BOARD_SIZE
    stride = bs + 1
    data = _get_fast_legal_data(pieces)

    # ---- Compute reject mask as padded (bs+1, bs+1) uint8 ----
    color_mask = (board == cv)
    occ_mask = (board != 0)

    # Forbidden: cells orthogonally adjacent to same-color pieces
    forbidden = np.zeros((bs, bs), dtype=bool)
    forbidden[1:, :] |= color_mask[:-1, :]
    forbidden[:-1, :] |= color_mask[1:, :]
    forbidden[:, 1:] |= color_mask[:, :-1]
    forbidden[:, :-1] |= color_mask[:, 1:]

    reject_pad = np.zeros((bs + 1, bs + 1), dtype=np.uint8)
    reject_pad[:bs, :bs] = (occ_mask | forbidden)

    # ---- Find anchor cells as flat indices ----
    if not has_played[ci]:
        cr, cc = COLOR_CORNERS[ci]
        anchor_flat_cells = [cr * stride + cc]
    else:
        anchor = np.zeros((bs, bs), dtype=bool)
        anchor[1:, 1:] |= color_mask[:-1, :-1]
        anchor[1:, :-1] |= color_mask[:-1, 1:]
        anchor[:-1, 1:] |= color_mask[1:, :-1]
        anchor[:-1, :-1] |= color_mask[1:, 1:]
        anchor &= ~occ_mask
        arows, acols = np.where(anchor)
        if len(arows) == 0:
            return []
        anchor_flat_cells = (arows * stride + acols).tolist()

    # ---- Gather candidate placements from anchor cells via CSR index ----
    offsets = data.csr_offsets
    csr = data.csr_data
    cmask = np.zeros(data.total, dtype=bool)
    for ac in anchor_flat_cells:
        s, e = offsets[ac], offsets[ac + 1]
        if s < e:
            cmask[csr[s:e]] = True

    # Filter by remaining pieces (using pre-computed per-piece boolean masks)
    remaining_mask = np.zeros(data.total, dtype=bool)
    for pid in remaining:
        remaining_mask |= data.pid_masks[pid]
    cmask &= remaining_mask
    cand = np.where(cmask)[0]
    if len(cand) == 0:
        return []

    # ---- Validate only candidates: no rejected cells ----
    reject_flat = reject_pad.ravel()
    rv = reject_flat[data.flat_indices[cand]]  # (C, 5)
    valid = rv.max(axis=1) == 0
    return data.actions[cand[valid]].tolist()


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
        """Reward per *agent* in {-1, 0, +1}.

        Standard mode (4 agents): +1 for highest score, -1 for lowest,
            0 for ties. When all scores are equal, all get 0.
        Dual mode (2 agents): +1 for winner, -1 for loser, 0 for tie.
            Winner is the agent whose combined score is higher.
        """
        scores = self.get_scores()

        if self.game_mode == 'standard':
            vals = list(scores.values())
            max_s = max(vals)
            min_s = min(vals)
            if max_s == min_s:
                return {i: 0.0 for i in range(4)}
            rewards = {}
            for i in range(4):
                if scores[i] == max_s:
                    rewards[i] = 1.0
                elif scores[i] == min_s:
                    rewards[i] = -1.0
                else:
                    rewards[i] = 0.0
            return rewards
        else:
            # Agent 0: colors 0+2, Agent 1: colors 1+3
            s0 = scores[0] + scores[2]
            s1 = scores[1] + scores[3]
            if s0 > s1:
                return {0: 1.0, 1: -1.0}
            elif s1 > s0:
                return {0: -1.0, 1: 1.0}
            else:
                return {0: 0.0, 1: 0.0}

    # ---- Legal action generation ----

    def get_legal_actions(self) -> List[int]:
        """All legal action indices for the current color.

        Returns an empty list when the current color has no moves (must pass).
        Uses Cython extension if available, else numpy-based fallback.
        """
        if _USE_CYTHON:
            data = _get_fast_legal_data(self.pieces)
            return _cython_legal_actions(
                self.board, self.pieces_remaining, self.current_color,
                self._has_played, self.pieces, data, COLOR_CORNERS,
            )
        return _fast_legal_actions(
            self.board, self.pieces_remaining, self.current_color,
            self._has_played, self.pieces,
        )

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
