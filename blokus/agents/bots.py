"""Move-selection players for evaluation and human play (dual mode).

Every bot exposes ``select_action(state) -> int`` and is only asked to move when
the current color has at least one legal action.

The two space bots are bitboard re-implementations of ``space_heuristic`` and
``space_heuristic2`` from blokus/engine/heuristics.py: same flood-fill value
(territory fraction + squares placed / 100) applied to the agent's two colors,
but contested cells stay neutral instead of going to whichever color the legacy
scan visited last, and they run about 1000x faster.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from blokus.engine.game_state import (
    BOARD_SIZE, COLOR_CORNERS, GameState, NUM_COLORS, decode_action, load_pieces,
)
from blokus.mcts.mcts import MCTS
from blokus.nn.network import make_pieces_remaining_vector, make_score_vector

_FULL = (1 << (BOARD_SIZE * BOARD_SIZE)) - 1
_COL_FIRST = sum(1 << (r * BOARD_SIZE) for r in range(BOARD_SIZE))
_COL_LAST = _COL_FIRST << (BOARD_SIZE - 1)
_NOT_FIRST = _FULL ^ _COL_FIRST
_NOT_LAST = _FULL ^ _COL_LAST


def _ortho(x: int) -> int:
    """Cells orthogonally adjacent to the set x."""
    return (((x & _NOT_FIRST) >> 1) | ((x & _NOT_LAST) << 1)
            | (x >> BOARD_SIZE) | (x << BOARD_SIZE)) & _FULL


def _diag(x: int) -> int:
    """Cells diagonally adjacent to the set x."""
    left, right = (x & _NOT_FIRST) >> 1, (x & _NOT_LAST) << 1
    return ((left >> BOARD_SIZE) | (left << BOARD_SIZE)
            | (right >> BOARD_SIZE) | (right << BOARD_SIZE)) & _FULL


_PIECE_MASKS: Optional[List[List[int]]] = None


def _action_mask(action: int) -> Tuple[int, int]:
    """Return (bitmask of the cells this action covers, number of cells)."""
    global _PIECE_MASKS
    if _PIECE_MASKS is None:
        _PIECE_MASKS = [
            [sum(1 << (r * BOARD_SIZE + c) for r, c in ori.occupied)
             for ori in piece.orientations]
            for piece in load_pieces()
        ]
    pid, oid, row, col = decode_action(action)
    mask = _PIECE_MASKS[pid][oid] << (row * BOARD_SIZE + col)
    return mask, bin(_PIECE_MASKS[pid][oid]).count('1')


def _color_masks(board: np.ndarray) -> List[int]:
    """Board as one bitmask per color."""
    masks = [0] * NUM_COLORS
    for r, c in zip(*np.nonzero(board)):
        masks[int(board[r, c]) - 1] |= 1 << (int(r) * BOARD_SIZE + int(c))
    return masks


def _flood_territory(mine: int, theirs: int, step: Callable[[int], int]) -> int:
    """Cells mine reaches strictly before theirs, flooding through empty cells."""
    empty = _FULL & ~(mine | theirs)
    owned, frontier_m, frontier_t = mine, mine, theirs
    while frontier_m or frontier_t:
        grow_m = step(frontier_m) & empty
        grow_t = step(frontier_t) & empty
        empty &= ~(grow_m | grow_t)
        frontier_m, frontier_t = grow_m & ~grow_t, grow_t & ~grow_m
        owned |= frontier_m
    return owned


OPENING_PLIES = 4


def random_opening(seed: int) -> List[int]:
    """Legal opening actions: a random pentomino in each color's corner."""
    rng = np.random.default_rng(seed)
    pieces = load_pieces()
    state = GameState.new_game('dual')
    actions = []
    for _ in range(OPENING_PLIES):
        legal = state.get_legal_actions()
        big = [a for a in legal if pieces[decode_action(a)[0]].num_cells == 5]
        action = big[rng.integers(len(big))]
        actions.append(action)
        state = state.apply_action(action)
    return actions


class RandomBot:
    """Uniform random legal move."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: GameState) -> int:
        legal = state.get_legal_actions()
        return legal[self.rng.integers(len(legal))]


class _ScoringBot:
    """Greedy one-ply bot: plays the legal move with the highest score."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _score(self, masks: List[int], color: int, size: int) -> float:
        raise NotImplementedError

    def select_action(self, state: GameState) -> int:
        legal = state.get_legal_actions()
        color = state.current_color
        base = _color_masks(state.board)
        best: List[int] = []
        best_score = -float('inf')
        for action in legal:
            mask, size = _action_mask(action)
            masks = list(base)
            masks[color] |= mask
            score = self._score(masks, color, size)
            if score > best_score + 1e-9:
                best_score, best = score, [action]
            elif score > best_score - 1e-9:
                best.append(action)
        return best[self.rng.integers(len(best))]


class GreedySizeBot(_ScoringBot):
    """Plays the largest available piece, breaking ties at random."""

    def _score(self, masks: List[int], color: int, size: int) -> float:
        return float(size)


class SpaceBot(_ScoringBot):
    """Flood-fill territory heuristic (see module docstring).

    diagonal=False is space_heuristic, diagonal=True is space_heuristic2.
    """

    def __init__(self, diagonal: bool = False, seed: Optional[int] = None):
        super().__init__(seed)
        self.step = _diag if diagonal else _ortho

    def _score(self, masks: List[int], color: int, size: int) -> float:
        mine = masks[color] | masks[(color + 2) % NUM_COLORS]
        theirs = masks[(color + 1) % NUM_COLORS] | masks[(color + 3) % NUM_COLORS]
        owned = _flood_territory(mine, theirs, self.step)
        placed = bin(mine).count('1')
        return bin(owned).count('1') / (BOARD_SIZE * BOARD_SIZE) + placed / 100.0


def _open_corners(own: int, occupied: int) -> int:
    """Number of cells where this color could start a new piece."""
    return bin(_diag(own) & ~_ortho(own) & ~occupied & _FULL).count('1')


class MobilityBot(_ScoringBot):
    """Favors big pieces, keeping its own corners open, and blocking the opponent's.

    Not from the legacy code. Open corners are how experienced players read a
    Blokus position, so this is a closer stand-in for a person than flood-fill.
    """

    def __init__(self, size_weight: float = 2.0, seed: Optional[int] = None):
        super().__init__(seed)
        self.size_weight = size_weight

    def _score(self, masks: List[int], color: int, size: int) -> float:
        occupied = masks[0] | masks[1] | masks[2] | masks[3]
        mine = sum(_open_corners(masks[(color + i) % NUM_COLORS], occupied) for i in (0, 2))
        theirs = sum(_open_corners(masks[(color + i) % NUM_COLORS], occupied) for i in (1, 3))
        return self.size_weight * size + mine - theirs


class LookaheadBot:
    """Two-ply mobility: takes the top moves by mobility and checks the best reply.

    Each candidate is scored as its mobility value minus the replying color's best
    mobility value. Much stronger than one ply, and about 20 ms per move.
    """

    def __init__(self, candidates: int = 15, size_weight: float = 2.0,
                 seed: Optional[int] = None):
        self.candidates = candidates
        self.scorer = MobilityBot(size_weight=size_weight, seed=seed)
        self.rng = np.random.default_rng(seed)

    def _ranked(self, state: GameState) -> List[Tuple[float, int]]:
        """Legal moves in this state, best mobility value first."""
        color = state.current_color
        base = _color_masks(state.board)
        ranked = []
        for action in state.get_legal_actions():
            mask, size = _action_mask(action)
            masks = list(base)
            masks[color] |= mask
            ranked.append((self.scorer._score(masks, color, size), action))
        ranked.sort(key=lambda item: -item[0])
        return ranked

    def select_action(self, state: GameState) -> int:
        best: List[int] = []
        best_value = -float('inf')
        for score, action in self._ranked(state)[:self.candidates]:
            reply_state = state.apply_action(action)
            value = score
            if reply_state.get_legal_actions():
                value -= self._ranked(reply_state)[0][0]
            if value > best_value + 1e-9:
                best_value, best = value, [action]
            elif value > best_value - 1e-9:
                best.append(action)
        return best[self.rng.integers(len(best))]


class RawPolicyBot:
    """Network policy argmax, no search."""

    def __init__(self, network: torch.nn.Module, device: torch.device):
        self.network = network
        self.device = device

    @torch.no_grad()
    def select_action(self, state: GameState) -> int:
        legal = state.get_legal_actions()
        board = torch.from_numpy(state.get_nn_state()).unsqueeze(0).to(self.device)
        pieces = torch.from_numpy(make_pieces_remaining_vector(state)).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(state.get_legal_actions_mask()).unsqueeze(0).to(self.device)
        score = None
        if getattr(self.network, 'score_input', False):
            score = torch.from_numpy(make_score_vector(state)).unsqueeze(0).to(self.device)
        log_policy, _ = self.network(board, pieces, mask, score_vector=score)
        probs = log_policy.squeeze(0).cpu().numpy()
        return max(legal, key=lambda a: probs[a])


class MctsBot:
    """Network plus MCTS, deterministic: no root noise, plays the most visited move."""

    def __init__(self, network: torch.nn.Module, device: torch.device,
                 sims: int = 200, c_puct: float = 1.5, batch_size: int = 8,
                 top_k: int = 50, value_scale: float = 1.0,
                 terminal_margin_scale: float = 0.0,
                 time_limit: Optional[float] = None):
        self.mcts = MCTS(
            network=network, c_puct=c_puct, num_simulations=sims,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0,
            device=device, batch_size=batch_size, top_k_actions=top_k,
            value_scale=value_scale,
            terminal_margin_scale=terminal_margin_scale,
            time_limit=time_limit,
        )

    def select_action(self, state: GameState) -> int:
        action, _, _ = self.mcts.select_action(state)
        return action

    @property
    def last_stats(self) -> Optional[dict]:
        return self.mcts.last_stats
