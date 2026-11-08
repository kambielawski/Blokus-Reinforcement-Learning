"""Exact endgame search for dual-color games.

Alpha-beta over the rule-book final score. The value of a position is the margin
agent 0 (colors 0 and 2) finishes ahead of agent 1 (colors 1 and 3) when both
play perfectly: agent 0 maximizes it, agent 1 minimizes it. Its sign is the
win/loss result, so a positive value proves agent 0 can force a win against any
opponent.

A transposition table stores bounds, not just exact values, so one table can be
shared across searches of related positions (for example, every position of one
game, solved from the end backward) and across null-window win tests.

The search gives up when it exceeds its node budget, so callers can fall back to
MCTS: `solve` and `has_win` return None in that case.
"""

from typing import Callable, Dict, List, Optional, Tuple

from blokus.engine.game_state import GameState, decode_action

INF = 10 ** 6


class BudgetExceeded(Exception):
    """The search needed more nodes than it was allowed."""


def margin(state: GameState) -> int:
    """Final margin of agent 0 over agent 1 (dual mode)."""
    s = state.get_scores()
    return s[0] + s[2] - s[1] - s[3]


def position_key(state: GameState) -> tuple:
    """Everything the rest of the game depends on."""
    return (state.board.tobytes(), state.pieces_remaining, state.current_color,
            state.consecutive_passes, state._last_piece)


def big_pieces_first(state: GameState, actions: List[int]) -> List[int]:
    """Move ordering: unplaced squares cost points, so big pieces tend to be best."""
    pieces = state.pieces
    return sorted(actions, key=lambda a: -pieces[decode_action(a)[0]].num_cells)


class EndgameSolver:
    """Exact minimax with alpha-beta and a bounds transposition table.

    Args:
        node_budget: Nodes one call may expand before giving up.
        order: Move ordering, (state, actions) -> actions. Better ordering
            means more cutoffs, not a different answer.
    """

    def __init__(self, node_budget: int = 200_000,
                 order: Callable[[GameState, List[int]], List[int]] = big_pieces_first):
        self.node_budget = node_budget
        self.order = order
        # key -> (lower bound, upper bound, best action or None)
        self.table: Dict[tuple, Tuple[int, int, Optional[int]]] = {}
        self.nodes = 0

    def solve(self, state: GameState) -> Optional[Tuple[int, Optional[int]]]:
        """Exact margin and a best action (None for a pass or a finished game)."""
        return self._run(state, -INF, INF)

    def has_win(self, state: GameState) -> Optional[bool]:
        """Can the agent to move force a final margin above zero?

        A null-window test, much cheaper than the exact margin.
        """
        if state.get_current_agent() == 0:
            result = self._run(state, 0, 1)        # Is the value at least 1?
            return None if result is None else result[0] >= 1
        result = self._run(state, -1, 0)           # Is the value at most -1?
        return None if result is None else result[0] <= -1

    def _run(self, state: GameState, alpha: int, beta: int):
        self.nodes = 0
        try:
            value = self._search(state, alpha, beta)
        except BudgetExceeded:
            return None
        entry = self.table.get(position_key(state))
        return value, (entry[2] if entry else None)

    def _search(self, state: GameState, alpha: int, beta: int) -> int:
        if state.is_terminal():
            return margin(state)
        key = position_key(state)
        lo, hi, best = self.table.get(key, (-INF, INF, None))
        if lo >= beta:
            return lo
        if hi <= alpha:
            return hi
        if lo == hi:
            return lo
        alpha, beta = max(alpha, lo), min(beta, hi)
        window = (alpha, beta)

        self.nodes += 1
        if self.nodes > self.node_budget:
            raise BudgetExceeded

        actions = state.get_legal_actions()
        if not actions:
            value = self._search(state.pass_turn(), alpha, beta)
            best = None
        else:
            actions = self.order(state, actions)
            if best in actions:                 # Previous best first
                actions.remove(best)
                actions.insert(0, best)
            maximizing = state.get_current_agent() == 0
            value = -INF if maximizing else INF
            for action in actions:
                child = self._search(state.apply_action(action), alpha, beta)
                if maximizing:
                    if child > value:
                        value, best = child, action
                    alpha = max(alpha, value)
                else:
                    if child < value:
                        value, best = child, action
                    beta = min(beta, value)
                if alpha >= beta:
                    break

        # Fail-soft: outside the entry window the value is only a bound
        if value <= window[0]:
            hi = min(hi, value)
        elif value >= window[1]:
            lo = max(lo, value)
        else:
            lo = hi = value
        self.table[key] = (lo, hi, best)
        return value
