"""Exact endgame search agrees with plain minimax."""

import pytest

from blokus.agents.bots import RandomBot
from blokus.engine.endgame import EndgameSolver, margin
from blokus.engine.game_state import GameState


def plain_minimax(state: GameState, counter: list, limit: int) -> int:
    """Reference: full minimax, no pruning, no table."""
    counter[0] += 1
    if counter[0] > limit:
        raise RuntimeError('too big')
    if state.is_terminal():
        return margin(state)
    actions = state.get_legal_actions()
    if not actions:
        return plain_minimax(state.pass_turn(), counter, limit)
    values = [plain_minimax(state.apply_action(a), counter, limit) for a in actions]
    return max(values) if state.get_current_agent() == 0 else min(values)


def endgame_positions(seed: int, limit: int = 5000):
    """Positions from one random game, from the end back while plain minimax stays small."""
    bot = RandomBot(seed)
    state = GameState.new_game('dual')
    history = []
    while not state.is_terminal():
        history.append(state)
        actions = state.get_legal_actions()
        state = state.apply_action(bot.select_action(state)) if actions else state.pass_turn()
    found = []
    for position in reversed(history):
        try:
            value = plain_minimax(position, [0], limit)
        except RuntimeError:
            break
        found.append((position, value))
    return found


@pytest.fixture(scope='module')
def positions():
    found = []
    for seed in range(15):
        found.extend(endgame_positions(seed))
    assert sum(len(s.get_legal_actions()) >= 2 for s, _ in found) >= 15
    return found


def test_exact_margin_matches_minimax(positions):
    for state, expected in positions:
        value, _ = EndgameSolver(node_budget=10 ** 6).solve(state)
        assert value == expected


def test_win_test_matches_sign(positions):
    for state, expected in positions:
        mover_wins = expected > 0 if state.get_current_agent() == 0 else expected < 0
        assert EndgameSolver(node_budget=10 ** 6).has_win(state) == mover_wins


def test_shared_table_gives_same_answers(positions):
    solver = EndgameSolver(node_budget=10 ** 6)
    for state, expected in positions:          # Mixed windows reuse stored bounds
        solver.has_win(state)
        assert solver.solve(state)[0] == expected


def test_best_action_achieves_value(positions):
    for state, expected in positions:
        if not state.get_legal_actions() or state.is_terminal():
            continue
        value, action = EndgameSolver(node_budget=10 ** 6).solve(state)
        assert EndgameSolver(node_budget=10 ** 6).solve(state.apply_action(action))[0] == value


def test_budget_exceeded_returns_none(positions):
    deepest = positions[-1][0]
    solver = EndgameSolver(node_budget=1)
    if solver.solve(deepest) is not None:     # Only possible if it solves in one node
        pytest.skip('position too small to exceed the budget')
    assert solver.has_win(deepest) is None
