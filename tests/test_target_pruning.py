"""Pruning noise-forced visits from the self-play policy target."""

import math

import numpy as np
import torch

from blokus.engine.game_state import GameState
from blokus.mcts.mcts import MCTS, MCTSNode
from blokus.nn.network import BlokusNetwork


def make_root(children):
    """Root with (action, visits, q from the root's view, clean prior) children."""
    root = MCTSNode(GameState.new_game('dual'))
    root.visit_count = sum(v for _, v, _, _ in children) + 1
    for action, visits, q, prior in children:
        child = MCTSNode(None, parent=root, action=action, prior=prior)
        child.clean_prior = prior
        child.visit_count = visits
        child.total_value = -q * visits     # Stored from the child's view
        root.children[action] = child
    return root


def test_pruning_matches_hand_computation():
    mcts = MCTS(network=None, c_puct=1.5)
    # best: 60 visits; a: a strong runner-up; b: noise-driven, low prior and Q
    root = make_root([(1, 60, 0.30, 0.40), (2, 30, 0.28, 0.30), (3, 20, -0.20, 0.01), (4, 1, 0.0, 0.1)])
    sqrt_n = math.sqrt(root.visit_count)
    best = 0.30 + 1.5 * 0.40 * sqrt_n / 61
    keep_a = min(30, int(1.5 * 0.30 * sqrt_n / (best - 0.28)))
    keep_b = min(20, int(1.5 * 0.01 * sqrt_n / (best + 0.20)))
    keep_b = 0 if keep_b <= 1 else keep_b
    expect = np.array([60, keep_a, keep_b, 0], dtype=np.float64)
    expect /= expect.sum()
    got = mcts._pruned_policy(root)
    assert np.allclose([got[1], got[2], got[3], got[4]], expect, atol=1e-6)
    assert got[3] == 0 and got[4] == 0         # Noise-driven and single-visit children go


def test_pruned_target_on_a_real_search():
    torch.manual_seed(0)
    np.random.seed(0)
    net = BlokusNetwork(num_blocks=1, channels=16).eval()
    state = GameState.new_game('dual')
    for _ in range(8):
        state = state.apply_action(state.get_legal_actions()[0])
    mcts = MCTS(net, num_simulations=120, dirichlet_alpha=0.1, dirichlet_epsilon=0.25,
                prune_targets=True, top_k_actions=20)
    action, target, _ = mcts.select_action(state)
    visits = {a: v for a, (v, _, _) in mcts.last_stats['children'].items() if a >= 0}
    raw_policy = np.zeros_like(target)
    for a, v in visits.items():
        raw_policy[a] = v
    raw_policy /= raw_policy.sum()
    assert abs(target.sum() - 1) < 1e-5
    assert set(np.nonzero(target)[0]) <= set(np.nonzero(raw_policy)[0])
    best = max(visits, key=visits.get)
    assert target[best] >= raw_policy[best]     # The most visited move keeps all its visits
    assert raw_policy[action] > 0                # Moves are still chosen from raw visits


def test_off_by_default():
    torch.manual_seed(0)
    np.random.seed(0)
    net = BlokusNetwork(num_blocks=1, channels=16).eval()
    mcts = MCTS(net, num_simulations=40, top_k_actions=10)
    _, target, _ = mcts.select_action(GameState.new_game('dual'))
    visits = {a: v for a, (v, _, _) in mcts.last_stats['children'].items() if a >= 0}
    for a, v in visits.items():
        assert abs(target[a] - v / sum(visits.values())) < 1e-6


def test_self_play_options():
    """Random openings skip the opening plies; pruned targets stay distributions."""
    from blokus.agents.alpha_zero import self_play_game
    from blokus.engine.game_state import NUM_COLORS
    torch.manual_seed(0)
    np.random.seed(0)
    net = BlokusNetwork(num_blocks=1, channels=8).eval()
    kwargs = dict(network=net, num_simulations=2, top_k_actions=5, margin_scale=20.0,
                  dirichlet_alpha=0.1)
    plain = self_play_game(**kwargs)
    assert plain[0].board_state[:NUM_COLORS].sum() == 0          # Starts from the empty board
    opened = self_play_game(random_opening_fraction=1.0, **kwargs)
    assert opened[0].board_state[:NUM_COLORS].sum() == 4 * 5      # Four pentominoes already down
    pruned = self_play_game(prune_targets=True, **kwargs)
    for ex in pruned:
        assert abs(ex.policy_target.sum() - 1) < 1e-5
        assert np.all(ex.policy_target[ex.legal_mask == 0] == 0)
