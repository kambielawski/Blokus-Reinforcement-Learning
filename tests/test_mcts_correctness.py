"""Regression tests for MCTS correctness — sign conventions and search quality.

These tests verify that:
1. _select_child correctly negates child values (parent picks what's bad for child)
2. _terminal_value returns reward from the terminal node's own perspective
3. MCTS search improves over raw policy when playing against random
"""

import pytest
import numpy as np
import torch

from blokus.engine.game_state import GameState, ACTION_SPACE_SIZE
from blokus.nn.network import BlokusNetwork, make_pieces_remaining_vector
from blokus.mcts.mcts import MCTS, MCTSNode


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def network(device):
    """Small network for testing."""
    net = BlokusNetwork(num_blocks=2, channels=32, value_fc_size=64).to(device)
    return net


# ---------------------------------------------------------------------------
# Test 1: Backup-select sign consistency
# ---------------------------------------------------------------------------

class TestSelectChildSign:
    """Verify that _select_child picks the child whose value is BAD for the
    child (i.e. good for the parent), due to the negation of child.mean_value."""

    def test_select_child_negates_value(self, network, device):
        """Parent should prefer the child with LOWER mean_value (from child's
        perspective), because -child.mean_value is higher."""
        mcts = MCTS(network=network, num_simulations=10, device=device)

        # Create a root with a dummy state
        state = GameState.new_game('dual')
        root = MCTSNode(state)
        root.is_expanded = True

        legal = state.get_legal_actions()
        assert len(legal) >= 2

        # Manually create two children with different values
        a0, a1 = legal[0], legal[1]
        child_good_for_parent = MCTSNode(
            state=None, parent=root, action=a0, prior=0.5, parent_state=state
        )
        child_good_for_parent.visit_count = 10
        child_good_for_parent.total_value = -5.0  # mean = -0.5 (bad for child = good for parent)

        child_bad_for_parent = MCTSNode(
            state=None, parent=root, action=a1, prior=0.5, parent_state=state
        )
        child_bad_for_parent.visit_count = 10
        child_bad_for_parent.total_value = 5.0  # mean = +0.5 (good for child = bad for parent)

        root.children = {a0: child_good_for_parent, a1: child_bad_for_parent}
        root.visit_count = 20

        selected = mcts._select_child(root)
        assert selected is child_good_for_parent, (
            "Parent should select child with negative mean_value (bad for child, good for parent)"
        )

    def test_select_child_with_equal_priors(self, network, device):
        """With equal priors and visit counts, selection should be driven by Q-values."""
        mcts = MCTS(network=network, num_simulations=10, c_puct=1.0, device=device)

        state = GameState.new_game('dual')
        root = MCTSNode(state)
        root.is_expanded = True

        legal = state.get_legal_actions()
        a0, a1 = legal[0], legal[1]

        # Both have same prior and visits, only value differs
        child_a = MCTSNode(state=None, parent=root, action=a0, prior=0.5, parent_state=state)
        child_a.visit_count = 5
        child_a.total_value = -3.0  # mean = -0.6 => -(-0.6) = +0.6 from parent's view

        child_b = MCTSNode(state=None, parent=root, action=a1, prior=0.5, parent_state=state)
        child_b.visit_count = 5
        child_b.total_value = 3.0   # mean = +0.6 => -(+0.6) = -0.6 from parent's view

        root.children = {a0: child_a, a1: child_b}
        root.visit_count = 10

        selected = mcts._select_child(root)
        assert selected is child_a


# ---------------------------------------------------------------------------
# Test 2: Terminal value perspective
# ---------------------------------------------------------------------------

class TestTerminalValue:
    """Verify that _terminal_value returns the reward for the terminal node's
    own current_agent, not the parent's."""

    def test_terminal_value_uses_leaf_agent(self, network, device):
        """Play a game to completion and verify _terminal_value returns
        reward from the perspective of the terminal state's current_agent."""
        mcts = MCTS(network=network, num_simulations=10, device=device)

        # Play random moves until terminal
        state = GameState.new_game('dual')
        rng = np.random.RandomState(42)
        while not state.is_terminal():
            legal = state.get_legal_actions()
            if not legal:
                state = state.pass_turn()
                continue
            action = legal[rng.randint(len(legal))]
            state = state.apply_action(action)

        # Create terminal node
        terminal_node = MCTSNode(state)
        assert terminal_node.is_terminal

        value = mcts._terminal_value(terminal_node)
        rewards = state.get_rewards()
        agent = state.get_current_agent()
        expected = rewards.get(agent, 0.0)
        assert value == expected, (
            f"Terminal value {value} should match reward for agent {agent}: {expected}"
        )


# ---------------------------------------------------------------------------
# Test 3: MCTS improves over random
# ---------------------------------------------------------------------------

def _play_eval_game(network, device, use_mcts, game_mode='dual'):
    """Play one game: agent 0 (trained) vs agent 1 (random).
    Returns 1 if agent 0 wins, 0 if loses, 0.5 if draw."""
    if use_mcts:
        mcts = MCTS(
            network=network, c_puct=1.5, num_simulations=25,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
            temperature=0.1, device=device,
        )

    state = GameState.new_game(game_mode)
    rng = np.random.RandomState()

    while not state.is_terminal():
        agent_idx = state.get_current_agent()
        legal = state.get_legal_actions()
        if not legal:
            state = state.pass_turn()
            continue

        if agent_idx == 0:
            if use_mcts:
                action, _, _ = mcts.select_action(state)
            else:
                # Raw policy: pick highest-prior legal action
                board = state.get_nn_state()
                pieces = make_pieces_remaining_vector(state)
                mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
                for a in legal:
                    mask[a] = 1.0

                with torch.no_grad():
                    board_t = torch.from_numpy(board[np.newaxis]).to(device)
                    pieces_t = torch.from_numpy(pieces[np.newaxis]).to(device)
                    mask_t = torch.from_numpy(mask[np.newaxis]).to(device)
                    network.eval()
                    log_policy, _ = network(board_t, pieces_t, mask_t)
                    probs = torch.exp(log_policy).cpu().numpy()[0]

                best_a, best_p = legal[0], -1.0
                for a in legal:
                    if probs[a] > best_p:
                        best_p = probs[a]
                        best_a = a
                action = best_a
        else:
            action = legal[rng.randint(len(legal))]

        state = state.apply_action(action)

    rewards = state.get_rewards()
    if rewards[0] > rewards[1]:
        return 1.0
    elif rewards[1] > rewards[0]:
        return 0.0
    return 0.5


class TestMCTSImproves:
    """MCTS should perform at least as well as raw policy against random."""

    @pytest.mark.slow
    def test_mcts_at_least_as_good_as_raw(self, network, device):
        """Run 20 games each (raw vs random, MCTS vs random).
        MCTS win rate should be >= raw policy win rate.
        With a random untrained network, MCTS search should still help
        by exploring more of the game tree."""
        num_games = 20

        raw_score = sum(_play_eval_game(network, device, use_mcts=False)
                        for _ in range(num_games))
        mcts_score = sum(_play_eval_game(network, device, use_mcts=True)
                         for _ in range(num_games))

        raw_rate = raw_score / num_games
        mcts_rate = mcts_score / num_games

        # MCTS should not be significantly worse than raw policy.
        # Allow small margin for randomness.
        assert mcts_rate >= raw_rate - 0.15, (
            f"MCTS win rate ({mcts_rate:.1%}) should not be much worse than "
            f"raw policy ({raw_rate:.1%}). Sign bug may be present."
        )


# ---------------------------------------------------------------------------
# Test 4: Top-K action expansion
# ---------------------------------------------------------------------------

class TestTopKActions:
    """Verify that top_k_actions limits the number of children created."""

    def test_top_k_limits_children(self, network, device):
        """With top_k_actions=5, a node should have at most 5 children."""
        mcts = MCTS(
            network=network, num_simulations=10, device=device,
            top_k_actions=5,
        )
        state = GameState.new_game('dual')
        root = MCTSNode(state)

        mcts._expand_single(root, add_noise=False)

        legal_count = len(state.get_legal_actions())
        assert legal_count > 5, "Need more than 5 legal actions for this test"
        assert len(root.children) == 5, (
            f"Expected 5 children with top_k_actions=5, got {len(root.children)}"
        )

    def test_top_k_zero_expands_all(self, network, device):
        """With top_k_actions=0, all legal actions should get children."""
        mcts = MCTS(
            network=network, num_simulations=10, device=device,
            top_k_actions=0,
        )
        state = GameState.new_game('dual')
        root = MCTSNode(state)

        mcts._expand_single(root, add_noise=False)

        legal_count = len(state.get_legal_actions())
        assert len(root.children) == legal_count
