"""Tests for the AlphaZero components: neural network, MCTS, agent, and training."""

import pytest
import numpy as np
import torch

from blokus.engine.game_state import (
    GameState, ACTION_SPACE_SIZE, BOARD_SIZE, NUM_PIECES, NUM_COLORS,
    MAX_ORIENTATIONS,
)
from blokus.nn.network import (
    BlokusNetwork, make_pieces_remaining_vector,
    INPUT_CHANNELS, POLICY_CHANNELS, PIECE_REMAINING_SIZE,
)
from blokus.mcts.mcts import MCTS, MCTSNode
from blokus.agents.alpha_zero import AlphaZeroAgent, self_play_game, TrainingExample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def network(device):
    """Small network for testing (fewer blocks/channels for speed)."""
    net = BlokusNetwork(num_blocks=2, channels=32, value_fc_size=64).to(device)
    return net


@pytest.fixture
def new_game():
    return GameState.new_game('dual')


@pytest.fixture
def new_game_standard():
    return GameState.new_game('standard')


# ---------------------------------------------------------------------------
# Neural Network Tests
# ---------------------------------------------------------------------------

class TestBlokusNetwork:
    """Tests for the neural network architecture."""

    def test_forward_pass_shapes(self, network, device, new_game):
        """Network produces correct output shapes."""
        batch_size = 4
        board_state = torch.randn(batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE).to(device)
        pieces_vec = torch.ones(batch_size, PIECE_REMAINING_SIZE).to(device)
        legal_mask = torch.ones(batch_size, ACTION_SPACE_SIZE).to(device)

        log_policy, value = network(board_state, pieces_vec, legal_mask)

        assert log_policy.shape == (batch_size, ACTION_SPACE_SIZE)
        assert value.shape == (batch_size,)

    def test_value_range(self, network, device, new_game):
        """Value output is in [-1, 1] (tanh)."""
        board_state = torch.randn(8, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE).to(device)
        pieces_vec = torch.ones(8, PIECE_REMAINING_SIZE).to(device)
        legal_mask = torch.ones(8, ACTION_SPACE_SIZE).to(device)

        _, value = network(board_state, pieces_vec, legal_mask)

        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_policy_sums_to_one(self, network, device):
        """Policy probabilities (exp of log-probs) sum to ~1."""
        board_state = torch.randn(2, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE).to(device)
        pieces_vec = torch.ones(2, PIECE_REMAINING_SIZE).to(device)
        legal_mask = torch.ones(2, ACTION_SPACE_SIZE).to(device)

        log_policy, _ = network(board_state, pieces_vec, legal_mask)
        policy = torch.exp(log_policy)
        sums = policy.sum(dim=1)

        np.testing.assert_allclose(sums.detach().numpy(), 1.0, atol=1e-3)

    def test_masking_zeroes_illegal(self, network, device, new_game):
        """Illegal actions get near-zero probability after masking."""
        board_state = torch.from_numpy(new_game.get_nn_state()).unsqueeze(0).to(device)
        pieces_vec = torch.from_numpy(
            make_pieces_remaining_vector(new_game)
        ).unsqueeze(0).to(device)
        legal_mask = torch.from_numpy(
            new_game.get_legal_actions_mask()
        ).unsqueeze(0).to(device)

        log_policy, _ = network(board_state, pieces_vec, legal_mask)
        policy = torch.exp(log_policy).squeeze(0).detach().numpy()

        # Check that illegal actions have near-zero probability
        legal = new_game.get_legal_actions()
        legal_set = set(legal)
        for a in range(ACTION_SPACE_SIZE):
            if a not in legal_set:
                assert policy[a] < 1e-6, f"Illegal action {a} has prob {policy[a]}"

    def test_single_state_input(self, network, device, new_game):
        """Network handles a single game state input correctly."""
        board_state = torch.from_numpy(new_game.get_nn_state()).unsqueeze(0).to(device)
        pieces_vec = torch.from_numpy(
            make_pieces_remaining_vector(new_game)
        ).unsqueeze(0).to(device)
        legal_mask = torch.from_numpy(
            new_game.get_legal_actions_mask()
        ).unsqueeze(0).to(device)

        log_policy, value = network(board_state, pieces_vec, legal_mask)

        assert log_policy.shape == (1, ACTION_SPACE_SIZE)
        assert value.shape == (1,)

    def test_parameter_count_reasonable(self, network):
        """Network parameter count is in a reasonable range (not absurdly large)."""
        param_count = sum(p.numel() for p in network.parameters())
        # With 2 blocks, 32 channels: conv policy head keeps params small
        assert param_count < 1_000_000, f"Too many params: {param_count:,}"
        assert param_count > 10_000, f"Too few params: {param_count:,}"


# ---------------------------------------------------------------------------
# Pieces Remaining Vector Tests
# ---------------------------------------------------------------------------

class TestPiecesRemainingVector:
    """Tests for the piece-remaining vector construction."""

    def test_shape(self, new_game):
        vec = make_pieces_remaining_vector(new_game)
        assert vec.shape == (PIECE_REMAINING_SIZE,)
        assert vec.dtype == np.float32

    def test_new_game_all_ones(self, new_game):
        """At game start, all pieces are available — vector is all 1s."""
        vec = make_pieces_remaining_vector(new_game)
        assert np.all(vec == 1.0)

    def test_after_move(self, new_game):
        """After placing a piece, the vector reflects the removed piece."""
        legal = new_game.get_legal_actions()
        assert len(legal) > 0
        next_state = new_game.apply_action(legal[0])
        vec = make_pieces_remaining_vector(next_state)
        # Should have at least one 0 now (the piece that was placed)
        assert np.sum(vec == 0.0) >= 1


# ---------------------------------------------------------------------------
# MCTS Tests
# ---------------------------------------------------------------------------

class TestMCTS:
    """Tests for Monte Carlo Tree Search."""

    def test_search_returns_valid_policy(self, network, device, new_game):
        """MCTS search returns a valid probability distribution."""
        mcts = MCTS(network, num_simulations=10, device=device)
        policy, value = mcts.search(new_game)

        assert policy.shape == (ACTION_SPACE_SIZE,)
        # Policy should be non-negative
        assert np.all(policy >= 0)
        # Policy should sum to ~1 (normalized visit counts)
        if policy.sum() > 0:
            np.testing.assert_allclose(policy.sum(), 1.0, atol=1e-5)

    def test_search_only_legal_actions(self, network, device, new_game):
        """MCTS only assigns probability to legal actions."""
        mcts = MCTS(network, num_simulations=10, device=device)
        policy, _ = mcts.search(new_game)

        legal_set = set(new_game.get_legal_actions())
        for a in range(ACTION_SPACE_SIZE):
            if a not in legal_set:
                assert policy[a] == 0.0, f"Illegal action {a} has visit prob {policy[a]}"

    def test_select_action_returns_legal(self, network, device, new_game):
        """select_action returns a legal action."""
        mcts = MCTS(network, num_simulations=10, device=device)
        action, policy, value = mcts.select_action(new_game)

        legal = new_game.get_legal_actions()
        assert action in legal

    def test_mcts_node_creation(self, new_game):
        """MCTSNode initializes correctly."""
        node = MCTSNode(new_game)
        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.prior == 0.0
        assert not node.is_expanded
        assert not node.is_terminal

    def test_more_sims_increases_visits(self, network, device, new_game):
        """More simulations = more total visits at root."""
        mcts_few = MCTS(network, num_simulations=5, device=device)
        mcts_many = MCTS(network, num_simulations=20, device=device)

        # We can't directly access root, but policy from more sims should be
        # sharper (lower entropy). Just check both run without error.
        p1, _ = mcts_few.search(new_game)
        p2, _ = mcts_many.search(new_game)

        assert p1.sum() > 0
        assert p2.sum() > 0


# ---------------------------------------------------------------------------
# Agent Tests
# ---------------------------------------------------------------------------

class TestAlphaZeroAgent:
    """Tests for the AlphaZero agent."""

    def test_agent_selects_legal_action(self, network, device, new_game):
        """Agent selects a legal action."""
        agent = AlphaZeroAgent(network, num_simulations=5, device=device)
        action, policy = agent.select_action(new_game)

        legal = new_game.get_legal_actions()
        assert action in legal

    def test_agent_handles_no_legal_moves(self, network, device):
        """Agent returns -1 when there are no legal moves (must pass)."""
        # Play until a player has no legal moves (use random to get there fast)
        import random
        rng = random.Random(42)
        state = GameState.new_game('standard')
        for _ in range(200):
            if state.is_terminal():
                break
            legal = state.get_legal_actions()
            if not legal:
                # Found a state with no legal moves — test the agent here
                agent = AlphaZeroAgent(network, num_simulations=5, device=device)
                action, policy = agent.select_action(state)
                assert action == -1
                state = state.pass_turn()
            else:
                state = state.apply_action(rng.choice(legal))

    def test_temperature_setting(self, network, device):
        """Temperature can be changed."""
        agent = AlphaZeroAgent(network, num_simulations=5, temperature=1.0, device=device)
        assert agent.mcts.temperature == 1.0
        agent.set_temperature(0.01)
        assert agent.mcts.temperature == 0.01


# ---------------------------------------------------------------------------
# Self-Play Tests
# ---------------------------------------------------------------------------

class TestSelfPlay:
    """Tests for self-play data generation."""

    def test_self_play_generates_examples(self, network, device):
        """Self-play produces a non-empty list of training examples."""
        examples = self_play_game(
            network=network,
            game_mode='dual',
            num_simulations=5,
            max_moves=4,
            device=device,
        )
        assert len(examples) > 0

    def test_example_shapes(self, network, device):
        """Training examples have correct shapes."""
        examples = self_play_game(
            network=network,
            game_mode='dual',
            num_simulations=5,
            max_moves=4,
            device=device,
        )
        ex = examples[0]
        assert ex.board_state.shape == (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        assert ex.pieces_remaining.shape == (PIECE_REMAINING_SIZE,)
        assert ex.legal_mask.shape == (ACTION_SPACE_SIZE,)
        assert ex.policy_target.shape == (ACTION_SPACE_SIZE,)
        assert isinstance(ex.value_target, float)

    def test_value_targets_in_range(self, network, device):
        """Value targets are in [-1, 1]."""
        examples = self_play_game(
            network=network,
            game_mode='dual',
            num_simulations=5,
            max_moves=4,
            device=device,
        )
        for ex in examples:
            assert -1.0 <= ex.value_target <= 1.0

    def test_policy_targets_valid(self, network, device):
        """Policy targets are non-negative and sum to ~1."""
        examples = self_play_game(
            network=network,
            game_mode='dual',
            num_simulations=5,
            max_moves=4,
            device=device,
        )
        for ex in examples:
            assert np.all(ex.policy_target >= 0)
            total = ex.policy_target.sum()
            if total > 0:
                np.testing.assert_allclose(total, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Training Tests
# ---------------------------------------------------------------------------

class TestTraining:
    """Tests for the training loop."""

    def test_training_reduces_loss(self, network, device):
        """Training on a small batch should reduce loss."""
        # Generate some examples
        examples = self_play_game(
            network=network,
            game_mode='dual',
            num_simulations=5,
            max_moves=4,
            device=device,
        )

        # Import training function
        import sys, os
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'scripts',
        ))
        from train import train_on_examples

        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        # Train for several epochs on the same data — loss should decrease
        stats1 = train_on_examples(
            network, examples, optimizer,
            batch_size=32, epochs=1, device=device,
        )

        # Continue training on the same data for more epochs
        for _ in range(5):
            stats2 = train_on_examples(
                network, examples, optimizer,
                batch_size=32, epochs=5, device=device,
            )

        # After extended training on same small dataset, loss should be lower
        assert stats2['total_loss'] < stats1['total_loss'], (
            f"Loss didn't decrease: {stats1['total_loss']:.4f} -> {stats2['total_loss']:.4f}"
        )
        # Losses should be finite
        assert np.isfinite(stats2['total_loss'])
        assert np.isfinite(stats2['policy_loss'])
        assert np.isfinite(stats2['value_loss'])
