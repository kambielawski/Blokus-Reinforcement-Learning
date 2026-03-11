"""AlphaZero agent for Blokus.

Uses a neural network + MCTS to select moves, and provides self-play
infrastructure for generating training data.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

from blokus.engine.game_state import GameState, ACTION_SPACE_SIZE
from blokus.nn.network import BlokusNetwork, make_pieces_remaining_vector
from blokus.mcts.mcts import MCTS


@dataclass
class TrainingExample:
    """A single training example from self-play.

    Attributes:
        board_state: (5, 20, 20) float32 from get_nn_state().
        pieces_remaining: (84,) float32 binary vector.
        legal_mask: (67200,) float32 binary mask.
        policy_target: (67200,) float32 MCTS visit count distribution.
        value_target: float in [-1, 1] — game outcome from this player's perspective.
    """
    board_state: np.ndarray
    pieces_remaining: np.ndarray
    legal_mask: np.ndarray
    policy_target: np.ndarray
    value_target: float


class AlphaZeroAgent:
    """Agent that selects moves using neural network + MCTS.

    Args:
        network: BlokusNetwork model.
        num_simulations: Number of MCTS simulations per move.
        c_puct: PUCT exploration constant.
        dirichlet_alpha: Dirichlet noise alpha (0 to disable).
        dirichlet_epsilon: Weight of Dirichlet noise.
        temperature: Move selection temperature (1.0 = proportional, ~0 = greedy).
        device: PyTorch device.
    """

    def __init__(self, network: BlokusNetwork,
                 num_simulations: int = 100,
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 temperature: float = 1.0,
                 device: Optional[torch.device] = None):
        self.network = network
        self.device = device or torch.device('cpu')
        self.mcts = MCTS(
            network=network,
            c_puct=c_puct,
            num_simulations=num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            temperature=temperature,
            device=self.device,
        )

    def select_action(self, state: GameState) -> Tuple[int, np.ndarray]:
        """Select an action for the current state.

        Returns:
            action: Selected action index (-1 for pass).
            policy: MCTS visit count distribution.
        """
        legal = state.get_legal_actions()
        if not legal:
            return -1, np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)

        action, policy, _ = self.mcts.select_action(state)
        return action, policy

    def set_temperature(self, temperature: float) -> None:
        """Update the move selection temperature."""
        self.mcts.temperature = temperature


def self_play_game(network: BlokusNetwork,
                   game_mode: str = 'dual',
                   num_simulations: int = 100,
                   c_puct: float = 1.5,
                   temp_threshold: int = 15,
                   max_moves: int = 0,
                   device: Optional[torch.device] = None
                   ) -> List[TrainingExample]:
    """Play a complete self-play game and collect training examples.

    All agents share the same network. Temperature is 1.0 for the first
    `temp_threshold` moves, then drops to ~0 for exploitation.

    Args:
        network: BlokusNetwork model (shared by all agents).
        game_mode: 'standard' or 'dual'.
        num_simulations: MCTS simulations per move.
        c_puct: PUCT exploration constant.
        temp_threshold: After this many moves, switch to greedy play.
        max_moves: Stop after this many moves (0 = play to completion).
        device: PyTorch device.

    Returns:
        List of TrainingExample objects for training.
    """
    device = device or torch.device('cpu')
    agent = AlphaZeroAgent(
        network=network,
        num_simulations=num_simulations,
        c_puct=c_puct,
        temperature=1.0,
        device=device,
    )

    state = GameState.new_game(game_mode)

    # Collect (state_data, policy, agent_idx) during play
    game_history: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    move_count = 0

    while not state.is_terminal():
        if max_moves > 0 and move_count >= max_moves:
            break

        legal = state.get_legal_actions()

        if not legal:
            state = state.pass_turn()
            continue

        # Switch to greedy after threshold
        if move_count >= temp_threshold:
            agent.set_temperature(0.01)

        # Record state before acting
        board_state = state.get_nn_state()
        pieces_vec = make_pieces_remaining_vector(state)
        legal_mask = state.get_legal_actions_mask()
        current_agent = state.get_current_agent()

        # MCTS action selection
        action, policy = agent.select_action(state)

        game_history.append((
            board_state, pieces_vec, legal_mask, policy, current_agent
        ))

        state = state.apply_action(action)
        move_count += 1

    # Game over: get final rewards
    rewards = state.get_rewards()

    # Build training examples with actual game outcome as value target
    examples: List[TrainingExample] = []
    for board_state, pieces_vec, legal_mask, policy, agent_idx in game_history:
        value_target = rewards.get(agent_idx, 0.0)
        examples.append(TrainingExample(
            board_state=board_state,
            pieces_remaining=pieces_vec,
            legal_mask=legal_mask,
            policy_target=policy,
            value_target=value_target,
        ))

    return examples
