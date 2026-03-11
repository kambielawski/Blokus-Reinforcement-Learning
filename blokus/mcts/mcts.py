"""AlphaZero-style Monte Carlo Tree Search for Blokus.

PUCT-based MCTS that uses a neural network for leaf evaluation and
prior probability estimation. Follows the AlphaZero paper (Silver et al. 2018).
"""

import math
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple

from blokus.engine.game_state import GameState, ACTION_SPACE_SIZE, NUM_COLORS, NUM_PIECES


class MCTSNode:
    """A node in the MCTS search tree.

    Each node corresponds to a game state. Edges to children represent actions.
    """
    __slots__ = ['state', 'parent', 'action', 'children',
                 'visit_count', 'total_value', 'prior',
                 'is_expanded', 'is_terminal']

    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None,
                 action: int = -1, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action  # action that led to this node from parent
        self.prior = prior    # P(s, a) from the neural network
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.is_expanded = False
        self.is_terminal = state.is_terminal()

    @property
    def mean_value(self) -> float:
        """Q(s, a) = W(s, a) / N(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCTS:
    """Monte Carlo Tree Search with neural network guidance.

    Args:
        network: BlokusNetwork instance for leaf evaluation.
        c_puct: Exploration constant for the PUCT formula.
        num_simulations: Number of MCTS simulations per move.
        dirichlet_alpha: Alpha parameter for Dirichlet noise at root.
        dirichlet_epsilon: Weight of Dirichlet noise vs. network prior.
        temperature: Temperature for action selection from visit counts.
        device: PyTorch device for network inference.
    """

    def __init__(self, network, c_puct: float = 1.5,
                 num_simulations: int = 100,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 temperature: float = 1.0,
                 device: Optional[torch.device] = None):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.device = device or torch.device('cpu')

    def search(self, state: GameState) -> Tuple[np.ndarray, float]:
        """Run MCTS from the given state.

        Returns:
            policy: (ACTION_SPACE_SIZE,) visit count distribution (normalized).
            value: Root value estimate.
        """
        root = MCTSNode(state)
        self._expand(root, add_noise=True)

        for _ in range(self.num_simulations):
            node = root
            # Selection: traverse tree using PUCT
            while node.is_expanded and not node.is_terminal:
                node = self._select_child(node)
            # Expansion and evaluation
            if not node.is_terminal:
                value = self._expand(node)
            else:
                # Terminal node: use actual game reward
                rewards = node.state.get_rewards()
                # Value from perspective of the player who moved TO this state
                # (i.e., the parent's current agent)
                if node.parent is not None:
                    agent = node.parent.state.get_current_agent()
                else:
                    agent = node.state.get_current_agent()
                value = rewards.get(agent, 0.0)
            # Backup
            self._backup(node, value)

        return self._get_policy(root), root.mean_value

    def select_action(self, state: GameState) -> Tuple[int, np.ndarray, float]:
        """Select an action using MCTS.

        Returns:
            action: Selected action index.
            policy: MCTS visit count distribution.
            value: Root value estimate.
        """
        policy, value = self.search(state)

        legal = state.get_legal_actions()
        if not legal:
            return -1, policy, value  # must pass

        if self.temperature < 0.01:
            # Greedy: pick highest visit count
            action = legal[0]
            best_count = 0.0
            for a in legal:
                if policy[a] > best_count:
                    best_count = policy[a]
                    action = a
        else:
            # Sample proportional to visit counts raised to 1/temperature
            probs = np.zeros(len(legal), dtype=np.float64)
            for i, a in enumerate(legal):
                probs[i] = policy[a]
            # Apply temperature
            if self.temperature != 1.0:
                probs = np.power(probs + 1e-10, 1.0 / self.temperature)
            probs /= probs.sum()
            idx = np.random.choice(len(legal), p=probs)
            action = legal[idx]

        return action, policy, value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest PUCT score."""
        best_score = -float('inf')
        best_child = None
        sqrt_parent = math.sqrt(node.visit_count)

        for child in node.children.values():
            # PUCT formula
            q = child.mean_value
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode, add_noise: bool = False) -> float:
        """Expand a leaf node using the neural network.

        Returns the value estimate from the network.
        """
        state = node.state

        # Handle pass (no legal moves)
        legal = state.get_legal_actions()
        if not legal:
            # If a color must pass, the node has one child: the pass state
            pass_state = state.pass_turn()
            child = MCTSNode(pass_state, parent=node, action=-1, prior=1.0)
            node.children[-1] = child
            node.is_expanded = True
            # Evaluate via network anyway for value
            return self._evaluate(state)

        # Get network predictions
        policy_probs, value = self._evaluate_with_policy(state)

        # Extract priors for legal actions only
        legal_priors = {}
        total = 0.0
        for a in legal:
            legal_priors[a] = policy_probs[a]
            total += policy_probs[a]

        # Renormalize
        if total > 0:
            for a in legal_priors:
                legal_priors[a] /= total
        else:
            # Uniform if network gives zero everywhere
            uniform = 1.0 / len(legal)
            for a in legal_priors:
                legal_priors[a] = uniform

        # Add Dirichlet noise at root
        if add_noise and len(legal) > 0:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(legal)
            )
            eps = self.dirichlet_epsilon
            for i, a in enumerate(legal):
                legal_priors[a] = (1 - eps) * legal_priors[a] + eps * noise[i]

        # Create children
        for a in legal:
            child_state = state.apply_action(a)
            child = MCTSNode(child_state, parent=node, action=a,
                             prior=legal_priors[a])
            node.children[a] = child

        node.is_expanded = True
        return value

    def _evaluate(self, state: GameState) -> float:
        """Get value estimate from network for a state."""
        _, value = self._evaluate_with_policy(state)
        return value

    @torch.no_grad()
    def _evaluate_with_policy(self, state: GameState) -> Tuple[np.ndarray, float]:
        """Get policy and value from the neural network.

        Returns:
            policy: (ACTION_SPACE_SIZE,) probability distribution.
            value: scalar in [-1, 1].
        """
        from blokus.nn.network import make_pieces_remaining_vector

        board_state = torch.from_numpy(
            state.get_nn_state()
        ).unsqueeze(0).to(self.device)

        pieces_vec = torch.from_numpy(
            make_pieces_remaining_vector(state)
        ).unsqueeze(0).to(self.device)

        legal_mask = torch.from_numpy(
            state.get_legal_actions_mask()
        ).unsqueeze(0).to(self.device)

        self.network.eval()
        log_policy, value = self.network(board_state, pieces_vec, legal_mask)

        # Convert log-probs to probs
        policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
        value = value.item()

        return policy, value

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Propagate value back up the tree.

        Value is negated at each level because players alternate
        (what's good for one player is bad for the opponent).
        In 4-player Blokus, we track value from the perspective of the
        agent at the root.
        """
        # For multi-player, we need to track whose perspective the value is from.
        # The value from _expand is from the current agent's perspective.
        # We propagate the actual value up, flipping sign based on whether
        # the node's agent matches the leaf's agent.
        leaf_agent = node.state.get_current_agent()

        current = node
        while current is not None:
            current.visit_count += 1
            # Determine if this node's controlling agent matches the leaf
            if current.state.get_current_agent() == leaf_agent:
                current.total_value += value
            else:
                current.total_value -= value
            current = current.parent

    def _get_policy(self, root: MCTSNode) -> np.ndarray:
        """Extract visit count distribution from root node."""
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        total = 0
        for action, child in root.children.items():
            if action >= 0:  # skip pass actions
                policy[action] = child.visit_count
                total += child.visit_count
        if total > 0:
            policy /= total
        return policy
