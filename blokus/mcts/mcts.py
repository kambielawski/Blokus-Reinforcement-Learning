"""AlphaZero-style Monte Carlo Tree Search for Blokus.

PUCT-based MCTS that uses a neural network for leaf evaluation and
prior probability estimation. Follows the AlphaZero paper (Silver et al. 2018).

Optimizations:
- Lazy child creation: child GameStates are only created when visited
- Single legal-move computation: legal actions and mask share one call
- Batched NN inference: multiple leaves evaluated in one forward pass
  using virtual losses to encourage exploration diversity
"""

import math
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple

from blokus.engine.game_state import (
    GameState, ACTION_SPACE_SIZE, NUM_COLORS, NUM_PIECES,
)


class MCTSNode:
    """A node in the MCTS search tree.

    Children use lazy state creation: the GameState for a child is only
    computed (via apply_action) when that child is first selected.
    """
    __slots__ = [
        'state', 'parent', 'action', 'children',
        'visit_count', 'total_value', 'prior',
        'is_expanded', 'is_terminal',
        '_parent_state',  # parent's GameState, for lazy child creation
    ]

    def __init__(self, state: Optional[GameState],
                 parent: Optional['MCTSNode'] = None,
                 action: int = -1, prior: float = 0.0,
                 parent_state: Optional[GameState] = None):
        self.state = state  # None until lazily created for non-root nodes
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.is_expanded = False
        self.is_terminal = False
        self._parent_state = parent_state

        if state is not None:
            self.is_terminal = state.is_terminal()

    def ensure_state(self) -> GameState:
        """Lazily create the GameState if not yet materialized."""
        if self.state is None:
            if self.action == -1:
                # Pass action
                self.state = self._parent_state.pass_turn()
            else:
                self.state = self._parent_state.apply_action(self.action)
            self.is_terminal = self.state.is_terminal()
            self._parent_state = None  # release reference
        return self.state

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
        batch_size: Number of leaves to batch for NN inference (0 = no batching).
    """

    def __init__(self, network, c_puct: float = 1.5,
                 num_simulations: int = 100,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 temperature: float = 1.0,
                 device: Optional[torch.device] = None,
                 batch_size: int = 8):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.device = device or torch.device('cpu')
        self.batch_size = batch_size
        # Virtual loss magnitude — discourages re-selecting the same leaf
        self._virtual_loss = 3.0

    def search(self, state: GameState) -> Tuple[np.ndarray, float]:
        """Run MCTS from the given state.

        Returns:
            policy: (ACTION_SPACE_SIZE,) visit count distribution (normalized).
            value: Root value estimate.
        """
        root = MCTSNode(state)
        # Expand root synchronously (need priors before batched search)
        self._expand_single(root, add_noise=True)

        sims_done = 0
        while sims_done < self.num_simulations:
            if self.batch_size > 1:
                sims_done += self._run_batched_sims(root, sims_done)
            else:
                self._run_single_sim(root)
                sims_done += 1

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
            return -1, policy, value

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
            if self.temperature != 1.0:
                probs = np.power(probs + 1e-10, 1.0 / self.temperature)
            probs /= probs.sum()
            idx = np.random.choice(len(legal), p=probs)
            action = legal[idx]

        return action, policy, value

    # ------------------------------------------------------------------
    # Single-sim path (batch_size <= 1)
    # ------------------------------------------------------------------

    def _run_single_sim(self, root: MCTSNode) -> None:
        """Run one MCTS simulation: select → expand → backup."""
        node = root
        while node.is_expanded and not node.is_terminal:
            node = self._select_child(node)

        if not node.is_terminal:
            node.ensure_state()
            value = self._expand_single(node)
        else:
            value = self._terminal_value(node)

        self._backup(node, value)

    # ------------------------------------------------------------------
    # Batched-sim path (batch_size > 1)
    # ------------------------------------------------------------------

    def _run_batched_sims(self, root: MCTSNode, sims_done: int) -> int:
        """Run a batch of MCTS simulations with virtual losses.

        Selects up to batch_size leaves, applies virtual losses during
        selection to encourage diversity, evaluates them in one NN call,
        then backs up real values and removes virtual losses.

        Returns the number of simulations completed.
        """
        remaining = self.num_simulations - sims_done
        batch_target = min(self.batch_size, remaining)

        leaves: List[MCTSNode] = []
        terminal_leaves: List[Tuple[MCTSNode, float]] = []

        for _ in range(batch_target):
            node = root
            # Selection with virtual loss applied to previously selected paths
            while node.is_expanded and not node.is_terminal:
                node = self._select_child(node)

            if node.is_terminal:
                value = self._terminal_value(node)
                terminal_leaves.append((node, value))
                # Apply virtual loss so we don't keep selecting this terminal
                node.visit_count += 1
                node.total_value -= self._virtual_loss
            else:
                node.ensure_state()
                leaves.append(node)
                # Apply virtual loss to discourage re-selecting this path
                self._apply_virtual_loss(node)

        # Backup terminal nodes immediately
        for node, value in terminal_leaves:
            # Remove the temporary visit/loss we added
            node.visit_count -= 1
            node.total_value += self._virtual_loss
            self._backup(node, value)

        if not leaves:
            return len(terminal_leaves)

        # Batched NN evaluation
        values = self._expand_batch(leaves)

        # Remove virtual losses and backup real values
        for node, value in zip(leaves, values):
            self._remove_virtual_loss(node)
            self._backup(node, value)

        return len(leaves) + len(terminal_leaves)

    def _apply_virtual_loss(self, node: MCTSNode) -> None:
        """Apply virtual loss along path from node to root."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value -= self._virtual_loss
            current = current.parent

    def _remove_virtual_loss(self, node: MCTSNode) -> None:
        """Remove virtual loss along path from node to root."""
        current = node
        while current is not None:
            current.visit_count -= 1
            current.total_value += self._virtual_loss
            current = current.parent

    # ------------------------------------------------------------------
    # Tree operations
    # ------------------------------------------------------------------

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest PUCT score."""
        best_score = -float('inf')
        best_child = None
        sqrt_parent = math.sqrt(node.visit_count)

        for child in node.children.values():
            q = child.mean_value
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand_single(self, node: MCTSNode, add_noise: bool = False) -> float:
        """Expand a leaf node using a single NN evaluation.

        Returns the value estimate.
        """
        state = node.ensure_state()

        legal = state.get_legal_actions()
        if not legal:
            pass_state = state.pass_turn()
            child = MCTSNode(pass_state, parent=node, action=-1, prior=1.0)
            node.children[-1] = child
            node.is_expanded = True
            return self._evaluate_single(state, legal)

        # Build legal actions mask from the known legal list (avoid recomputation)
        legal_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for a in legal:
            legal_mask[a] = 1.0

        policy_probs, value = self._evaluate_single(state, legal, legal_mask)

        self._create_children(node, state, legal, policy_probs, add_noise)
        node.is_expanded = True
        return value

    def _expand_batch(self, leaves: List[MCTSNode]) -> List[float]:
        """Expand multiple leaf nodes with one batched NN evaluation.

        Returns list of value estimates, one per leaf.
        """
        from blokus.nn.network import make_pieces_remaining_vector

        n = len(leaves)
        # Pre-allocate numpy arrays for the batch
        board_states = np.empty((n, 5, 20, 20), dtype=np.float32)
        pieces_vecs = np.empty((n, NUM_PIECES * NUM_COLORS), dtype=np.float32)
        legal_masks = np.zeros((n, ACTION_SPACE_SIZE), dtype=np.float32)

        legal_actions_list: List[List[int]] = []

        for i, node in enumerate(leaves):
            state = node.state  # already ensured
            legal = state.get_legal_actions()
            legal_actions_list.append(legal)

            board_states[i] = state.get_nn_state()
            pieces_vecs[i] = make_pieces_remaining_vector(state)
            for a in legal:
                legal_masks[i, a] = 1.0

        # Single batched NN forward pass
        policies, values = self._nn_forward_batch(board_states, pieces_vecs, legal_masks)

        # Create children for each leaf
        for i, node in enumerate(leaves):
            state = node.state
            legal = legal_actions_list[i]

            if not legal:
                pass_state = state.pass_turn()
                child = MCTSNode(pass_state, parent=node, action=-1, prior=1.0)
                node.children[-1] = child
            else:
                self._create_children(
                    node, state, legal, policies[i],
                    add_noise=False,
                )
            node.is_expanded = True

        return values

    def _create_children(self, node: MCTSNode, state: GameState,
                         legal: List[int], policy_probs: np.ndarray,
                         add_noise: bool) -> None:
        """Create lazy child nodes with priors from the policy."""
        # Extract and renormalize priors for legal actions
        priors = np.array([policy_probs[a] for a in legal], dtype=np.float64)
        total = priors.sum()
        if total > 0:
            priors /= total
        else:
            priors[:] = 1.0 / len(legal)

        # Add Dirichlet noise at root
        if add_noise and len(legal) > 0:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(legal)
            )
            eps = self.dirichlet_epsilon
            priors = (1 - eps) * priors + eps * noise

        # Create children lazily (no apply_action yet)
        for i, a in enumerate(legal):
            child = MCTSNode(
                state=None,  # lazy — created on first visit
                parent=node,
                action=a,
                prior=float(priors[i]),
                parent_state=state,
            )
            node.children[a] = child

    # ------------------------------------------------------------------
    # NN inference
    # ------------------------------------------------------------------

    def _evaluate_single(self, state: GameState,
                         legal: List[int],
                         legal_mask: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, float]:
        """Evaluate a single state with the NN.

        Accepts pre-computed legal actions list and optional mask to
        avoid redundant get_legal_actions() calls.
        """
        from blokus.nn.network import make_pieces_remaining_vector

        board_np = state.get_nn_state()
        pieces_np = make_pieces_remaining_vector(state)

        if legal_mask is None:
            legal_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            for a in legal:
                legal_mask[a] = 1.0

        policies, values = self._nn_forward_batch(
            board_np[np.newaxis],
            pieces_np[np.newaxis],
            legal_mask[np.newaxis],
        )
        return policies[0], values[0]

    @torch.no_grad()
    def _nn_forward_batch(self, board_states: np.ndarray,
                          pieces_vecs: np.ndarray,
                          legal_masks: np.ndarray
                          ) -> Tuple[np.ndarray, List[float]]:
        """Run batched NN forward pass.

        Args:
            board_states: (N, 5, 20, 20) float32
            pieces_vecs: (N, 84) float32
            legal_masks: (N, 67200) float32

        Returns:
            policies: (N, 67200) probability arrays
            values: list of N floats
        """
        board_t = torch.from_numpy(board_states).to(self.device)
        pieces_t = torch.from_numpy(pieces_vecs).to(self.device)
        mask_t = torch.from_numpy(legal_masks).to(self.device)

        self.network.eval()
        log_policy, value = self.network(board_t, pieces_t, mask_t)

        policies = torch.exp(log_policy).cpu().numpy()
        values = value.cpu().tolist()

        return policies, values

    def _terminal_value(self, node: MCTSNode) -> float:
        """Get value for a terminal node from actual game rewards."""
        state = node.ensure_state()
        rewards = state.get_rewards()
        if node.parent is not None:
            agent = node.parent.ensure_state().get_current_agent()
        else:
            agent = state.get_current_agent()
        return rewards.get(agent, 0.0)

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Propagate value back up the tree."""
        leaf_agent = node.ensure_state().get_current_agent()

        current = node
        while current is not None:
            current.visit_count += 1
            if current.ensure_state().get_current_agent() == leaf_agent:
                current.total_value += value
            else:
                current.total_value -= value
            current = current.parent

    def _get_policy(self, root: MCTSNode) -> np.ndarray:
        """Extract visit count distribution from root node."""
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        total = 0
        for action, child in root.children.items():
            if action >= 0:
                policy[action] = child.visit_count
                total += child.visit_count
        if total > 0:
            policy /= total
        return policy
