#!/usr/bin/env python3
"""AlphaZero training loop for Blokus.

Supports YAML config files with CLI overrides, multi-process self-play,
robust per-iteration checkpointing, and optional W&B logging.

Usage:
    # Use a config file
    python scripts/train.py --config configs/full.yaml

    # Override specific values
    python scripts/train.py --config configs/full.yaml --lr 0.0005 --sims 200

    # Pure CLI (uses configs/default.yaml as base)
    python scripts/train.py --iterations 10 --games-per-iter 4 --sims 25

    # Resume from latest checkpoint
    python scripts/train.py --config configs/full.yaml --resume
"""

import argparse
import glob
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import yaml
from typing import List, Dict, Any, Optional

# Add repo root to path for script execution
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from blokus.nn.network import BlokusNetwork, make_pieces_remaining_vector
from blokus.agents.alpha_zero import self_play_game, TrainingExample
from blokus.engine.game_state import GameState, ACTION_SPACE_SIZE
from blokus.mcts.mcts import MCTS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config, falling back to default.yaml if no path given."""
    default_path = os.path.join(REPO_ROOT, 'configs', 'default.yaml')
    with open(default_path) as f:
        cfg = yaml.safe_load(f)

    if config_path and os.path.abspath(config_path) != os.path.abspath(default_path):
        with open(config_path) as f:
            override = yaml.safe_load(f) or {}
        _deep_merge(cfg, override)

    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base dict (mutates base)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to the config dict."""
    # Map flat CLI args to nested config paths
    overrides = {
        'iterations':      ('training', 'iterations'),
        'games_per_iter':  ('self_play', 'games_per_iteration'),
        'sims':            ('mcts', 'num_simulations'),
        'lr':              ('training', 'learning_rate'),
        'batch_size':      ('training', 'batch_size'),
        'epochs':          ('training', 'epochs_per_iter'),
        'num_blocks':      ('network', 'num_blocks'),
        'channels':        ('network', 'channels'),
        'game_mode':       ('self_play', 'game_mode'),
        'device':          ('device',),
        'save_dir':        ('checkpoint', 'dir'),
        'num_workers':     ('self_play', 'num_workers'),
        'wandb':           ('wandb', 'enabled'),
        'wandb_project':   ('wandb', 'project'),
        'wandb_run_name':  ('wandb', 'run_name'),
        'c_puct':          ('mcts', 'c_puct'),
        'max_moves':       ('self_play', 'max_moves'),
        'buffer_size':     ('training', 'replay_buffer_size'),
        'value_loss_weight': ('training', 'value_loss_weight'),
    }
    for arg_name, path in overrides.items():
        val = getattr(args, arg_name, None)
        if val is None:
            continue
        # For store_true args, only override if explicitly set
        if isinstance(val, bool) and arg_name not in sys.argv:
            # Check if --flag was explicitly passed
            flag = f'--{arg_name.replace("_", "-")}'
            if flag not in sys.argv and f'--{arg_name}' not in sys.argv:
                continue

        target = cfg
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = val


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages saving/loading checkpoints with best-model tracking."""

    def __init__(self, save_dir: str, save_every: int = 1, keep_top_k: int = 3):
        self.save_dir = save_dir
        self.save_every = save_every
        self.keep_top_k = keep_top_k
        os.makedirs(save_dir, exist_ok=True)

    def save(self, iteration: int, network: nn.Module,
             optimizer: optim.Optimizer, meta: Dict[str, Any],
             cfg: Dict[str, Any]) -> Optional[str]:
        """Save checkpoint if due. Always saves 'latest'. Returns path or None.

        The replay buffer is saved to a separate file (replay_buffer.npz) to
        keep per-iteration checkpoints small (~19 MB vs ~13+ GB).
        """
        if iteration % self.save_every != 0:
            return None

        # Separate replay buffer from the main checkpoint payload
        replay_state = meta.pop('replay_buffer', None)

        payload = {
            'iteration': iteration,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': cfg,
            **meta,
        }

        # Always save as 'latest'
        latest_path = os.path.join(self.save_dir, 'checkpoint_latest.pt')
        torch.save(payload, latest_path)

        # Save numbered checkpoint
        iter_path = os.path.join(self.save_dir, f'checkpoint_{iteration:04d}.pt')
        torch.save(payload, iter_path)

        # Save replay buffer separately using np.savez (streams arrays to
        # disk without pickle).
        if replay_state is not None:
            buf_path = os.path.join(self.save_dir, 'replay_buffer.npz')
            save_kwargs = {
                'max_size': np.array(replay_state['max_size']),
                'size': np.array(replay_state['size']),
                'board_states': replay_state['board_states'],
                'pieces_remaining': replay_state['pieces_remaining'],
                'value_targets': replay_state['value_targets'],
            }
            # Sparse CSR format (new) or legacy dense format
            if 'sparse_offsets' in replay_state:
                save_kwargs['sparse_offsets'] = replay_state['sparse_offsets']
                save_kwargs['sparse_indices'] = replay_state['sparse_indices']
                save_kwargs['sparse_policy'] = replay_state['sparse_policy']
            else:
                save_kwargs['legal_masks'] = replay_state['legal_masks']
                save_kwargs['policy_targets'] = replay_state['policy_targets']
            np.savez(buf_path, **save_kwargs)

        return iter_path

    def update_best(self, iteration: int, network: nn.Module,
                    optimizer: optim.Optimizer, meta: Dict[str, Any],
                    cfg: Dict[str, Any], policy_loss: float) -> None:
        """Track top-K checkpoints by lowest policy loss."""
        best_path = os.path.join(self.save_dir, 'best_models.yaml')
        entries = []
        if os.path.exists(best_path):
            with open(best_path) as f:
                entries = yaml.safe_load(f) or []

        entry = {'iteration': iteration, 'policy_loss': float(policy_loss)}
        entries.append(entry)
        entries.sort(key=lambda e: e['policy_loss'])

        # Keep top K
        to_keep = set()
        for e in entries[:self.keep_top_k]:
            to_keep.add(e['iteration'])

        # Remove checkpoints outside top K (but never remove 'latest')
        for e in entries[self.keep_top_k:]:
            old_path = os.path.join(self.save_dir, f'checkpoint_{e["iteration"]:04d}.pt')
            if os.path.exists(old_path):
                os.remove(old_path)

        entries = entries[:self.keep_top_k]
        with open(best_path, 'w') as f:
            yaml.dump(entries, f)

        # Save/update the best checkpoint file
        if entries and entries[0]['iteration'] == iteration:
            best_ckpt_path = os.path.join(self.save_dir, 'checkpoint_best.pt')
            # Exclude replay buffer from best checkpoint (saved separately)
            meta_clean = {k: v for k, v in meta.items() if k != 'replay_buffer'}
            payload = {
                'iteration': iteration,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                **meta_clean,
            }
            torch.save(payload, best_ckpt_path)

    def load_latest(self, device: torch.device) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint if it exists."""
        latest_path = os.path.join(self.save_dir, 'checkpoint_latest.pt')
        if os.path.exists(latest_path):
            return torch.load(latest_path, map_location=device, weights_only=False)
        return None

    def load(self, path: str, device: torch.device) -> Dict[str, Any]:
        """Load a specific checkpoint."""
        return torch.load(path, map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Rolling FIFO buffer with sparse storage for legal masks and policies.

    Dense columns (board_state, pieces_remaining, value_target) use
    pre-allocated numpy arrays with geometric growth.  Legal masks and
    policy targets are stored as sparse index/value pairs (lists of small
    arrays), reducing per-example memory from ~533 KB to ~14 KB.

    Memory layout (per example, avg ~500 legal actions):
        board_state:      5 * 20 * 20 = 2,000 floats  (8 KB)
        pieces_remaining: 84 floats                    (336 B)
        legal_indices:    ~500 int32                   (~2 KB)
        policy_values:    ~500 float32                 (~2 KB)
        value_target:     1 float                      (4 B)
        Total: ~12 KB/example, ~2.4 GB for 200K examples
    """

    _BOARD_SHAPE = (5, 20, 20)
    _PIECES_SHAPE = (84,)
    _ACTION_SIZE = 67200

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._size = 0
        # Dense columnar storage — allocated lazily on first add/load
        self._board_states: Optional[np.ndarray] = None
        self._pieces_remaining: Optional[np.ndarray] = None
        self._value_targets: Optional[np.ndarray] = None
        # Sparse storage — lists of small arrays, one per example
        self._legal_indices: List[np.ndarray] = []   # each (nnz,) int32
        self._policy_values: List[np.ndarray] = []   # each (nnz,) float32

    def _grow_and_copy(self, new_cap: int):
        """Reallocate dense arrays to new_cap, preserving existing data."""
        n = self._size
        old_bs = self._board_states
        old_pr = self._pieces_remaining
        old_vt = self._value_targets

        self._board_states = np.empty((new_cap, *self._BOARD_SHAPE), dtype=np.float32)
        self._pieces_remaining = np.empty((new_cap, *self._PIECES_SHAPE), dtype=np.float32)
        self._value_targets = np.empty(new_cap, dtype=np.float32)

        if old_bs is not None and n > 0:
            self._board_states[:n] = old_bs[:n]
            self._pieces_remaining[:n] = old_pr[:n]
            self._value_targets[:n] = old_vt[:n]

    def add(self, examples: List[TrainingExample]) -> int:
        """Add examples, dropping oldest if over capacity. Returns count dropped."""
        n_new = len(examples)
        if n_new == 0:
            return 0

        old_size = self._size
        total = old_size + n_new
        dropped = max(0, total - self.max_size)
        new_size = min(total, self.max_size)

        keep_old = old_size - dropped if dropped < old_size else 0
        skip_new = max(0, n_new - self.max_size)

        # Ensure dense capacity
        cur_cap = self._board_states.shape[0] if self._board_states is not None else 0
        if cur_cap < new_size:
            new_cap = min(max(new_size, cur_cap * 2), self.max_size)
            new_cap = max(new_cap, new_size)
            self._grow_and_copy(new_cap)

        # Drop oldest from sparse lists
        if dropped > 0:
            drop_count = min(dropped, len(self._legal_indices))
            self._legal_indices = self._legal_indices[drop_count:]
            self._policy_values = self._policy_values[drop_count:]

        # Shift surviving dense data to front
        if dropped > 0 and keep_old > 0:
            src_start = old_size - keep_old
            self._board_states[:keep_old] = self._board_states[src_start:old_size]
            self._pieces_remaining[:keep_old] = self._pieces_remaining[src_start:old_size]
            self._value_targets[:keep_old] = self._value_targets[src_start:old_size]

        # Add new examples
        dst = keep_old
        for i in range(skip_new, n_new):
            ex = examples[i]
            self._board_states[dst] = ex.board_state
            self._pieces_remaining[dst] = ex.pieces_remaining
            self._value_targets[dst] = ex.value_target
            # Convert dense → sparse
            nonzero = np.nonzero(ex.legal_mask)[0].astype(np.int32)
            self._legal_indices.append(nonzero)
            self._policy_values.append(ex.policy_target[nonzero].astype(np.float32).copy())
            dst += 1

        self._size = new_size
        return dropped

    def __len__(self) -> int:
        return self._size

    @property
    def value_targets_array(self) -> np.ndarray:
        """Direct view of value targets (no copy)."""
        return self._value_targets[:self._size]

    def _reconstruct_dense(self, idx: int):
        """Reconstruct dense legal_mask and policy_target for one example."""
        legal_mask = np.zeros(self._ACTION_SIZE, dtype=np.float32)
        policy_target = np.zeros(self._ACTION_SIZE, dtype=np.float32)
        indices = self._legal_indices[idx]
        legal_mask[indices] = 1.0
        policy_target[indices] = self._policy_values[idx]
        return legal_mask, policy_target

    def reconstruct_sparse_batch(self, indices: np.ndarray):
        """Reconstruct dense legal_masks and policy_targets for a mini-batch.

        Args:
            indices: 1D array of example indices (int).

        Returns:
            (legal_masks, policy_targets) as numpy arrays of shape
            (len(indices), ACTION_SIZE), dtype float32.
        """
        n = len(indices)
        legal_masks = np.zeros((n, self._ACTION_SIZE), dtype=np.float32)
        policy_targets = np.zeros((n, self._ACTION_SIZE), dtype=np.float32)
        for i, idx in enumerate(indices):
            li = self._legal_indices[idx]
            legal_masks[i, li] = 1.0
            policy_targets[i, li] = self._policy_values[idx]
        return legal_masks, policy_targets

    def get_dense_tensors(self):
        """Return dense columns as CPU tensors (zero-copy via from_numpy).

        Returns (board_states, pieces_vecs, value_targets).
        """
        n = self._size
        return (
            torch.from_numpy(self._board_states[:n]),
            torch.from_numpy(self._pieces_remaining[:n]),
            torch.from_numpy(self._value_targets[:n]),
        )

    def sample(self, n: int) -> List[TrainingExample]:
        """Sample n examples uniformly at random (for tests/compat)."""
        if self._size == 0:
            return []
        indices = np.random.randint(0, self._size, size=min(n, self._size))
        results = []
        for i in indices:
            lm, pt = self._reconstruct_dense(i)
            results.append(TrainingExample(
                board_state=self._board_states[i],
                pieces_remaining=self._pieces_remaining[i],
                legal_mask=lm,
                policy_target=pt,
                value_target=float(self._value_targets[i]),
            ))
        return results

    def get_all(self) -> List[TrainingExample]:
        """Reconstruct list of TrainingExample objects (for tests/compat).

        WARNING: This creates per-example Python objects with dense arrays.
        Do NOT use in the hot training path.
        """
        n = self._size
        results = []
        for i in range(n):
            lm, pt = self._reconstruct_dense(i)
            results.append(TrainingExample(
                board_state=self._board_states[i],
                pieces_remaining=self._pieces_remaining[i],
                legal_mask=lm,
                policy_target=pt,
                value_target=float(self._value_targets[i]),
            ))
        return results

    def state_dict(self) -> dict:
        """Serialize for checkpointing using CSR-style sparse format.

        Sparse data is flattened into three arrays: offsets, indices, values.
        This is compact and compatible with np.savez streaming.
        """
        if self._size == 0:
            return {'max_size': self.max_size, 'size': 0}
        n = self._size
        cap = self._board_states.shape[0]
        # Dense columns
        if n == cap:
            bs, pr, vt = self._board_states, self._pieces_remaining, self._value_targets
        else:
            bs = self._board_states[:n].copy()
            pr = self._pieces_remaining[:n].copy()
            vt = self._value_targets[:n].copy()
        # Sparse columns → CSR
        offsets = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            offsets[i + 1] = offsets[i] + len(self._legal_indices[i])
        total_nnz = int(offsets[n])
        flat_indices = np.empty(total_nnz, dtype=np.int32)
        flat_policy = np.empty(total_nnz, dtype=np.float32)
        for i in range(n):
            start, end = int(offsets[i]), int(offsets[i + 1])
            flat_indices[start:end] = self._legal_indices[i]
            flat_policy[start:end] = self._policy_values[i]
        return {
            'max_size': self.max_size,
            'size': n,
            'board_states': bs,
            'pieces_remaining': pr,
            'value_targets': vt,
            'sparse_offsets': offsets,
            'sparse_indices': flat_indices,
            'sparse_policy': flat_policy,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer from checkpoint (supports both sparse and legacy dense)."""
        self.max_size = state.get('max_size', self.max_size)
        n = state.get('size', 0)
        if n == 0 and 'board_states' not in state:
            self._size = 0
            self._board_states = None
            self._legal_indices = []
            self._policy_values = []
            return
        if n == 0 and 'board_states' in state:
            n = len(state['board_states'])
        # Dense columns
        self._grow_and_copy(max(n, 1))
        self._board_states[:n] = state['board_states'][:n]
        self._pieces_remaining[:n] = state['pieces_remaining'][:n]
        self._value_targets[:n] = state['value_targets'][:n]
        # Sparse columns
        if 'sparse_offsets' in state:
            offsets = state['sparse_offsets']
            flat_indices = state['sparse_indices']
            flat_policy = state['sparse_policy']
            self._legal_indices = []
            self._policy_values = []
            for i in range(n):
                start, end = int(offsets[i]), int(offsets[i + 1])
                self._legal_indices.append(flat_indices[start:end].copy())
                self._policy_values.append(flat_policy[start:end].copy())
        else:
            # Legacy dense format — convert to sparse on load
            self._legal_indices = []
            self._policy_values = []
            lm = state['legal_masks']
            pt = state['policy_targets']
            for i in range(n):
                nonzero = np.nonzero(lm[i])[0].astype(np.int32)
                self._legal_indices.append(nonzero)
                self._policy_values.append(pt[i][nonzero].astype(np.float32).copy())
        self._size = n


# ---------------------------------------------------------------------------
# Self-play (parallel)
# ---------------------------------------------------------------------------

def _self_play_worker(rank: int, model_state_dict: dict, config: dict,
                      result_queue: mp.Queue) -> None:
    """Worker process that plays one self-play game.

    Workers use the configured device (typically GPU).  The CUDA OOM fix is
    in train_on_examples (mini-batch GPU loading), so GPU memory is free
    during self-play.
    """
    device = torch.device(config['device'])
    network = BlokusNetwork(
        num_blocks=config['num_blocks'],
        channels=config['channels'],
    ).to(device)
    network.load_state_dict(model_state_dict)
    network.eval()

    examples = self_play_game(
        network=network,
        game_mode=config['game_mode'],
        num_simulations=config['sims'],
        c_puct=config['c_puct'],
        temp_threshold=config['temp_threshold_move'],
        max_moves=config['max_moves'],
        device=device,
        top_k_actions=config.get('top_k_actions', 0),
    )
    result_queue.put(examples)


def run_self_play_parallel(network: BlokusNetwork, num_games: int,
                           num_workers: int, config: dict
                           ) -> List[TrainingExample]:
    """Run self-play games in parallel using multiple worker processes."""
    model_state = {k: v.cpu() for k, v in network.state_dict().items()}

    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()

    all_examples: List[TrainingExample] = []
    games_launched = 0
    games_collected = 0

    while games_collected < num_games:
        batch = min(num_workers, num_games - games_launched)
        processes = []
        for i in range(batch):
            p = ctx.Process(
                target=_self_play_worker,
                args=(games_launched + i, model_state, config, result_queue),
            )
            p.start()
            processes.append(p)
        games_launched += batch

        for _ in range(batch):
            examples = result_queue.get()
            all_examples.extend(examples)
            games_collected += 1
            print(f"  Game {games_collected}/{num_games}: "
                  f"{len(examples)} examples", flush=True)

        for p in processes:
            p.join()

    return all_examples


def run_self_play_sequential(network: BlokusNetwork, num_games: int,
                             cfg: Dict[str, Any], device: torch.device
                             ) -> List[TrainingExample]:
    """Run self-play games sequentially."""
    all_examples: List[TrainingExample] = []
    for game_idx in range(num_games):
        print(f"  Self-play game {game_idx+1}/{num_games}...",
              end='', flush=True)
        t0 = time.time()
        examples = self_play_game(
            network=network,
            game_mode=cfg['self_play']['game_mode'],
            num_simulations=cfg['mcts']['num_simulations'],
            c_puct=cfg['mcts']['c_puct'],
            temp_threshold=cfg['mcts']['temp_threshold_move'],
            max_moves=cfg['self_play']['max_moves'],
            device=device,
            top_k_actions=cfg['mcts'].get('top_k_actions', 0),
        )
        dt = time.time() - t0
        print(f" {len(examples)} examples in {dt:.1f}s")
        all_examples.extend(examples)
    return all_examples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_on_examples(network: BlokusNetwork,
                      replay_buffer: 'ReplayBuffer',
                      optimizer: optim.Optimizer,
                      batch_size: int = 64,
                      epochs: int = 5,
                      value_loss_weight: float = 1.0,
                      device: torch.device = torch.device('cpu')
                      ) -> dict:
    """Train the network on replay buffer contents.

    Loss = policy_loss + value_loss_weight * MSE(value_pred, value_target)

    Dense columns (board, pieces, value) are zero-copy CPU tensor views.
    Sparse columns (legal_mask, policy_target) are reconstructed per
    mini-batch from the buffer's compressed storage.
    """
    network.train()

    # Zero-copy CPU tensors for dense columns
    board_states, pieces_vecs, value_targets = replay_buffer.get_dense_tensors()

    n = len(replay_buffer)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
        perm = torch.randperm(n)

        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            idx = perm[i:end]

            # Dense columns — move to device
            bs = board_states[idx].to(device, non_blocking=True)
            pv = pieces_vecs[idx].to(device, non_blocking=True)
            vt = value_targets[idx].to(device, non_blocking=True)

            # Sparse → dense reconstruction for this mini-batch
            lm_np, pt_np = replay_buffer.reconstruct_sparse_batch(idx.numpy())
            lm = torch.from_numpy(lm_np).to(device, non_blocking=True)
            pt = torch.from_numpy(pt_np).to(device, non_blocking=True)

            optimizer.zero_grad()
            log_policy, value_pred = network(bs, pv, lm)
            policy_loss = -torch.sum(pt * log_policy) / bs.size(0)
            value_loss = nn.functional.mse_loss(value_pred, vt)
            loss = policy_loss + value_loss_weight * value_loss
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_batches += 1

    # Compute value accuracy: % of non-draw examples where sign(pred) == sign(target)
    network.eval()
    correct = 0
    non_draw = 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            bs = board_states[i:end].to(device, non_blocking=True)
            pv = pieces_vecs[i:end].to(device, non_blocking=True)
            vt = value_targets[i:end]

            # Reconstruct sparse for this batch
            batch_idx = np.arange(i, end)
            lm_np, _ = replay_buffer.reconstruct_sparse_batch(batch_idx)
            lm = torch.from_numpy(lm_np).to(device, non_blocking=True)

            _, value_pred = network(bs, pv, lm)
            vp = value_pred.cpu()

            mask = vt != 0
            non_draw += mask.sum().item()
            correct += ((vp[mask] > 0) == (vt[mask] > 0)).sum().item()

    value_accuracy = correct / max(non_draw, 1)

    return {
        'policy_loss': total_policy_loss / max(total_batches, 1),
        'value_loss': total_value_loss / max(total_batches, 1),
        'total_loss': (total_policy_loss + total_value_loss) / max(total_batches, 1),
        'value_accuracy': value_accuracy,
        'num_examples': n,
        'num_batches': total_batches,
    }


# ---------------------------------------------------------------------------
# Periodic Evaluation vs Random
# ---------------------------------------------------------------------------

def _eval_select_action_raw(network, state, device):
    """Select action using raw network policy (greedy, no MCTS)."""
    legal = state.get_legal_actions()
    if not legal:
        return -1
    board = torch.from_numpy(state.get_nn_state()).unsqueeze(0).to(device)
    pieces = torch.from_numpy(make_pieces_remaining_vector(state)).unsqueeze(0).to(device)
    mask = torch.from_numpy(state.get_legal_actions_mask()).unsqueeze(0).to(device)
    with torch.no_grad():
        log_policy, _ = network(board, pieces, mask)
    probs = torch.exp(log_policy).squeeze(0).cpu().numpy()
    best_a, best_p = legal[0], -1.0
    for a in legal:
        if probs[a] > best_p:
            best_p = probs[a]
            best_a = a
    return best_a


def _eval_play_game(network, device, mode, mcts_sims, game_mode):
    """Play one eval game: trained agent (agent 0) vs random (agent 1).

    Returns (winner, score_diff) where winner is 0/1/-1 and
    score_diff = trained_score - random_score.
    """
    state = GameState.new_game(game_mode)
    mcts = None
    if mode == 'mcts':
        mcts = MCTS(
            network=network,
            c_puct=1.5,
            num_simulations=mcts_sims,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            temperature=0.1,
            device=device,
        )

    while not state.is_terminal():
        agent_idx = state.get_current_agent()
        legal = state.get_legal_actions()
        if not legal:
            state = state.pass_turn()
            continue

        if agent_idx == 0:
            if mode == 'mcts':
                action, _, _ = mcts.select_action(state)
            else:
                action = _eval_select_action_raw(network, state, device)
        else:
            action = legal[np.random.randint(len(legal))]

        state = state.apply_action(action)

    rewards = state.get_rewards()
    scores = state.get_scores()

    if game_mode == 'dual':
        trained_score = scores[0] + scores[2]
        random_score = scores[1] + scores[3]
    else:
        trained_score = scores.get(0, 0)
        random_score = sum(scores.get(i, 0) for i in range(1, 4))

    if rewards[0] > rewards[1]:
        winner = 0
    elif rewards[1] > rewards[0]:
        winner = 1
    else:
        winner = -1

    return winner, trained_score - random_score


def evaluate_vs_random(network, device, num_games, mcts_sims, game_mode):
    """Run eval games for both raw policy and MCTS, return metrics dict."""
    network.eval()
    results = {}

    for mode in ('raw', 'mcts'):
        sims = mcts_sims if mode == 'mcts' else 0
        wins = 0
        total_score_diff = 0.0

        t0 = time.time()
        for _ in range(num_games):
            winner, score_diff = _eval_play_game(
                network, device, mode, sims, game_mode
            )
            if winner == 0:
                wins += 1
            total_score_diff += score_diff
        elapsed = time.time() - t0

        win_rate = wins / num_games
        avg_score_diff = total_score_diff / num_games
        results[f'eval/{mode}_win_rate'] = win_rate
        results[f'eval/{mode}_avg_score_diff'] = avg_score_diff

        print(f"  Eval ({mode:4s}): win_rate={win_rate:.1%}, "
              f"avg_score_diff={avg_score_diff:+.1f} "
              f"({num_games} games in {elapsed:.1f}s)")

    network.train()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Blokus AlphaZero Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--resume', nargs='?', const='latest', default=None,
                        help='Resume from checkpoint (default: latest, or specify path)')
    # CLI overrides (all optional — config file provides defaults)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--games-per-iter', type=int, default=None, dest='games_per_iter')
    parser.add_argument('--sims', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None, dest='batch_size')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--num-blocks', type=int, default=None, dest='num_blocks')
    parser.add_argument('--channels', type=int, default=None)
    parser.add_argument('--game-mode', type=str, default=None, dest='game_mode',
                        choices=['standard', 'dual'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None, dest='save_dir')
    parser.add_argument('--num-workers', type=int, default=None, dest='num_workers')
    parser.add_argument('--wandb', action='store_true', default=None)
    parser.add_argument('--wandb-project', type=str, default=None, dest='wandb_project')
    parser.add_argument('--wandb-run-name', type=str, default=None, dest='wandb_run_name')
    parser.add_argument('--c-puct', type=float, default=None, dest='c_puct')
    parser.add_argument('--max-moves', type=int, default=None, dest='max_moves')
    parser.add_argument('--buffer-size', type=int, default=None, dest='buffer_size')
    parser.add_argument('--value-loss-weight', type=float, default=None, dest='value_loss_weight')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    # Resolve device
    device_str = cfg['device']
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # Create network
    network = BlokusNetwork(
        num_blocks=cfg['network']['num_blocks'],
        channels=cfg['network']['channels'],
    ).to(device)

    param_count = sum(p.numel() for p in network.parameters())
    print(f"Network: {cfg['network']['num_blocks']} res blocks, "
          f"{cfg['network']['channels']} channels, {param_count:,} parameters")

    optimizer = optim.Adam(
        network.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
    )

    # Checkpoint manager
    ckpt_mgr = CheckpointManager(
        save_dir=cfg['checkpoint']['dir'],
        save_every=cfg['checkpoint']['save_every_n_iters'],
        keep_top_k=cfg['checkpoint']['keep_top_k'],
    )

    # Replay buffer
    buffer_size = cfg['training'].get('replay_buffer_size', 30000)
    replay_buffer = ReplayBuffer(max_size=buffer_size)
    print(f"Replay buffer: capacity {buffer_size}")

    start_iteration = 1
    cumulative_games = 0
    cumulative_examples = 0
    loss_history: List[Dict[str, float]] = []

    # Resume
    if args.resume:
        if args.resume == 'latest':
            ckpt = ckpt_mgr.load_latest(device)
        else:
            ckpt = ckpt_mgr.load(args.resume, device)

        if ckpt:
            network.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_iteration = ckpt.get('iteration', 0) + 1
            cumulative_games = ckpt.get('cumulative_games', 0)
            cumulative_examples = ckpt.get('cumulative_examples', 0)
            loss_history = ckpt.get('loss_history', [])
            # Restore replay buffer — check npz first, then legacy .pt, then inline
            buf_npz = os.path.join(cfg['checkpoint']['dir'], 'replay_buffer.npz')
            buf_pt = os.path.join(cfg['checkpoint']['dir'], 'replay_buffer.pt')
            if os.path.exists(buf_npz):
                buf_state = dict(np.load(buf_npz))
                buf_state['max_size'] = int(buf_state['max_size'])
                buf_state['size'] = int(buf_state['size'])
                replay_buffer.load_state_dict(buf_state)
                del buf_state
                print(f"Resumed from iteration {start_iteration - 1}: "
                      f"{cumulative_games} games, {cumulative_examples} examples, "
                      f"{len(replay_buffer)} examples in buffer")
            elif os.path.exists(buf_pt):
                buf_state = torch.load(buf_pt, map_location='cpu',
                                       weights_only=False)
                replay_buffer.load_state_dict(buf_state)
                del buf_state
                print(f"Resumed from iteration {start_iteration - 1}: "
                      f"{cumulative_games} games, {cumulative_examples} examples, "
                      f"{len(replay_buffer)} examples in buffer (from legacy .pt)")
            elif 'replay_buffer' in ckpt:
                # Backwards compat: buffer was embedded in old checkpoints
                replay_buffer.load_state_dict(ckpt['replay_buffer'])
                print(f"Resumed from iteration {start_iteration - 1}: "
                      f"{cumulative_games} games, {cumulative_examples} examples, "
                      f"{len(replay_buffer)} examples in buffer (from inline ckpt)")
            else:
                print(f"Resumed from iteration {start_iteration - 1}: "
                      f"{cumulative_games} games, {cumulative_examples} examples "
                      f"(no replay buffer found)")
            del ckpt  # Free checkpoint memory
        else:
            print("No checkpoint found — starting from scratch")

    # W&B setup
    wandb_run = None
    if cfg['wandb']['enabled']:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg['wandb']['project'],
                name=cfg['wandb']['run_name'],
                config=cfg,
                resume='allow',
            )
            print(f"W&B run: {wandb_run.url}")
        except ImportError:
            print("WARNING: wandb not installed, logging disabled")
            cfg['wandb']['enabled'] = False

    # Worker config for parallel self-play
    worker_config = {
        'device': str(device),
        'game_mode': cfg['self_play']['game_mode'],
        'sims': cfg['mcts']['num_simulations'],
        'c_puct': cfg['mcts']['c_puct'],
        'temp_threshold_move': cfg['mcts']['temp_threshold_move'],
        'max_moves': cfg['self_play']['max_moves'],
        'num_blocks': cfg['network']['num_blocks'],
        'channels': cfg['network']['channels'],
        'top_k_actions': cfg['mcts'].get('top_k_actions', 0),
    }

    num_workers = cfg['self_play']['num_workers']
    use_parallel = num_workers > 1
    if use_parallel:
        print(f"Self-play: {num_workers} parallel workers")
        if device.type == 'cuda':
            mp.set_start_method('spawn', force=True)
    else:
        print("Self-play: sequential (1 worker)")

    total_iterations = cfg['training']['iterations']
    end_iteration = start_iteration + total_iterations - 1

    # Print config summary
    print(f"\nConfig: {total_iterations} iterations, "
          f"{cfg['self_play']['games_per_iteration']} games/iter, "
          f"{cfg['mcts']['num_simulations']} sims, "
          f"lr={cfg['training']['learning_rate']}")

    # Training loop
    for iteration in range(start_iteration, end_iteration + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{end_iteration}")
        print(f"{'='*60}")

        # ------ Self-play ------
        t_sp = time.time()
        games_this_iter = cfg['self_play']['games_per_iteration']

        if use_parallel:
            all_examples = run_self_play_parallel(
                network=network,
                num_games=games_this_iter,
                num_workers=num_workers,
                config=worker_config,
            )
        else:
            all_examples = run_self_play_sequential(
                network=network,
                num_games=games_this_iter,
                cfg=cfg,
                device=device,
            )

        sp_time = time.time() - t_sp
        iter_examples = len(all_examples)
        cumulative_games += games_this_iter
        cumulative_examples += iter_examples
        games_per_hour = games_this_iter / sp_time * 3600 if sp_time > 0 else 0
        examples_per_hour = iter_examples / sp_time * 3600 if sp_time > 0 else 0
        print(f"  Self-play: {iter_examples} examples in {sp_time:.1f}s "
              f"({games_per_hour:.0f} games/hr)")

        # ------ Add to replay buffer ------
        dropped = replay_buffer.add(all_examples)
        buf_len = len(replay_buffer)
        print(f"  Buffer: {buf_len}/{buffer_size} examples"
              f"{f' (dropped {dropped} oldest)' if dropped else ''}")

        # ------ Value target diagnostics ------
        buf_vt = replay_buffer.value_targets_array
        vt_mean = float(buf_vt.mean())
        vt_std = float(buf_vt.std())
        vt_min = float(buf_vt.min())
        vt_max = float(buf_vt.max())
        # Also check current iteration's targets
        iter_vt = np.array([ex.value_target for ex in all_examples], dtype=np.float32)
        iter_vt_mean = float(iter_vt.mean())
        iter_vt_std = float(iter_vt.std())
        print(f"  Value targets (buffer): mean={vt_mean:+.4f}, std={vt_std:.4f}, "
              f"range=[{vt_min:+.3f}, {vt_max:+.3f}]")
        print(f"  Value targets (iter):   mean={iter_vt_mean:+.4f}, std={iter_vt_std:.4f}")

        # ------ Training (on full buffer) ------
        t_tr = time.time()
        stats = train_on_examples(
            network=network,
            replay_buffer=replay_buffer,
            optimizer=optimizer,
            batch_size=cfg['training']['batch_size'],
            epochs=cfg['training']['epochs_per_iter'],
            value_loss_weight=cfg['training'].get('value_loss_weight', 1.0),
            device=device,
        )
        train_time = time.time() - t_tr

        # Free cached GPU memory before self-play workers spawn
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        total_iter_time = sp_time + train_time

        loss_entry = {
            'iteration': iteration,
            'policy_loss': stats['policy_loss'],
            'value_loss': stats['value_loss'],
            'total_loss': stats['total_loss'],
        }
        loss_history.append(loss_entry)

        print(f"  Training: policy_loss={stats['policy_loss']:.4f}, "
              f"value_loss={stats['value_loss']:.4f}, "
              f"total_loss={stats['total_loss']:.4f}, "
              f"value_acc={stats['value_accuracy']:.1%} "
              f"({train_time:.1f}s)")

        # ------ Checkpoint ------
        meta = {
            'cumulative_games': cumulative_games,
            'cumulative_examples': cumulative_examples,
            'loss_history': loss_history,
            'stats': stats,
            'replay_buffer': replay_buffer.state_dict(),
        }
        ckpt_path = ckpt_mgr.save(iteration, network, optimizer, meta, cfg)
        if ckpt_path:
            print(f"  Checkpoint: {ckpt_path}")
        ckpt_mgr.update_best(iteration, network, optimizer, meta, cfg,
                             stats['policy_loss'])

        # ------ Periodic Eval vs Random ------
        eval_metrics = {}
        eval_cfg = cfg.get('eval', {})
        eval_interval = eval_cfg.get('interval', 0)
        if eval_interval > 0 and iteration % eval_interval == 0:
            eval_metrics = evaluate_vs_random(
                network=network,
                device=device,
                num_games=eval_cfg.get('games', 50),
                mcts_sims=eval_cfg.get('mcts_sims', 25),
                game_mode=cfg['self_play']['game_mode'],
            )
            # Free GPU cache after eval games
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # ------ W&B ------
        if cfg['wandb']['enabled'] and wandb_run:
            import wandb
            log_data = {
                'iteration': iteration,
                'policy_loss': stats['policy_loss'],
                'value_loss': stats['value_loss'],
                'total_loss': stats['total_loss'],
                'iter_games': games_this_iter,
                'iter_examples': iter_examples,
                'self_play_time_s': sp_time,
                'train_time_s': train_time,
                'total_iter_time_s': total_iter_time,
                'games_per_hour': games_per_hour,
                'examples_per_hour': examples_per_hour,
                'games_played': cumulative_games,
                'training_examples': cumulative_examples,
                'buffer_size': buf_len,
                'buffer_capacity': buffer_size,
                'value_target_mean': vt_mean,
                'value_target_std': vt_std,
                'value_target_min': vt_min,
                'value_target_max': vt_max,
                'iter_value_target_mean': iter_vt_mean,
                'iter_value_target_std': iter_vt_std,
                'value_accuracy': stats['value_accuracy'],
            }
            log_data.update(eval_metrics)
            wandb.log(log_data, step=iteration)

    print(f"\nTraining complete: {cumulative_games} games, "
          f"{cumulative_examples} examples over {total_iterations} iterations")

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
