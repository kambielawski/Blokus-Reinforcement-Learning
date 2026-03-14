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

from blokus.nn.network import BlokusNetwork
from blokus.agents.alpha_zero import self_play_game, TrainingExample


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
        """Save checkpoint if due. Always saves 'latest'. Returns path or None."""
        if iteration % self.save_every != 0:
            return None

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
            payload = {
                'iteration': iteration,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                **meta,
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
    """Rolling FIFO buffer that stores training examples across iterations.

    New examples are appended each iteration. When the buffer exceeds
    max_size, the oldest examples are dropped.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: List[TrainingExample] = []

    def add(self, examples: List[TrainingExample]) -> int:
        """Add examples to the buffer, dropping oldest if over capacity.

        Returns the number of examples dropped.
        """
        self.buffer.extend(examples)
        dropped = 0
        if len(self.buffer) > self.max_size:
            dropped = len(self.buffer) - self.max_size
            self.buffer = self.buffer[dropped:]
        return dropped

    def sample(self, n: int) -> List[TrainingExample]:
        """Sample n examples uniformly at random (with replacement if n > len)."""
        if not self.buffer:
            return []
        indices = np.random.randint(0, len(self.buffer), size=min(n, len(self.buffer)))
        return [self.buffer[i] for i in indices]

    def get_all(self) -> List[TrainingExample]:
        """Return all examples in the buffer."""
        return self.buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def state_dict(self) -> dict:
        """Serialize buffer contents for checkpointing."""
        if not self.buffer:
            return {'examples': [], 'max_size': self.max_size}
        return {
            'max_size': self.max_size,
            'board_states': np.stack([ex.board_state for ex in self.buffer]),
            'pieces_remaining': np.stack([ex.pieces_remaining for ex in self.buffer]),
            'legal_masks': np.stack([ex.legal_mask for ex in self.buffer]),
            'policy_targets': np.stack([ex.policy_target for ex in self.buffer]),
            'value_targets': np.array([ex.value_target for ex in self.buffer],
                                      dtype=np.float32),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer from checkpoint."""
        if 'board_states' not in state:
            self.buffer = []
            return
        self.max_size = state['max_size']
        n = len(state['value_targets'])
        self.buffer = []
        for i in range(n):
            self.buffer.append(TrainingExample(
                board_state=state['board_states'][i],
                pieces_remaining=state['pieces_remaining'][i],
                legal_mask=state['legal_masks'][i],
                policy_target=state['policy_targets'][i],
                value_target=float(state['value_targets'][i]),
            ))


# ---------------------------------------------------------------------------
# Self-play (parallel)
# ---------------------------------------------------------------------------

def _self_play_worker(rank: int, model_state_dict: dict, config: dict,
                      result_queue: mp.Queue) -> None:
    """Worker process that plays one self-play game."""
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
        )
        dt = time.time() - t0
        print(f" {len(examples)} examples in {dt:.1f}s")
        all_examples.extend(examples)
    return all_examples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_on_examples(network: BlokusNetwork,
                      examples: List[TrainingExample],
                      optimizer: optim.Optimizer,
                      batch_size: int = 64,
                      epochs: int = 5,
                      device: torch.device = torch.device('cpu')
                      ) -> dict:
    """Train the network on self-play examples.

    Loss = MSE(value_pred, value_target) + CE(policy_pred, policy_target)
    """
    network.train()

    board_states = torch.from_numpy(
        np.stack([ex.board_state for ex in examples])
    ).to(device)
    pieces_vecs = torch.from_numpy(
        np.stack([ex.pieces_remaining for ex in examples])
    ).to(device)
    legal_masks = torch.from_numpy(
        np.stack([ex.legal_mask for ex in examples])
    ).to(device)
    policy_targets = torch.from_numpy(
        np.stack([ex.policy_target for ex in examples])
    ).to(device)
    value_targets = torch.from_numpy(
        np.array([ex.value_target for ex in examples], dtype=np.float32)
    ).to(device)

    n = len(examples)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
        perm = torch.randperm(n)
        board_states = board_states[perm]
        pieces_vecs = pieces_vecs[perm]
        legal_masks = legal_masks[perm]
        policy_targets = policy_targets[perm]
        value_targets = value_targets[perm]

        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            bs = board_states[i:end]
            pv = pieces_vecs[i:end]
            lm = legal_masks[i:end]
            pt = policy_targets[i:end]
            vt = value_targets[i:end]

            optimizer.zero_grad()
            log_policy, value_pred = network(bs, pv, lm)
            policy_loss = -torch.sum(pt * log_policy) / bs.size(0)
            value_loss = nn.functional.mse_loss(value_pred, vt)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_batches += 1

    return {
        'policy_loss': total_policy_loss / max(total_batches, 1),
        'value_loss': total_value_loss / max(total_batches, 1),
        'total_loss': (total_policy_loss + total_value_loss) / max(total_batches, 1),
        'num_examples': n,
        'num_batches': total_batches,
    }


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
            # Restore replay buffer
            if 'replay_buffer' in ckpt:
                replay_buffer.load_state_dict(ckpt['replay_buffer'])
                print(f"Resumed from iteration {start_iteration - 1}: "
                      f"{cumulative_games} games, {cumulative_examples} examples, "
                      f"{len(replay_buffer)} examples in buffer")
            else:
                print(f"Resumed from iteration {start_iteration - 1}: "
                      f"{cumulative_games} games, {cumulative_examples} examples "
                      f"(no replay buffer in checkpoint)")
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
        value_targets = np.array([ex.value_target for ex in replay_buffer.get_all()],
                                 dtype=np.float32)
        vt_mean = float(value_targets.mean())
        vt_std = float(value_targets.std())
        vt_min = float(value_targets.min())
        vt_max = float(value_targets.max())
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
            examples=replay_buffer.get_all(),
            optimizer=optimizer,
            batch_size=cfg['training']['batch_size'],
            epochs=cfg['training']['epochs_per_iter'],
            device=device,
        )
        train_time = time.time() - t_tr
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
              f"total_loss={stats['total_loss']:.4f} "
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

        # ------ W&B ------
        if cfg['wandb']['enabled'] and wandb_run:
            import wandb
            wandb.log({
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
            }, step=iteration)

    print(f"\nTraining complete: {cumulative_games} games, "
          f"{cumulative_examples} examples over {total_iterations} iterations")

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
