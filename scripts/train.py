#!/usr/bin/env python3
"""AlphaZero training loop for Blokus.

Supports multi-process self-play for parallel game generation and optional
Weights & Biases logging for experiment tracking.

Usage:
    python scripts/train.py [--iterations N] [--games-per-iter N] [--sims N]
                            [--num-workers N] [--wandb] [--wandb-project NAME]
                            [--lr LR] [--batch-size N] [--epochs N]
                            [--device DEVICE] [--save-dir DIR]
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from typing import List

# Add repo root to path for script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus.nn.network import BlokusNetwork
from blokus.agents.alpha_zero import self_play_game, TrainingExample


# ---------------------------------------------------------------------------
# Self-play worker for multiprocessing
# ---------------------------------------------------------------------------

def _self_play_worker(rank: int, model_state_dict: dict, config: dict,
                      result_queue: mp.Queue) -> None:
    """Worker process that plays one self-play game and puts examples in queue.

    Each worker creates its own model copy on the target device. For CUDA,
    PyTorch handles concurrent access across processes automatically.
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
        device=device,
    )
    result_queue.put(examples)


def run_self_play_parallel(network: BlokusNetwork, num_games: int,
                           num_workers: int, config: dict
                           ) -> List[TrainingExample]:
    """Run self-play games in parallel using multiple worker processes.

    Args:
        network: Current network (weights copied to each worker).
        num_games: Total number of games to play.
        num_workers: Number of parallel worker processes.
        config: Dict with 'device', 'game_mode', 'sims', 'num_blocks', 'channels'.

    Returns:
        Combined list of TrainingExample from all games.
    """
    # Share model weights (CPU tensors for cross-process transfer)
    model_state = {k: v.cpu() for k, v in network.state_dict().items()}

    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()

    all_examples: List[TrainingExample] = []
    games_launched = 0
    games_collected = 0

    # Launch games in waves of num_workers
    while games_collected < num_games:
        # Launch up to num_workers processes
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

        # Collect results from this wave
        for _ in range(batch):
            examples = result_queue.get()
            all_examples.extend(examples)
            games_collected += 1
            print(f"  Game {games_collected}/{num_games}: "
                  f"{len(examples)} examples", flush=True)

        # Ensure all processes have exited
        for p in processes:
            p.join()

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
    """Train the network on a batch of self-play examples.

    Loss = MSE(value_pred, value_target) + CE(policy_pred, policy_target)

    Returns dict with loss statistics.
    """
    network.train()

    # Stack all examples into tensors
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
        # Shuffle
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

            # Policy loss: cross-entropy with MCTS visit distribution
            # -sum(pi * log(p)) where pi is the target distribution
            policy_loss = -torch.sum(pt * log_policy) / bs.size(0)

            # Value loss: MSE
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
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of training iterations')
    parser.add_argument('--games-per-iter', type=int, default=4,
                        help='Self-play games per iteration')
    parser.add_argument('--sims', type=int, default=50,
                        help='MCTS simulations per move')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs per iteration')
    parser.add_argument('--num-blocks', type=int, default=5,
                        help='Number of residual blocks')
    parser.add_argument('--channels', type=int, default=128,
                        help='Number of backbone channels')
    parser.add_argument('--game-mode', type=str, default='dual',
                        choices=['standard', 'dual'],
                        help='Game mode for self-play')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu/mps/cuda)')
    parser.add_argument('--save-dir', type=str, default='data/checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel self-play workers (1=sequential)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='blokus-alphazero',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name (auto-generated if not set)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create network
    network = BlokusNetwork(
        num_blocks=args.num_blocks,
        channels=args.channels,
    ).to(device)

    param_count = sum(p.numel() for p in network.parameters())
    print(f"Network: {args.num_blocks} res blocks, {args.channels} channels, "
          f"{param_count:,} parameters")

    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-4)

    start_iteration = 1
    cumulative_games = 0
    cumulative_examples = 0

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        network.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iteration = ckpt.get('iteration', 0) + 1
        cumulative_games = ckpt.get('cumulative_games', 0)
        cumulative_examples = ckpt.get('cumulative_examples', 0)
        print(f"  Resumed at iteration {start_iteration}, "
              f"{cumulative_games} games, {cumulative_examples} examples")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # --- W&B setup ---
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    'iterations': args.iterations,
                    'games_per_iter': args.games_per_iter,
                    'sims': args.sims,
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'num_blocks': args.num_blocks,
                    'channels': args.channels,
                    'game_mode': args.game_mode,
                    'device': str(device),
                    'num_workers': args.num_workers,
                    'param_count': param_count,
                },
                resume='allow',
            )
            print(f"W&B run: {wandb_run.url}")
        except ImportError:
            print("WARNING: wandb not installed, logging disabled")
            args.wandb = False

    # Config dict for workers
    worker_config = {
        'device': str(device),
        'game_mode': args.game_mode,
        'sims': args.sims,
        'num_blocks': args.num_blocks,
        'channels': args.channels,
    }

    use_parallel = args.num_workers > 1
    if use_parallel:
        print(f"Self-play: {args.num_workers} parallel workers")
        # Required for CUDA multiprocessing
        if device.type == 'cuda':
            mp.set_start_method('spawn', force=True)
    else:
        print("Self-play: sequential (1 worker)")

    # Training loop
    end_iteration = start_iteration + args.iterations - 1
    for iteration in range(start_iteration, end_iteration + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{end_iteration}")
        print(f"{'='*60}")

        # ------ Self-play phase ------
        t_sp_start = time.time()

        if use_parallel:
            all_examples = run_self_play_parallel(
                network=network,
                num_games=args.games_per_iter,
                num_workers=args.num_workers,
                config=worker_config,
            )
        else:
            all_examples: List[TrainingExample] = []
            for game_idx in range(args.games_per_iter):
                print(f"  Self-play game {game_idx+1}/{args.games_per_iter}...",
                      end='', flush=True)
                game_t0 = time.time()
                examples = self_play_game(
                    network=network,
                    game_mode=args.game_mode,
                    num_simulations=args.sims,
                    device=device,
                )
                game_dt = time.time() - game_t0
                print(f" {len(examples)} examples in {game_dt:.1f}s")
                all_examples.extend(examples)

        sp_time = time.time() - t_sp_start
        iter_games = args.games_per_iter
        iter_examples = len(all_examples)
        cumulative_games += iter_games
        cumulative_examples += iter_examples
        print(f"  Self-play: {iter_examples} examples in {sp_time:.1f}s "
              f"({iter_games/sp_time*3600:.0f} games/hr)")

        # ------ Training phase ------
        t_tr_start = time.time()
        stats = train_on_examples(
            network=network,
            examples=all_examples,
            optimizer=optimizer,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device,
        )
        train_time = time.time() - t_tr_start
        total_iter_time = sp_time + train_time

        print(f"  Training: policy_loss={stats['policy_loss']:.4f}, "
              f"value_loss={stats['value_loss']:.4f}, "
              f"total_loss={stats['total_loss']:.4f} "
              f"({train_time:.1f}s)")

        # ------ W&B logging ------
        if args.wandb and wandb_run:
            import wandb
            games_per_hour = iter_games / sp_time * 3600 if sp_time > 0 else 0
            examples_per_hour = iter_examples / sp_time * 3600 if sp_time > 0 else 0
            wandb.log({
                'iteration': iteration,
                # Losses
                'policy_loss': stats['policy_loss'],
                'value_loss': stats['value_loss'],
                'total_loss': stats['total_loss'],
                # Per-iteration stats
                'iter_games': iter_games,
                'iter_examples': iter_examples,
                'self_play_time_s': sp_time,
                'train_time_s': train_time,
                'total_iter_time_s': total_iter_time,
                # Throughput
                'games_per_hour': games_per_hour,
                'examples_per_hour': examples_per_hour,
                # Cumulative
                'games_played': cumulative_games,
                'training_examples': cumulative_examples,
            }, step=iteration)

        # ------ Save checkpoint ------
        if iteration % 5 == 0 or iteration == end_iteration:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_{iteration:04d}.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stats': stats,
                'cumulative_games': cumulative_games,
                'cumulative_examples': cumulative_examples,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.save_dir, 'model_latest.pt')
    torch.save({
        'iteration': end_iteration,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'cumulative_games': cumulative_games,
        'cumulative_examples': cumulative_examples,
    }, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Total: {cumulative_games} games, {cumulative_examples} examples")

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
