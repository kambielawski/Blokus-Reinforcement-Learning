#!/usr/bin/env python3
"""AlphaZero training loop for Blokus.

Usage:
    python scripts/train.py [--iterations N] [--games-per-iter N] [--sims N]
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
import numpy as np
from typing import List

# Add repo root to path for script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus.nn.network import BlokusNetwork
from blokus.agents.alpha_zero import self_play_game, TrainingExample


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

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{args.iterations}")
        print(f"{'='*60}")

        # Self-play phase
        all_examples: List[TrainingExample] = []
        t0 = time.time()

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

        sp_time = time.time() - t0
        print(f"  Self-play: {len(all_examples)} total examples in {sp_time:.1f}s")

        # Training phase
        t0 = time.time()
        stats = train_on_examples(
            network=network,
            examples=all_examples,
            optimizer=optimizer,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device,
        )
        train_time = time.time() - t0

        print(f"  Training: policy_loss={stats['policy_loss']:.4f}, "
              f"value_loss={stats['value_loss']:.4f}, "
              f"total_loss={stats['total_loss']:.4f} "
              f"({train_time:.1f}s)")

        # Save checkpoint
        if iteration % 5 == 0 or iteration == args.iterations:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_{iteration:04d}.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stats': stats,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.save_dir, 'model_latest.pt')
    torch.save({
        'iteration': args.iterations,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == '__main__':
    main()
