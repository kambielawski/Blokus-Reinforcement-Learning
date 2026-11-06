#!/usr/bin/env python3
"""Train a network from games played by the mobility heuristic.

Gives self-play a starting policy that already plays sensibly, and a value head
trained on tanh(margin / margin_scale) so Q is large enough to steer search.
Writes a checkpoint in the normal format that the UI, ladder, and train.py can load.

Usage:
    python scripts/pretrain_imitation.py --games 4000 --epochs 4 \\
        --out data/pretrain/imitation.pt --device mps
"""

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus.agents.bots import MobilityBot
from blokus.engine.game_state import (
    ACTION_SPACE_SIZE, GameState, RULES_VERSION, decode_action, load_pieces,
)
from blokus.engine.symmetry import rotation_action_tables
from blokus.nn.network import BlokusNetwork, make_pieces_remaining_vector


def play_one(args: Tuple[int, float, float]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]:
    """Play a noisy mobility-vs-mobility game; return one example per scored move."""
    seed, noise, margin_scale = args
    rng = np.random.default_rng(seed)
    bot = MobilityBot(seed=seed)
    pieces = load_pieces()
    state = GameState.new_game('dual')
    history = []
    plies = 0
    while not state.is_terminal():
        legal = state.get_legal_actions()
        if not legal:
            state = state.pass_turn()
            continue
        if plies < 4:  # Random pentomino openings, as in the ladder
            big = [a for a in legal if pieces[decode_action(a)[0]].num_cells == 5]
            action = big[rng.integers(len(big))]
        elif rng.random() < noise:
            action = legal[rng.integers(len(legal))]
        else:
            action = bot.select_action(state)
            history.append((state.get_nn_state().astype(np.uint8),
                            make_pieces_remaining_vector(state).astype(np.uint8),
                            np.array(legal, dtype=np.int32), action,
                            state.get_current_agent()))
        state = state.apply_action(action)
        plies += 1
    scores = state.get_scores()
    margin0 = math.tanh((scores[0] + scores[2] - scores[1] - scores[3]) / margin_scale)
    return [(b, p, l, a, margin0 if agent == 0 else -margin0) for b, p, l, a, agent in history]


def main() -> None:
    parser = argparse.ArgumentParser(description='Imitation pretraining from the mobility bot')
    parser.add_argument('--games', type=int, default=4000)
    parser.add_argument('--noise', type=float, default=0.1, help='Fraction of random moves')
    parser.add_argument('--margin-scale', type=float, default=20.0)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-blocks', type=int, default=5)
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    started = time.time()
    tasks = [(seed, args.noise, args.margin_scale) for seed in range(args.games)]
    with mp.get_context('spawn').Pool(args.workers) as pool:
        games = pool.map(play_one, tasks, chunksize=16)
    examples = [e for game in games for e in game]
    print(f'{len(examples)} examples from {args.games} games in {time.time() - started:.0f}s',
          flush=True)

    # Hold out whole games so validation measures generalization
    holdout = max(1, args.games // 20)
    val = [e for game in games[:holdout] for e in game]
    train = [e for game in games[holdout:] for e in game]
    del games, examples

    tables = rotation_action_tables()
    device = torch.device(args.device)
    network = BlokusNetwork(num_blocks=args.num_blocks, channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * math.ceil(len(train) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)

    def batch_tensors(batch: List[Any], turns: np.ndarray) -> Tuple[torch.Tensor, ...]:
        n = len(batch)
        boards = np.empty((n, 5, 20, 20), dtype=np.float32)
        pieces = np.empty((n, 84), dtype=np.float32)
        masks = np.zeros((n, ACTION_SPACE_SIZE), dtype=np.float32)
        targets = np.empty(n, dtype=np.int64)
        values = np.empty(n, dtype=np.float32)
        for i, (b, p, legal, action, value) in enumerate(batch):
            table = tables[turns[i]]
            boards[i] = np.rot90(b, k=-int(turns[i]), axes=(1, 2))
            masks[i, table[legal]] = 1.0
            targets[i] = table[action]
            pieces[i] = p
            values[i] = value
        return tuple(torch.from_numpy(x).to(device)
                     for x in (boards, pieces, masks, targets, values))

    def validate() -> Dict[str, float]:
        network.eval()
        policy_loss = value_loss = correct = count = 0.0
        predictions, outcomes = [], []
        with torch.no_grad():
            for i in range(0, len(val), args.batch_size):
                chunk = val[i:i + args.batch_size]
                boards, pieces, masks, targets, values = batch_tensors(chunk, np.zeros(len(chunk), int))
                log_policy, value = network(boards, pieces, masks)
                policy_loss += F.nll_loss(log_policy, targets, reduction='sum').item()
                value_loss += F.mse_loss(value, values, reduction='sum').item()
                correct += (log_policy.argmax(1) == targets).sum().item()
                count += len(chunk)
                predictions.append(value.cpu().numpy())
                outcomes.append(values.cpu().numpy())
        pred, true = np.concatenate(predictions), np.concatenate(outcomes)
        return {'policy_loss': policy_loss / count, 'value_mse': value_loss / count,
                'top1': correct / count, 'value_corr': float(np.corrcoef(pred, true)[0, 1]),
                'target_var': float(true.var())}

    rng = np.random.default_rng(0)
    step = 0
    for epoch in range(args.epochs):
        network.train()
        order = rng.permutation(len(train))
        for start in range(0, len(train), args.batch_size):
            chunk = [train[i] for i in order[start:start + args.batch_size]]
            boards, pieces, masks, targets, values = batch_tensors(chunk, rng.integers(0, 4, size=len(chunk)))
            log_policy, value = network(boards, pieces, masks)
            loss = F.nll_loss(log_policy, targets) + F.mse_loss(value, values)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 200 == 0:
                print(f'  step {step}/{total_steps} loss {loss.item():.3f} '
                      f'({time.time() - started:.0f}s)', flush=True)
        print(f'epoch {epoch + 1}: {validate()} ({time.time() - started:.0f}s)', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        'iteration': 0,
        'rules_version': RULES_VERSION,
        'model_state_dict': {k: v.cpu() for k, v in network.state_dict().items()},
        'config': {
            'network': {'num_blocks': args.num_blocks, 'channels': args.channels},
            'self_play': {'game_mode': 'dual'},
            'training': {'value_target': 'margin', 'margin_scale': args.margin_scale},
            'pretrain': vars(args),
        },
    }, args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
