#!/usr/bin/env python3
"""Offline fine-tune of a checkpoint on its own replay buffer, one arm per learning rate.

Tests whether the policy is optimization-limited: if a lower learning rate fits the
self-play targets better than the same number of steps at the training LR, the
live schedule is too hot. Every arm starts from the same weights and optimizer
state and sees the same sample order.

Held-out positions come from a later buffer: its newest --holdout-count examples
were played by the parent net after the parent buffer was saved, so no arm has
trained on them.

Usage:
  python scripts/finetune_buffer.py --checkpoint parent.pt --buffer-dir parent/ \
      --holdout-dir later/ --holdout-count 10300 --arm low=3e-5 --arm control=3.1e-4 \
      --epochs 2 --out out_dir
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blokus.engine.game_state import ACTION_SPACE_SIZE, RULES_VERSION
from blokus.nn.checkpoint import load_network
from train import ReplayBuffer, train_on_examples, _seed_everything


def load_buffer(run_dir: str) -> ReplayBuffer:
    state = dict(np.load(os.path.join(run_dir, 'replay_buffer.npz')))
    state['max_size'] = int(state['max_size'])
    state['size'] = int(state['size'])
    buf = ReplayBuffer(max_size=state['max_size'])
    buf.load_state_dict(state)
    return buf


def holdout_metrics(network, device, data, count: int, batch: int = 256) -> dict:
    """Policy cross-entropy, KL(target||prior), argmax agreement and value error."""
    off, idxs, pol = data['sparse_offsets'], data['sparse_indices'], data['sparse_policy']
    boards, pieces, values = data['board_states'], data['pieces_remaining'], data['value_targets']
    n = len(values)
    rows = np.arange(n - count, n)
    ce, kl, agree, sq, sign = [], [], [], [], []
    network.eval()
    for b in range(0, len(rows), batch):
        ch = rows[b:b + batch]
        mask = np.zeros((len(ch), ACTION_SPACE_SIZE), np.float32)
        for j, i in enumerate(ch):
            mask[j, idxs[off[i]:off[i + 1]]] = 1
        with torch.no_grad():
            lp, v = network(torch.from_numpy(boards[ch]).to(device),
                            torch.from_numpy(pieces[ch]).to(device),
                            torch.from_numpy(mask).to(device))
        lp = lp.cpu().numpy().astype(np.float64)
        v = v.cpu().numpy().ravel()
        for j, i in enumerate(ch):
            a = idxs[off[i]:off[i + 1]]
            t = pol[off[i]:off[i + 1]].astype(np.float64)
            t /= t.sum()
            q = lp[j, a]
            nz = t > 0
            ce.append(-(t[nz] * q[nz]).sum())
            kl.append((t[nz] * (np.log(t[nz]) - q[nz])).sum())
            agree.append(t.argmax() == q.argmax())
        sq.extend((v - values[ch]) ** 2)
        sign.extend(np.sign(v) == np.sign(values[ch]))
    return {'policy_ce': float(np.mean(ce)), 'kl': float(np.mean(kl)),
            'argmax_agree': float(np.mean(agree)), 'value_mse': float(np.mean(sq)),
            'value_sign_acc': float(np.mean(sign)), 'positions': int(len(rows))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', required=True, help='Parent checkpoint')
    parser.add_argument('--buffer-dir', required=True, help='Holds the parent replay_buffer.npz')
    parser.add_argument('--holdout-dir', required=True, help='Holds a later replay_buffer.npz')
    parser.add_argument('--holdout-count', type=int, required=True,
                        help='Newest examples of the later buffer played after the parent')
    parser.add_argument('--arm', action='append', required=True, help='name=learning_rate')
    parser.add_argument('--epochs', type=float, default=2.0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = parent['config']
    buf = load_buffer(args.buffer_dir)
    holdout = dict(np.load(os.path.join(args.holdout_dir, 'replay_buffer.npz')))
    print(f"Parent iteration {parent['iteration']}, buffer {len(buf)} examples, "
          f"holdout {args.holdout_count}", flush=True)

    network, _ = load_network(args.checkpoint, device)
    results = {'parent': {'iteration': parent['iteration'],
                          **holdout_metrics(network, device, holdout, args.holdout_count)}}
    print('parent', results['parent'], flush=True)

    for arm in args.arm:
        name, lr = arm.split('=')
        lr = float(lr)
        _seed_everything(args.seed)   # Same sample order and rotations in every arm
        network, _ = load_network(args.checkpoint, device)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr,
                                     weight_decay=cfg['training']['weight_decay'])
        optimizer.load_state_dict(parent['optimizer_state_dict'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        t0 = time.time()
        stats = train_on_examples(network, buf, optimizer, batch_size=args.batch_size,
                                  epochs=args.epochs,
                                  value_loss_weight=cfg['training'].get('value_loss_weight', 1.0),
                                  device=device,
                                  rotation_augment=cfg['training'].get('rotation_augment', False))
        metrics = holdout_metrics(network, device, holdout, args.holdout_count)
        results[name] = {'lr': lr, 'steps': stats['num_batches'],
                         'train_policy_loss': stats['policy_loss'],
                         'train_value_loss': stats['value_loss'],
                         'seconds': time.time() - t0, **metrics}
        print(name, results[name], flush=True)
        torch.save({'iteration': parent['iteration'], 'model_state_dict': network.state_dict(),
                    'config': cfg, 'rules_version': RULES_VERSION,
                    'finetune': {'parent': args.checkpoint, 'lr': lr, 'epochs': args.epochs}},
                   os.path.join(args.out, f'{name}.pt'))

    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
