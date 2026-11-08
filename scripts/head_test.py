#!/usr/bin/env python3
"""Is the policy head the limit? Fit curves for several heads from one parent.

Every arm starts from the parent's trunk, trains the whole network on the parent's
replay buffer with the same learning rate, seed, sample order and rotations, and is
scored every --eval-every steps on a fixed slice of the training buffer and on
held-out games (the newest --holdout-count examples of a later buffer, played by
the parent after its buffer was saved).

Arms:
  kept       parent network, policy head and all
  reinit     parent trunk, current head re-initialized (control for a fresh head)
  fp5        parent trunk, footprint head (3x3 hidden layer, 5x5 masked conv)
  fp7        same with a 7x7 mask that also covers cells touching the shape
  corners    parent network plus 4 corner-availability input planes, new input
             weights zero, so it starts as the parent

Usage:
  python scripts/head_test.py --checkpoint parent.pt --buffer-dir parent/ \
      --holdout-dir later/ --holdout-count 10254 --arm fp5 --steps 7000 --out DIR
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blokus.engine.game_state import RULES_VERSION
from blokus.nn.network import BlokusNetwork, INPUT_CHANNELS
from finetune_buffer import load_buffer

ARMS = {'kept': ('linear', False), 'reinit': ('linear', False), 'fp5': ('footprint5', False),
        'fp7': ('footprint7', False), 'corners': ('linear', True)}


def build_arm(arm: str, parent: dict) -> BlokusNetwork:
    head, corners = ARMS[arm]
    net_cfg = parent['config']['network']
    net = BlokusNetwork(num_blocks=net_cfg['num_blocks'], channels=net_cfg['channels'],
                        policy_head=head, corner_planes=corners)
    state = parent['model_state_dict']
    if arm in ('reinit', 'fp5', 'fp7'):        # Fresh policy head, parent trunk and value head
        state = {k: v for k, v in state.items() if not k.startswith('policy_')}
    if arm == 'corners':                       # Zero weights on the new planes
        state = dict(state)
        old = state['input_conv.weight']
        new = torch.zeros(old.shape[0], net.input_conv.weight.shape[1], *old.shape[2:])
        new[:, :INPUT_CHANNELS] = old
        state['input_conv.weight'] = new
    missing, unexpected = net.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith('policy_') for k in missing), missing
    return net


def batch_tensors(buf, idx, device, rotations=None):
    boards, pieces, values, _ = buf.get_dense_tensors()
    t = torch.from_numpy(idx)
    bs = boards[t]
    if rotations is not None:
        bs = bs.clone()
        for k in range(1, 4):
            chosen = torch.from_numpy(rotations == k)
            bs[chosen] = torch.rot90(bs[chosen], k=-k, dims=(-2, -1))
    lm, pt = buf.reconstruct_sparse_batch(idx, rotations)
    return (bs.to(device), pieces[t].to(device), torch.from_numpy(lm).to(device),
            torch.from_numpy(pt).to(device), values[t].to(device))


def evaluate(net, buf, rows, device, batch=512) -> dict:
    net.eval()
    kl = ce = agree = sq = 0.0
    with torch.no_grad():
        for b in range(0, len(rows), batch):
            bs, pv, lm, pt, vt = batch_tensors(buf, rows[b:b + batch], device)
            logq, v = net(bs, pv, lm)
            plogp = torch.where(pt > 0, pt * torch.log(pt.clamp_min(1e-12)), torch.zeros_like(pt))
            cross = -(pt * logq).sum(1)
            ce += cross.sum().item()
            kl += (plogp.sum(1) + cross).sum().item()
            agree += (pt.argmax(1) == logq.argmax(1)).sum().item()
            sq += ((v - vt) ** 2).sum().item()
    n = len(rows)
    net.train()
    return {'kl': kl / n, 'ce': ce / n, 'agree': agree / n, 'value_mse': sq / n}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--buffer-dir', required=True)
    parser.add_argument('--holdout-dir', required=True)
    parser.add_argument('--holdout-count', type=int, required=True)
    parser.add_argument('--arm', required=True, choices=sorted(ARMS))
    parser.add_argument('--steps', type=int, default=7000)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--train-slice', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=3.1e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = parent['config']
    train_buf = load_buffer(args.buffer_dir)
    hold_buf = load_buffer(args.holdout_dir)
    n = len(train_buf)
    hold_rows = np.arange(len(hold_buf) - args.holdout_count, len(hold_buf))
    train_rows = np.sort(np.random.default_rng(12345).choice(n, args.train_slice, replace=False))

    net = build_arm(args.arm, parent).to(device)
    params = sum(p.numel() for p in net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 weight_decay=cfg['training']['weight_decay'])
    gen = torch.Generator().manual_seed(args.seed)     # Same sample order in every arm
    rng = np.random.default_rng(args.seed)             # Same rotations in every arm
    rotate = cfg['training'].get('rotation_augment', False)
    value_weight = cfg['training'].get('value_loss_weight', 1.0)
    print(f'arm {args.arm}: {params:,} parameters, {n} training examples, '
          f'{len(hold_rows)} held out', flush=True)

    curve = []

    def log(step, loss=None):
        row = {'step': step, 'train': evaluate(net, train_buf, train_rows, device),
               'holdout': evaluate(net, hold_buf, hold_rows, device), 'loss': loss}
        curve.append(row)
        print(json.dumps(row), flush=True)
        with open(os.path.join(args.out, f'{args.arm}.json'), 'w') as f:
            json.dump({'arm': args.arm, 'parameters': params, 'lr': args.lr, 'curve': curve}, f)

    log(0)
    net.train()
    perm, pos, step, started, running = torch.randperm(n, generator=gen).numpy(), 0, 0, time.time(), []
    while step < args.steps:
        if pos + args.batch_size > n:
            perm, pos = torch.randperm(n, generator=gen).numpy(), 0
        idx = perm[pos:pos + args.batch_size]
        pos += args.batch_size
        rotations = rng.integers(0, 4, size=len(idx)) if rotate else None
        bs, pv, lm, pt, vt = batch_tensors(train_buf, idx, device, rotations)
        logq, v = net(bs, pv, lm)
        loss = -(pt * logq).sum(1).mean() + value_weight * nn.functional.mse_loss(v, vt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running.append(loss.item())
        step += 1
        if step % args.eval_every == 0:
            log(step, float(np.mean(running)))
            running = []
            print(f'  {step} steps, {time.time() - started:.0f}s', flush=True)

    torch.save({'iteration': parent['iteration'], 'model_state_dict': net.state_dict(),
                'config': {**cfg, 'network': {**cfg['network'], 'policy_head': ARMS[args.arm][0],
                                              'corner_planes': ARMS[args.arm][1]}},
                'rules_version': RULES_VERSION, 'head_test': vars(args)},
               os.path.join(args.out, f'{args.arm}.pt'))


if __name__ == '__main__':
    main()
