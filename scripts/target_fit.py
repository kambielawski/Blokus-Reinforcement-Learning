#!/usr/bin/env python3
"""Does the policy fit its self-play targets? Compares the net's prior with the
buffer's visit targets on the newest positions, by game phase, plus how saturated
the value targets are.

Usage: python scripts/target_fit.py RUN_DIR CHECKPOINT
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus.nn.checkpoint import load_network
from blokus.engine.game_state import ACTION_SPACE_SIZE

torch.set_num_threads(2)
run, ckpt = sys.argv[1], sys.argv[2]
d = np.load(run + '/replay_buffer.npz')
off, idxs, pol, val = d['sparse_offsets'], d['sparse_indices'], d['sparse_policy'], d['value_targets']
boards, pieces = d['board_states'], d['pieces_remaining']
net, info = load_network(ckpt, 'cpu')
print('net iteration', info['iteration'])
n = len(val); rem = pieces.reshape(n, -1).sum(1)
rng = np.random.default_rng(0)
newest = np.arange(n - 30000, n)
for lo, hi in [(70, 85), (50, 70), (30, 50), (0, 30)]:
    sel = newest[(rem[newest] >= lo) & (rem[newest] < hi)]
    sel = rng.choice(sel, 300, replace=False)
    hp, ht, kl, agree, vpred = [], [], [], [], []
    for b in range(0, len(sel), 50):
        ch = sel[b:b + 50]
        mask = np.zeros((len(ch), ACTION_SPACE_SIZE), np.float32)
        for j, i in enumerate(ch): mask[j, idxs[off[i]:off[i + 1]]] = 1
        with torch.no_grad():
            lp, v = net(torch.from_numpy(boards[ch]), torch.from_numpy(pieces[ch]), torch.from_numpy(mask))
        lp = lp.numpy(); v = v.numpy().ravel()
        for j, i in enumerate(ch):
            a = idxs[off[i]:off[i + 1]]; t = pol[off[i]:off[i + 1]].astype(np.float64); t /= t.sum()
            q = np.exp(lp[j, a].astype(np.float64)); q /= q.sum()
            hp.append(-(q * np.log(q + 1e-12)).sum()); nz = t > 0
            ht.append(-(t[nz] * np.log(t[nz])).sum())
            kl.append((t[nz] * (np.log(t[nz]) - np.log(q[nz] + 1e-12))).sum())
            agree.append(a[t.argmax()] == a[q.argmax()])
        vpred.extend(np.sign(v) == np.sign(val[ch]))
    print(f'pieces left {lo:2d}-{hi:2d}: prior eff moves {np.exp(np.mean(hp)):5.1f}  target eff moves {np.exp(np.mean(ht)):5.1f}  '
          f'KL(target||prior) {np.mean(kl):.3f}  argmax agree {np.mean(agree):.2f}  value sign acc {np.mean(vpred):.2f}')
print(f'value targets, newest 30000: |v|>0.9 {np.mean(np.abs(val[newest]) > 0.9):.2f}, '
      f'|v|<0.245 (margin under 5) {np.mean(np.abs(val[newest]) < 0.245):.2f}')
