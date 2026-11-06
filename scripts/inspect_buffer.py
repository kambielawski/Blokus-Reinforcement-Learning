#!/usr/bin/env python3
"""Summarize self-play targets in a replay buffer: are visit distributions flat?

Usage: python scripts/inspect_buffer.py RUN_DIR
"""

import os
import sys

import numpy as np

path = os.path.join(sys.argv[1], 'replay_buffer.npz')
with np.load(path) as data:
    offsets = data['sparse_offsets']
    policy = data['sparse_policy']
    values = data['value_targets']

# Newest examples are at the end of the buffer
for name, picks in (('oldest 5000', range(0, min(5000, len(offsets) - 1))),
                    ('newest 5000', range(max(0, len(offsets) - 5001), len(offsets) - 1))):
    entropy, top, legal = [], [], []
    for i in picks:
        p = policy[offsets[i]:offsets[i + 1]]
        p = p[p > 0]
        if p.size == 0:
            continue
        entropy.append(-(p * np.log(p)).sum())
        top.append(p.max())
        legal.append(offsets[i + 1] - offsets[i])
    print(f'{name}: mean target entropy {np.mean(entropy):.2f} '
          f'(effective moves {np.exp(np.mean(entropy)):.1f}), mean top share {np.mean(top):.3f}, '
          f'mean legal moves {np.mean(legal):.0f}')
print(f'value targets: std {values.std():.3f}, examples {len(values)}')
