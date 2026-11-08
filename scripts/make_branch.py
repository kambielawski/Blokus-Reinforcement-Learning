#!/usr/bin/env python3
"""Seed a new run directory from a parent checkpoint and its replay buffer.

The branch resumes from the parent with train.sbatch (it finds checkpoint_latest.pt)
but starts its own W&B run. With --rescale-margin, value targets stored as
tanh(margin / FROM) are rewritten as tanh(margin / TO), so a branch with a new
margin_scale does not train on the old scale for a whole buffer window. Targets at
|v| = 1 in float32 (margin beyond about 9 * FROM) stay at +-1 under any scale.

Usage: python scripts/make_branch.py --checkpoint P.pt --buffer-dir DIR --out RUN_DIR \
           [--rescale-margin 20 50]
"""

import argparse
import os
import shutil

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--buffer-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--rescale-margin', nargs=2, type=float, metavar=('FROM', 'TO'))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=False)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt.pop('wandb_run_id', None)
    ckpt['branch_of'] = os.path.abspath(args.checkpoint)
    torch.save(ckpt, os.path.join(args.out, 'checkpoint_latest.pt'))

    src = os.path.join(args.buffer_dir, 'replay_buffer.npz')
    dst = os.path.join(args.out, 'replay_buffer.npz')
    if args.rescale_margin:
        old, new = args.rescale_margin
        data = dict(np.load(src))
        v = data['value_targets'].astype(np.float64)
        margin = old * np.arctanh(np.clip(v, -1 + 1e-7, 1 - 1e-7))
        data['value_targets'] = np.tanh(margin / new).astype(np.float32)
        np.savez(dst, **data)
        print(f'value targets rescaled {old:g} -> {new:g}: std {v.std():.3f} -> '
              f'{data["value_targets"].std():.3f}')
    else:
        shutil.copyfile(src, dst)
    print(f'branch of iteration {ckpt["iteration"]} written to {args.out}')


if __name__ == '__main__':
    main()
