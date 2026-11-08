#!/usr/bin/env python3
"""How noisy are self-play policy targets? Search the same position three times.

Each position gets three independent searches with the self-play settings (root
noise included) and different seeds. If two searches disagree with each other
about as much as the prior disagrees with a search, the prior is at the noise
floor of its targets and a better optimizer cannot help.

Searches leave most moves unvisited, so KL(A||B) is infinite without smoothing;
B is mixed with SMOOTH of the prior for the KL. Total variation and argmax
agreement need no smoothing.

Positions come from games in ladder JSONL files, sampled evenly per game phase
(pieces left over all colors).

Usage:
  python scripts/target_noise.py --checkpoint i276.pt --games a.jsonl b.jsonl \
      --per-phase 300 --workers 16 --out noise.json
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus.engine.game_state import GameState
from blokus.mcts.mcts import MCTS
from blokus.nn.checkpoint import load_network
from blokus.nn.network import make_pieces_remaining_vector

PHASES = [(70, 85), (50, 70), (30, 50), (0, 30)]
SMOOTH = 0.05
_NET = {}


def pieces_left(state: GameState) -> int:
    return sum(len(p) for p in state.pieces_remaining)


def sample_positions(paths, per_phase: int, seed: int):
    """(game line, ply) pairs, evenly per phase, positions with at least two moves."""
    found = {phase: [] for phase in PHASES}
    for path in paths:
        for line in open(path):
            game = json.loads(line)
            state = GameState.new_game('dual')
            for ply, action in enumerate(game['moves']):
                if ply >= 4 and action != -1 and len(state.get_legal_actions()) >= 2:
                    left = pieces_left(state)
                    for lo, hi in PHASES:
                        if lo <= left < hi:
                            found[(lo, hi)].append((game['moves'][:ply], left))
                state = state.pass_turn() if action == -1 else state.apply_action(action)
    rng = random.Random(seed)
    return {phase: rng.sample(items, min(per_phase, len(items))) for phase, items in found.items()}


def search_policy(net, state, sims, seed, cfg):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    mcts = MCTS(net, c_puct=cfg['c_puct'], num_simulations=sims,
                dirichlet_alpha=cfg['dirichlet_alpha'], dirichlet_epsilon=cfg['dirichlet_epsilon'],
                temperature=1.0, device=torch.device('cpu'), batch_size=cfg['batch_size'],
                top_k_actions=cfg['top_k_actions'], terminal_margin_scale=cfg['margin_scale'],
                root_top_k=cfg['root_top_k'])
    policy, _ = mcts.search(state)
    return policy


def measure(task):
    moves, left, checkpoint, sims, seeds, cfg = task
    torch.set_num_threads(1)
    if checkpoint not in _NET:
        _NET[checkpoint] = load_network(checkpoint, torch.device('cpu'))[0]
    net = _NET[checkpoint]
    state = GameState.new_game('dual')
    for action in moves:
        state = state.pass_turn() if action == -1 else state.apply_action(action)
    legal = np.array(state.get_legal_actions())
    with torch.no_grad():
        mask = torch.from_numpy(state.get_legal_actions_mask()[None].astype(np.float32))
        log_prior, _ = net(torch.from_numpy(state.get_nn_state()[None]),
                           torch.from_numpy(make_pieces_remaining_vector(state)[None]), mask)
    prior = np.exp(log_prior[0].numpy()[legal].astype(np.float64))
    prior /= prior.sum()
    a, b, c = (search_policy(net, state, sims, s, cfg)[legal].astype(np.float64) for s in seeds)

    def kl(p, q):
        nz = p > 0
        return float((p[nz] * (np.log(p[nz]) - np.log(q[nz]))).sum())

    smooth = lambda q: (1 - SMOOTH) * q + SMOOTH * prior
    tv = lambda p, q: float(0.5 * np.abs(p - q).sum())
    return {
        'pieces_left': left, 'legal': len(legal),
        'kl_a_prior': kl(a, prior), 'kl_a_b': kl(a, smooth(b)),
        'kl_a_bc': kl(a, smooth((b + c) / 2)),
        'tv_a_prior': tv(a, prior), 'tv_a_b': tv(a, b),
        'agree_a_prior': bool(a.argmax() == prior.argmax()),
        'agree_a_b': bool(a.argmax() == b.argmax()),
        'eff_prior': float(np.exp(-(prior * np.log(prior + 1e-12)).sum())),
        'eff_search': float(np.exp(-(a[a > 0] * np.log(a[a > 0])).sum())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--games', nargs='+', required=True)
    parser.add_argument('--per-phase', type=int, default=300)
    parser.add_argument('--sims', type=int, default=None, help='Default: the checkpoint config')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    info = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['config']
    m = info['mcts']
    cfg = {'c_puct': m['c_puct'], 'dirichlet_alpha': m['dirichlet_alpha'],
           'dirichlet_epsilon': m['dirichlet_epsilon'], 'batch_size': m.get('batch_size', 8),
           'top_k_actions': m.get('top_k_actions', 0), 'root_top_k': m.get('root_top_k'),
           'margin_scale': info['training'].get('margin_scale', 0.0)}
    sims = args.sims or m['num_simulations']
    print(f'sims {sims}, settings {cfg}', flush=True)

    positions = sample_positions(args.games, args.per_phase, args.seed)
    tasks = []
    for i, (phase, items) in enumerate(positions.items()):
        for j, (moves, left) in enumerate(items):
            base = 1000 * (len(tasks) + 1)
            tasks.append((moves, left, args.checkpoint, sims, (base + 1, base + 2, base + 3), cfg))
    print(f'{len(tasks)} positions', flush=True)

    rows = []
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for k, row in enumerate(pool.imap_unordered(measure, tasks, chunksize=4)):
            rows.append(row)
            if (k + 1) % 100 == 0:
                print(f'{k + 1}/{len(tasks)}', flush=True)

    report = {'sims': sims, 'settings': cfg, 'smooth': SMOOTH, 'phases': {}}
    for lo, hi in PHASES:
        sel = [r for r in rows if lo <= r['pieces_left'] < hi]
        if not sel:
            continue
        mean = {k: float(np.mean([r[k] for r in sel])) for k in sel[0] if k not in ('pieces_left',)}
        report['phases'][f'{lo}-{hi}'] = {'positions': len(sel), **mean}
        print(f"pieces left {lo:2d}-{hi:2d} (n={len(sel)}): "
              f"KL(A||prior) {mean['kl_a_prior']:.3f}  KL(A||B) {mean['kl_a_b']:.3f}  "
              f"KL(A||mean BC) {mean['kl_a_bc']:.3f}  TV A-prior {mean['tv_a_prior']:.3f}  "
              f"TV A-B {mean['tv_a_b']:.3f}  agree A-prior {mean['agree_a_prior']:.2f}  "
              f"agree A-B {mean['agree_a_b']:.2f}  eff moves prior {mean['eff_prior']:.1f} "
              f"search {mean['eff_search']:.1f}", flush=True)
    with open(args.out, 'w') as f:
        json.dump({'report': report, 'rows': rows}, f)


if __name__ == '__main__':
    main()
