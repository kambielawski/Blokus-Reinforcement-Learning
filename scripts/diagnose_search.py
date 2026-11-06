#!/usr/bin/env python3
"""Measure whether search moves play away from the network's prior.

Plays a game with the network, then searches sampled positions and reports how
concentrated the visits are, how far they are from the prior, and how much Q varies
across moves. If visits simply match the prior, search adds nothing.

Usage:
    python scripts/diagnose_search.py --checkpoint PATH --sims 200 800 --value-scale 1 0.1 \\
        --out data/diagnostics/imitation.json
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus.engine.game_state import GameState
from blokus.mcts.mcts import MCTS
from blokus.nn.checkpoint import load_network


def sample_positions(network: Any, device: torch.device, games: int, seed: int,
                     margin_scale: float) -> List[GameState]:
    """Positions from fast network self-play, every 4th ply with real choices."""
    np.random.seed(seed)
    mcts = MCTS(network, num_simulations=16, dirichlet_alpha=0.3, temperature=1.0,
                device=device, top_k_actions=50, terminal_margin_scale=margin_scale)
    positions = []
    for _ in range(games):
        state = GameState.new_game('dual')
        ply = 0
        while not state.is_terminal():
            legal = state.get_legal_actions()
            if not legal:
                state = state.pass_turn()
                continue
            if ply % 4 == 2 and len(legal) >= 5:
                positions.append((ply, state))
            state = state.apply_action(mcts.select_action(state)[0])
            ply += 1
    return positions


def measure(mcts: MCTS, state: GameState) -> Dict[str, float]:
    mcts.search(state)
    children = {a: c for a, c in mcts.last_stats['children'].items() if a >= 0}
    visits = np.array([c[0] for c in children.values()], dtype=np.float64)
    q = np.array([c[1] for c in children.values()])
    prior = np.array([c[2] for c in children.values()])
    share = visits / visits.sum()
    prior = prior / prior.sum()
    used = share > 0
    # Moves with at least 2% of the visits have a Q worth comparing
    solid = share >= 0.02
    return {
        'top_share': float(share.max()),
        'effective_moves': float(math.exp(-(share[used] * np.log(share[used])).sum())),
        'prior_effective_moves': float(math.exp(-(prior * np.log(prior + 1e-12)).sum())),
        'kl_visits_prior': float((share[used] * np.log(share[used] / (prior[used] + 1e-12))).sum()),
        'q_spread': float(q[solid].max() - q[solid].min()) if solid.sum() > 1 else 0.0,
        'moved_off_prior_top': float(np.argmax(share) != np.argmax(prior)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--label', default=None)
    parser.add_argument('--sims', type=int, nargs='+', default=[200, 800])
    parser.add_argument('--value-scale', type=float, nargs='+', default=[1.0])
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--max-positions', type=int, default=30)
    parser.add_argument('--margin-scale', type=float, default=None,
                        help='Terminal value scale; default comes from the checkpoint')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    device = torch.device('cpu')
    torch.set_num_threads(1)
    network, info = load_network(args.checkpoint, device)
    margin = args.margin_scale
    if margin is None:
        margin = info['config'].get('training', {}).get('margin_scale', 0.0)

    positions = sample_positions(network, device, args.games, 0, margin)
    rng = np.random.default_rng(0)
    if len(positions) > args.max_positions:
        chosen = rng.choice(len(positions), args.max_positions, replace=False)
        positions = [positions[i] for i in chosen]

    results = []
    for sims in args.sims:
        for scale in args.value_scale:
            mcts = MCTS(network, num_simulations=sims, dirichlet_alpha=0.0, temperature=0.0,
                        device=device, top_k_actions=50, value_scale=scale,
                        terminal_margin_scale=margin)
            rows = [dict(measure(mcts, state), ply=ply) for ply, state in positions]
            summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'ply'}
            summary.update(sims=sims, value_scale=scale, positions=len(rows))
            results.append(summary)
            print(json.dumps(summary), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'label': args.label or args.checkpoint, 'iteration': info['iteration'],
                   'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
