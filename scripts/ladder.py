#!/usr/bin/env python3
"""Round-robin or gauntlet strength ladder between Blokus players (dual mode).

Every pairing plays each opening twice with the seats swapped, so neither first
move nor a lucky opening favors a player. Results append to games.jsonl as games
finish, and rerunning with the same --out skips games already recorded.

Player specs are name=kind[:key=value...]. Kinds:
    random, size, space1, space2, mobility
    raw:ckpt=PATH
    mcts:ckpt=PATH,sims=200,vs=1,batch=8,topk=50,cpuct=1.5,tsd=0,margin=0

Usage:
    python scripts/ladder.py --out data/ladder/base --pairs-per-matchup 20 \\
        --player random=random --player mob=mobility \\
        --player new=mcts:ckpt=data/vacc/scale-v2/checkpoint_latest.pt,sims=200 \\
        --gauntlet new
"""

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus.agents.bots import (
    GreedySizeBot, MctsBot, MobilityBot, OPENING_PLIES, RandomBot, RawPolicyBot,
    SpaceBot, random_opening,
)
from blokus.engine.game_state import GameState
from blokus.nn.checkpoint import load_network

def build_bot(spec: str, seed: int) -> Any:
    """Create a bot from the part of a player spec after 'name='."""
    kind, _, rest = spec.partition(':')
    opts = dict(kv.split('=', 1) for kv in rest.split(',')) if rest else {}
    if kind == 'random':
        return RandomBot(seed)
    if kind == 'size':
        return GreedySizeBot(seed=seed)
    if kind == 'space1':
        return SpaceBot(diagonal=False, seed=seed)
    if kind == 'space2':
        return SpaceBot(diagonal=True, seed=seed)
    if kind == 'mobility':
        return MobilityBot(seed=seed)
    device = torch.device('cpu')
    network, _ = load_network(opts['ckpt'], device)
    if kind == 'raw':
        return RawPolicyBot(network, device)
    if kind == 'mcts':
        return MctsBot(
            network, device,
            sims=int(opts.get('sims', 200)),
            c_puct=float(opts.get('cpuct', 1.5)),
            batch_size=int(opts.get('batch', 8)),
            top_k=int(opts.get('topk', 50)),
            value_scale=float(opts.get('vs', 1.0)),
            terminal_score_diff=bool(int(opts.get('tsd', 0))),
            terminal_margin_scale=float(opts.get('margin', 0)),
        )
    raise ValueError(f'Unknown player kind: {kind}')


_BOTS: Dict[str, Any] = {}


def play_game(task: Dict[str, Any]) -> Dict[str, Any]:
    """Play one game in a worker process. Task seats: index 0 plays agent 0."""
    torch.set_num_threads(1)
    names, specs = task['names'], task['specs']
    bots = []
    for name, spec in zip(names, specs):
        if name not in _BOTS:
            _BOTS[name] = build_bot(spec, seed=hash(name) % 2**32 + os.getpid())
        bots.append(_BOTS[name])

    started = time.time()
    state = GameState.new_game('dual')
    for action in random_opening(task['opening_seed']):
        state = state.apply_action(action)
    plies = OPENING_PLIES
    while not state.is_terminal():
        if not state.get_legal_actions():
            state = state.pass_turn()
            continue
        state = state.apply_action(bots[state.get_current_agent()].select_action(state))
        plies += 1

    scores = state.get_scores()
    return {
        'key': task['key'], 'seat0': names[0], 'seat1': names[1],
        'score0': scores[0] + scores[2], 'score1': scores[1] + scores[3],
        'opening_seed': task['opening_seed'], 'plies': plies,
        'seconds': round(time.time() - started, 1),
    }


def wilson(points: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% interval for a win rate (draws count as half a win)."""
    if n == 0:
        return 0.0, 1.0
    p = points / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(records: List[Dict[str, Any]]) -> str:
    """Per-matchup table with confidence intervals, plus seat balance."""
    matchups: Dict[Tuple[str, str], List[Tuple[float, int, bool]]] = defaultdict(list)
    for r in records:
        a, b = sorted((r['seat0'], r['seat1']))
        a_is_seat0 = r['seat0'] == a
        diff = r['score0'] - r['score1']
        diff = diff if a_is_seat0 else -diff
        result = 1.0 if diff > 0 else (0.5 if diff == 0 else 0.0)
        matchups[(a, b)].append((result, diff, a_is_seat0))

    lines = [f"{'A':<22}{'B':<22}{'games':>6}{'A score':>9}{'95% CI':>15}"
             f"{'diff':>7}{'+/-':>6}{'as seat0':>10}{'as seat1':>10}"]
    for (a, b), rows in sorted(matchups.items()):
        results = np.array([r[0] for r in rows])
        diffs = np.array([r[1] for r in rows], dtype=float)
        seat0 = np.array([r[2] for r in rows])
        lo, hi = wilson(results.sum(), len(rows))
        sem = diffs.std(ddof=1) / math.sqrt(len(rows)) if len(rows) > 1 else float('nan')
        lines.append(
            f'{a:<22}{b:<22}{len(rows):>6}{results.mean():>9.1%}'
            f'{f"[{lo:.1%},{hi:.1%}]":>15}{diffs.mean():>+7.1f}{1.96 * sem:>6.1f}'
            f'{results[seat0].mean() if seat0.any() else float("nan"):>10.1%}'
            f'{results[~seat0].mean() if (~seat0).any() else float("nan"):>10.1%}'
        )
    seat0_wins = sum((r['score0'] > r['score1']) + 0.5 * (r['score0'] == r['score1'])
                     for r in records)
    if records:
        lines.append(f'\nSeat 0 (moves first) scored {seat0_wins / len(records):.1%} '
                     f'over {len(records)} games.')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Blokus strength ladder')
    parser.add_argument('--player', action='append', required=True, help='name=spec')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--pairs-per-matchup', type=int, default=20,
                        help='Openings per matchup; each is played twice with seats swapped')
    parser.add_argument('--gauntlet', action='append', default=None,
                        help='Only pair these players against everyone (repeatable); '
                             'default is a full round-robin')
    parser.add_argument('--matchup', action='append', default=None,
                        help='Explicit A,B pairing (repeatable); overrides --gauntlet')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    players = dict(p.split('=', 1) for p in args.player)
    if args.matchup:
        pairs = [tuple(m.split(',')) for m in args.matchup]
    elif args.gauntlet:
        pairs = [(g, o) for g in args.gauntlet for o in players if o != g]
        pairs = list({tuple(sorted(p)): p for p in pairs}.values())
    else:
        pairs = list(itertools.combinations(players, 2))

    os.makedirs(args.out, exist_ok=True)
    games_path = os.path.join(args.out, 'games.jsonl')
    done: List[Dict[str, Any]] = []
    if os.path.exists(games_path):
        with open(games_path) as f:
            done = [json.loads(line) for line in f if line.strip()]
    done_keys = {r['key'] for r in done}
    with open(os.path.join(args.out, 'players.json'), 'w') as f:
        json.dump(players, f, indent=2)

    tasks = []
    for a, b in pairs:
        for pair_idx in range(args.pairs_per_matchup):
            opening_seed = args.seed * 100003 + pair_idx
            for seats in ((a, b), (b, a)):
                key = f'{seats[0]}|{seats[1]}|{opening_seed}'
                if key not in done_keys:
                    tasks.append({'key': key, 'names': seats,
                                  'specs': (players[seats[0]], players[seats[1]]),
                                  'opening_seed': opening_seed})
    # Spread matchups so partial results are useful and worker load stays mixed
    np.random.default_rng(args.seed).shuffle(tasks)
    print(f'{len(pairs)} matchups, {len(tasks)} games to play, {len(done)} already done',
          flush=True)

    records = list(done)
    started = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers) as pool, open(games_path, 'a') as out:
        for i, record in enumerate(pool.imap_unordered(play_game, tasks), 1):
            out.write(json.dumps(record) + '\n')
            out.flush()
            records.append(record)
            if i % 20 == 0 or i == len(tasks):
                print(f'[{i}/{len(tasks)}] {time.time() - started:.0f}s', flush=True)
                with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
                    f.write(summarize(records) + '\n')

    report = summarize(records)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
