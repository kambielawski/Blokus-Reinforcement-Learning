#!/usr/bin/env python3
"""Where was a lost game lost? Label the loser's positions from the end backward.

For each game, every position where the model is to move gets a label:
  exact  -- EndgameSolver proves whether the model could still force a win.
            Runs from the last move backward while each position fits the budget.
  mcts   -- deep MCTS value estimate (from the model's side), for positions
            before the exact search's reach, plus a few inside it to show how
            well MCTS estimates agree with proof.

Input is a JSONL of games as written by scripts/ladder.py, each with extra fields
'model_seat' and 'checkpoint'.

Usage:
  python scripts/endgame_labels.py GAMES.jsonl --out labels.json \
      --node-budget 5000000 --mcts-sims 25000 --mcts-back 12 --workers 32
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus.agents.bots import MctsBot
from blokus.engine.endgame import EndgameSolver
from blokus.engine.game_state import GameState
from blokus.nn.checkpoint import load_network


def model_positions(game: dict):
    """(ply, state) for every position where the model is to move with a choice of moves."""
    seat = game['model_seat']
    state = GameState.new_game('dual')
    out = []
    for ply, action in enumerate(game['moves']):
        if action != -1 and state.get_current_agent() == seat and ply >= 4:
            out.append((ply, state))
        state = state.pass_turn() if action == -1 else state.apply_action(action)
    return out


def pieces_left(state: GameState) -> int:
    return sum(len(p) for p in state.pieces_remaining)


def exact_labels(args):
    """Backward exact pass for one game; stops at the first position over budget."""
    index, game, budget = args
    solver = EndgameSolver(node_budget=budget)
    labels = []
    for ply, state in reversed(model_positions(game)):
        started = time.time()
        win = solver.has_win(state)
        labels.append({'ply': ply, 'pieces_left': pieces_left(state),
                       'legal': len(state.get_legal_actions()),
                       'nodes': solver.nodes, 'seconds': round(time.time() - started, 1),
                       'source': 'exact', 'win': win})
        print(f'game {index} ply {ply}: win {win} ({solver.nodes} nodes)', flush=True)
        if win is None:
            break
    return index, labels


def mcts_label(args):
    index, game, ply, sims = args
    torch.set_num_threads(1)
    state = dict(model_positions(game))[ply]
    network, info = load_network(game['checkpoint'], torch.device('cpu'))
    scale = info['config'].get('training', {}).get('margin_scale', 0)
    bot = MctsBot(network, torch.device('cpu'), sims=sims, terminal_margin_scale=scale)
    started = time.time()
    action = bot.select_action(state)
    stats = bot.last_stats
    visits, q, _ = stats['children'][action]
    return index, {'ply': ply, 'pieces_left': pieces_left(state), 'source': 'mcts',
                   'sims': stats['sims'], 'root_value': round(stats['value'], 3),
                   'best_q': round(q, 3), 'best_visits': visits,
                   'win': q > 0, 'seconds': round(time.time() - started, 1)}


def summarize(game: dict, labels: list) -> dict:
    exact = [l for l in labels if l['source'] == 'exact' and l['win'] is not None]
    mcts = [l for l in labels if l['source'] == 'mcts']
    proven = [l for l in exact if l['win']]
    estimated = [l for l in mcts if l['win']]
    reach = min(exact, key=lambda l: l['ply']) if exact else None
    last_proven = max(proven, key=lambda l: l['ply']) if proven else None
    before_reach = [l for l in estimated if reach is None or l['ply'] < reach['ply']]
    last_estimated = max(before_reach, key=lambda l: l['ply']) if before_reach else None
    pick = lambda l: None if l is None else {'ply': l['ply'], 'pieces_left': l['pieces_left']}
    return {'key': game['key'], 'model_seat': game['model_seat'], 'margin': game['margin'],
            'plies': len(game['moves']), 'checkpoint': os.path.basename(game['checkpoint']),
            'exact_reach': pick(reach), 'last_proven_win': pick(last_proven),
            'last_mcts_win_before_reach': pick(last_estimated),
            'turned_within_exact_range': last_proven is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('games')
    parser.add_argument('--out', required=True)
    parser.add_argument('--node-budget', type=int, default=5_000_000)
    parser.add_argument('--mcts-sims', type=int, default=25_000)
    parser.add_argument('--mcts-back', type=int, default=12,
                        help='Model positions before the exact reach to label with MCTS')
    parser.add_argument('--mcts-inside', type=int, default=3,
                        help='Positions inside the exact reach also labeled with MCTS')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    games = [json.loads(line) for line in open(args.games)]
    labels = {i: [] for i in range(len(games))}
    ctx = mp.get_context('spawn')

    with ctx.Pool(min(args.workers, len(games))) as pool:
        for index, found in pool.imap_unordered(
                exact_labels, [(i, g, args.node_budget) for i, g in enumerate(games)]):
            labels[index].extend(found)

    tasks = []
    for i, game in enumerate(games):
        plies = [p for p, _ in model_positions(game)]
        solved = sorted(l['ply'] for l in labels[i] if l['win'] is not None)
        reach = solved[0] if solved else plies[-1] + 1
        before = [p for p in plies if p < reach][-args.mcts_back:]
        inside = solved[:args.mcts_inside]
        tasks += [(i, game, p, args.mcts_sims) for p in before + inside]
    print(f'{len(tasks)} MCTS positions at {args.mcts_sims} sims', flush=True)
    with ctx.Pool(args.workers) as pool:
        for index, label in pool.imap_unordered(mcts_label, tasks):
            labels[index].append(label)
            print(f"game {index} ply {label['ply']}: mcts q {label['best_q']:+.3f}", flush=True)

    report = []
    for i, game in enumerate(games):
        rows = sorted(labels[i], key=lambda l: (l['ply'], l['source']))
        report.append({'summary': summarize(game, rows), 'labels': rows})
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=1)
    for r in report:
        print(json.dumps(r['summary']))


if __name__ == '__main__':
    main()
