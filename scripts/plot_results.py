#!/usr/bin/env python3
"""Charts for strength ladders and search diagnostics.

Usage:
    python scripts/plot_results.py ladder data/ladder/search-test --opponent mobility \\
        --title "..." --out output/ladder.png
    python scripts/plot_results.py search data/diagnostics/*.json --title "..." --out output/search.png
    python scripts/plot_results.py training a.out b.out --names 5b 10b --title "..." --out output/train.png
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import wilson

SERIES = ('#2a78d6', '#eb6834')  # Checked with the dataviz palette validator
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def plot_ladder(directory: str, opponent: str, labels: Dict[str, str], title: str, out: str) -> None:
    """Score against one opponent, with a 95% interval, one row per player."""
    results: Dict[str, List[float]] = defaultdict(list)
    with open(os.path.join(directory, 'games.jsonl')) as f:
        for line in f:
            game = json.loads(line)
            if opponent not in (game['seat0'], game['seat1']):
                continue
            player = game['seat1'] if game['seat0'] == opponent else game['seat0']
            diff = game['score0'] - game['score1']
            diff = -diff if game['seat0'] == opponent else diff
            results[player].append(1.0 if diff > 0 else (0.5 if diff == 0 else 0.0))

    players = [p for p in labels if p in results]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(players) + 1.9), facecolor=SURFACE)
    _style(ax)
    ax.axvline(50, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    for y, player in enumerate(reversed(players)):
        n = len(results[player])
        points = sum(results[player])
        lo, hi = wilson(points, n)
        small_q = 'x0.1' in labels[player]
        color = SERIES[1] if small_q else SERIES[0]
        ax.plot([100 * lo, 100 * hi], [y, y], color=color, lw=2, solid_capstyle='round', zorder=2)
        ax.scatter(100 * points / n, y, s=64, color=color, marker='s' if small_q else 'o',
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(101.5, y, f'{100 * points / n:.1f}%  (n={n})', va='center', color=INK, fontsize=9)
    ax.set_yticks(range(len(players)))
    ax.set_yticklabels([labels[p] for p in reversed(players)], color=INK, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f'Score against {opponent} (win = 1, draw = ½), with 95% Wilson interval', color=MUTED)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker='o', color=SERIES[0], lw=2, label='Normal value scale'),
               plt.Line2D([], [], marker='s', color=SERIES[1], lw=2, label='Values shrunk x0.1')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              labelcolor=MUTED, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(right=0.86, top=1 - 1.0 / (0.5 * len(players) + 1.9))
    fig.suptitle(title, x=0.01, ha='left', color=INK, fontsize=12, fontweight='bold')
    fig.savefig(out, dpi=170, facecolor=SURFACE)


def plot_search(paths: List[str], title: str, out: str) -> None:
    """Small multiples of how far search moves from the prior."""
    panels = (('kl_visits_prior', 'Visits vs prior (KL)', 'How far search moves play from the prior'),
              ('top_share', 'Top move visit share', 'How decisive search is'),
              ('q_spread', 'Q spread', 'Best minus worst Q among moves with 2%+ visits'))
    rows: List[Tuple[str, Dict[str, float]]] = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for r in sorted(data['results'], key=lambda r: (r['sims'], r['value_scale'])):
            rows.append((f"{data['label']}\nQ x{r['value_scale']:g}, {r['sims']} sims", r))

    fig, axes = plt.subplots(1, 3, figsize=(13, 0.48 * len(rows) + 2.0), sharey=True, facecolor=SURFACE)
    for ax, (key, name, note) in zip(axes, panels):
        _style(ax)
        for y, (_, r) in enumerate(reversed(rows)):
            value = r[key]
            ax.barh(y, value, height=0.62, color=SERIES[0], zorder=2)
            ax.text(value, y, f'  {value:.3f}', va='center', color=INK, fontsize=8.5)
        ax.margins(x=0.22)
        ax.set_title(name, loc='left', color=INK, fontsize=10.5, fontweight='bold')
        ax.set_xlabel(note, color=MUTED, fontsize=8.5)
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([label for label, _ in reversed(rows)], color=INK, fontsize=8.5)
    fig.suptitle(title, x=0.01, ha='left', color=INK, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)


def plot_training(logs: List[str], names: List[str], title: str, out: str) -> None:
    """Training curves from Slurm logs: losses and in-training score against the heuristic."""
    panels = (('policy', 'Policy loss'), ('value', 'Value loss'),
              ('eval', 'Score vs mobility at 50 sims (%)'))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor=SURFACE)
    for index, (path, name) in enumerate(zip(logs, names)):
        series: Dict[str, Tuple[List[int], List[float]]] = {p[0]: ([], []) for p in panels}
        iteration = 0
        with open(path) as f:
            for line in f:
                found = re.match(r'Iteration (\d+)/', line)
                if found:
                    iteration = int(found.group(1))
                found = re.search(r'policy_loss=([\d.]+), value_loss=([\d.]+)', line)
                if found:
                    series['policy'][0].append(iteration)
                    series['policy'][1].append(float(found.group(1)))
                    series['value'][0].append(iteration)
                    series['value'][1].append(float(found.group(2)))
                found = re.search(r'Eval vs mobility: score=([\d.]+)%', line)
                if found:
                    series['eval'][0].append(iteration)
                    series['eval'][1].append(float(found.group(1)))
        for ax, (key, _) in zip(axes, panels):
            marker = 'o' if index == 0 else 's'
            ax.plot(*series[key], color=SERIES[index], lw=2, marker=marker if key == 'eval' else None,
                    markersize=6, label=name)
    for ax, (_, label) in zip(axes, panels):
        _style(ax)
        ax.set_title(label, loc='left', color=INK, fontsize=10.5, fontweight='bold')
        ax.set_xlabel('Iteration', color=MUTED, fontsize=9)
        ax.grid(axis='y', color=GRID, lw=0.8)
    axes[2].set_ylim(0, 100)
    axes[0].legend(frameon=False, labelcolor=MUTED, fontsize=9)
    fig.suptitle(title, x=0.01, ha='left', color=INK, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['ladder', 'search', 'training'])
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--opponent', default='mobility')
    parser.add_argument('--names', nargs='*', default=[], help='Run names, for training charts')
    parser.add_argument('--label', action='append', default=[], help='player=Display name, in row order')
    parser.add_argument('--title', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.kind == 'ladder':
        labels = dict(item.split('=', 1) for item in args.label)
        plot_ladder(args.inputs[0], args.opponent, labels, args.title, args.out)
    elif args.kind == 'training':
        plot_training(args.inputs, args.names, args.title, args.out)
    else:
        plot_search(args.inputs, args.title, args.out)


if __name__ == '__main__':
    main()
