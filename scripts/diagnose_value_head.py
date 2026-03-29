#!/usr/bin/env python3
"""Diagnostic script for the value head.

Loads a checkpoint and analyzes value head predictions across multiple
dimensions: distribution, game stage, win/loss correlation, and position
discrimination.

Usage:
    python scripts/diagnose_value_head.py --checkpoint data/checkpoints/checkpoint_best_iter117.pt
"""

import argparse
import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from blokus.engine.game_state import GameState, ACTION_SPACE_SIZE
from blokus.nn.network import BlokusNetwork, make_pieces_remaining_vector
from blokus.mcts.mcts import MCTS


def load_model(checkpoint_path: str, device: torch.device) -> BlokusNetwork:
    """Load network from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    net_cfg = cfg.get('network', {})
    network = BlokusNetwork(
        num_blocks=net_cfg.get('num_blocks', 5),
        channels=net_cfg.get('channels', 128),
    ).to(device)
    network.load_state_dict(ckpt['model_state_dict'])
    network.eval()
    print(f"Loaded checkpoint: iteration {ckpt.get('iteration', '?')}")
    return network


@torch.no_grad()
def evaluate_state(network: BlokusNetwork, state: GameState, device: torch.device) -> float:
    """Get value prediction for a single state."""
    board = torch.from_numpy(state.get_nn_state()).unsqueeze(0).to(device)
    pieces = torch.from_numpy(make_pieces_remaining_vector(state)).unsqueeze(0).to(device)
    mask = torch.from_numpy(state.get_legal_actions_mask()).unsqueeze(0).to(device)
    _, value = network(board, pieces, mask)
    return value.item()


def select_action_with_policy(network: BlokusNetwork, state: GameState, device: torch.device) -> int:
    """Select action using greedy raw policy."""
    legal = state.get_legal_actions()
    if not legal:
        return -1
    board = torch.from_numpy(state.get_nn_state()).unsqueeze(0).to(device)
    pieces = torch.from_numpy(make_pieces_remaining_vector(state)).unsqueeze(0).to(device)
    mask = torch.from_numpy(state.get_legal_actions_mask()).unsqueeze(0).to(device)
    with torch.no_grad():
        log_policy, _ = network(board, pieces, mask)
    probs = torch.exp(log_policy).squeeze(0).cpu().numpy()
    best_a = max(legal, key=lambda a: probs[a])
    return best_a


def play_diagnostic_game(network, device):
    """Play one game with trained policy, recording value predictions at each position.

    Returns list of dicts with keys: move_num, value_pred, agent_idx, eventually_won
    """
    state = GameState.new_game('dual')
    history = []
    move_num = 0

    while not state.is_terminal():
        legal = state.get_legal_actions()
        if not legal:
            state = state.pass_turn()
            continue

        agent_idx = state.get_current_agent()
        value_pred = evaluate_state(network, state, device)

        history.append({
            'move_num': move_num,
            'value_pred': value_pred,
            'agent_idx': agent_idx,
            'state': state,  # keep for test 4
        })

        action = select_action_with_policy(network, state, device)
        state = state.apply_action(action)
        move_num += 1

    # Determine winner
    rewards = state.get_rewards()
    winner = 0 if rewards[0] > rewards[1] else (1 if rewards[1] > rewards[0] else -1)

    # Tag each position with whether that agent eventually won
    for h in history:
        if winner == -1:
            h['eventually_won'] = None  # draw
        else:
            h['eventually_won'] = (h['agent_idx'] == winner)

    return history


def test1_value_distribution(all_preds):
    """Test 1: Overall value prediction distribution."""
    preds = np.array(all_preds)
    print("\n" + "=" * 60)
    print("TEST 1: Value Prediction Distribution")
    print("=" * 60)
    print(f"  N positions:  {len(preds)}")
    print(f"  Mean:         {preds.mean():+.4f}")
    print(f"  Std:          {preds.std():.4f}")
    print(f"  Min:          {preds.min():+.4f}")
    print(f"  Max:          {preds.max():+.4f}")
    print(f"  Median:       {np.median(preds):+.4f}")
    print(f"  |pred| < 0.1: {(np.abs(preds) < 0.1).mean():.1%}")
    print(f"  |pred| < 0.3: {(np.abs(preds) < 0.3).mean():.1%}")
    print(f"  |pred| > 0.5: {(np.abs(preds) > 0.5).mean():.1%}")

    # Histogram
    bins = np.linspace(-1.0, 1.0, 21)
    counts, edges = np.histogram(preds, bins=bins)
    print("\n  Histogram (value range → count):")
    for i in range(len(counts)):
        bar = "#" * max(1, int(counts[i] / max(counts) * 40))
        print(f"    [{edges[i]:+.2f}, {edges[i+1]:+.2f}): {counts[i]:5d}  {bar}")

    if preds.std() > 0.3:
        print("\n  >> DIFFERENTIATED: std > 0.3 — value head is producing varied predictions")
    elif preds.std() > 0.1:
        print("\n  >> MODERATE: std 0.1-0.3 — value head has some differentiation")
    else:
        print("\n  >> NEAR-CONSTANT: std < 0.1 — value head outputs ~same value for everything")


def test2_value_vs_game_stage(all_history):
    """Test 2: Value predictions by game stage."""
    print("\n" + "=" * 60)
    print("TEST 2: Value vs Game Stage")
    print("=" * 60)

    bins = [
        ("Early (0-15)", 0, 15),
        ("Mid (16-35)", 16, 35),
        ("Late (36+)", 36, 999),
    ]

    for label, lo, hi in bins:
        preds = [h['value_pred'] for h in all_history if lo <= h['move_num'] <= hi]
        if preds:
            preds = np.array(preds)
            print(f"  {label:16s}: n={len(preds):4d}, mean={preds.mean():+.4f}, "
                  f"std={preds.std():.4f}, |mean|={abs(preds.mean()):.4f}")
        else:
            print(f"  {label:16s}: no data")

    print("\n  >> Expected: |mean| and std should increase in later stages")
    print("     (late-game positions have more certain outcomes)")


def test3_won_vs_lost(all_history):
    """Test 3: Value predictions for won vs lost positions."""
    print("\n" + "=" * 60)
    print("TEST 3: Value on Won vs Lost Positions")
    print("=" * 60)

    won_preds = [h['value_pred'] for h in all_history if h['eventually_won'] is True]
    lost_preds = [h['value_pred'] for h in all_history if h['eventually_won'] is False]
    draw_preds = [h['value_pred'] for h in all_history if h['eventually_won'] is None]

    if won_preds:
        won = np.array(won_preds)
        print(f"  Won positions:   n={len(won):4d}, mean={won.mean():+.4f}, std={won.std():.4f}")
    if lost_preds:
        lost = np.array(lost_preds)
        print(f"  Lost positions:  n={len(lost):4d}, mean={lost.mean():+.4f}, std={lost.std():.4f}")
    if draw_preds:
        draws = np.array(draw_preds)
        print(f"  Draw positions:  n={len(draws):4d}, mean={draws.mean():+.4f}, std={draws.std():.4f}")

    if won_preds and lost_preds:
        gap = np.mean(won_preds) - np.mean(lost_preds)
        print(f"\n  Gap (won - lost): {gap:+.4f}")
        if abs(gap) > 0.3:
            print("  >> GOOD: value head distinguishes won from lost positions")
        elif abs(gap) > 0.1:
            print("  >> WEAK: some signal, but not strong separation")
        else:
            print("  >> BAD: value head doesn't distinguish won from lost (gap < 0.1)")

    # Breakdown by game stage
    print("\n  Won vs Lost by game stage:")
    bins = [("Early", 0, 15), ("Mid", 16, 35), ("Late", 36, 999)]
    for label, lo, hi in bins:
        w = [h['value_pred'] for h in all_history if h['eventually_won'] is True and lo <= h['move_num'] <= hi]
        l = [h['value_pred'] for h in all_history if h['eventually_won'] is False and lo <= h['move_num'] <= hi]
        if w and l:
            gap = np.mean(w) - np.mean(l)
            print(f"    {label:6s}: won_mean={np.mean(w):+.4f}, lost_mean={np.mean(l):+.4f}, gap={gap:+.4f}")


def test4_position_discrimination(all_history, network, device, num_states=20, num_alternatives=10):
    """Test 4: Does value head differentiate between better and worse next positions?"""
    print("\n" + "=" * 60)
    print("TEST 4: Position Discrimination (On vs Off Trajectory)")
    print("=" * 60)

    # Pick mid-game positions
    mid_states = [h for h in all_history if 15 <= h['move_num'] <= 40]
    if len(mid_states) < num_states:
        mid_states = [h for h in all_history if 10 <= h['move_num'] <= 50]

    rng = np.random.RandomState(42)
    selected = rng.choice(len(mid_states), size=min(num_states, len(mid_states)), replace=False)

    on_traj_values = []
    off_traj_stds = []
    all_alt_values = []

    for idx in selected:
        h = mid_states[idx]
        state = h['state']
        on_value = h['value_pred']
        on_traj_values.append(on_value)

        legal = state.get_legal_actions()
        if len(legal) < 2:
            continue

        # Sample random legal moves and evaluate resulting positions
        sample_size = min(num_alternatives, len(legal))
        alt_actions = rng.choice(legal, size=sample_size, replace=False)
        alt_values = []
        for a in alt_actions:
            next_state = state.apply_action(a)
            v = evaluate_state(network, next_state, device)
            alt_values.append(v)

        alt_values = np.array(alt_values)
        all_alt_values.extend(alt_values.tolist())
        off_traj_stds.append(alt_values.std())

    on_arr = np.array(on_traj_values)
    alt_arr = np.array(all_alt_values) if all_alt_values else np.array([0.0])
    std_arr = np.array(off_traj_stds) if off_traj_stds else np.array([0.0])

    print(f"  Tested {len(selected)} mid-game positions, {num_alternatives} alternatives each")
    print(f"\n  On-trajectory values:   mean={on_arr.mean():+.4f}, std={on_arr.std():.4f}")
    print(f"  Off-trajectory values:  mean={alt_arr.mean():+.4f}, std={alt_arr.std():.4f}")
    print(f"  Per-position alt std:   mean={std_arr.mean():.4f}, max={std_arr.max():.4f}")

    if std_arr.mean() > 0.1:
        print("\n  >> DIFFERENTIATED: value head gives different values to different moves")
    elif std_arr.mean() > 0.03:
        print("\n  >> WEAK: some discrimination, but values are similar across moves")
    else:
        print("\n  >> FLAT: value head gives nearly identical values regardless of move (~constant)")


def main():
    parser = argparse.ArgumentParser(description='Value Head Diagnostics')
    parser.add_argument('--checkpoint', type=str,
                        default='data/checkpoints/checkpoint_best_iter117.pt',
                        help='Path to checkpoint')
    parser.add_argument('--num-games', type=int, default=50,
                        help='Number of games to play for analysis')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    ckpt_path = os.path.join(REPO_ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    network = load_model(ckpt_path, device)

    # Play games and collect data
    print(f"\nPlaying {args.num_games} games with trained policy...")
    all_history = []
    all_preds = []
    wins = 0
    for g in range(args.num_games):
        history = play_diagnostic_game(network, device)
        all_history.extend(history)
        all_preds.extend([h['value_pred'] for h in history])

        # Track win rate
        won_positions = [h for h in history if h['eventually_won'] is True and h['agent_idx'] == 0]
        if won_positions:
            wins += 1
        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{args.num_games} games done ({len(all_preds)} positions)")

    print(f"\nTotal: {len(all_preds)} positions from {args.num_games} games")

    # Run all diagnostics
    test1_value_distribution(all_preds)
    test2_value_vs_game_stage(all_history)
    test3_won_vs_lost(all_history)
    test4_position_discrimination(all_history, network, device)

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
