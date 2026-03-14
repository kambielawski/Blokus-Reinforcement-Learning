#!/usr/bin/env python3
"""Evaluate a trained AlphaZero network against a random agent.

Usage:
    python scripts/evaluate.py --checkpoint data/checkpoints/checkpoint_best.pt --games 1000
    python scripts/evaluate.py --checkpoint data/checkpoints/checkpoint_best.pt --games 1000 --mode mcts --sims 25
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus.engine.game_state import GameState, ACTION_SPACE_SIZE
from blokus.nn.network import BlokusNetwork, make_pieces_remaining_vector
from blokus.mcts.mcts import MCTS


def load_checkpoint(path: str, device: torch.device):
    """Load checkpoint and return network."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    num_blocks = cfg.get('network', {}).get('num_blocks', 5)
    channels = cfg.get('network', {}).get('channels', 128)
    # Fallback for older checkpoints that stored flat config
    if 'num_blocks' in cfg:
        num_blocks = cfg['num_blocks']
    if 'channels' in cfg:
        channels = cfg['channels']

    network = BlokusNetwork(num_blocks=num_blocks, channels=channels).to(device)
    network.load_state_dict(ckpt['model_state_dict'])
    network.eval()
    iteration = ckpt.get('iteration', '?')
    print(f"Loaded checkpoint: iteration {iteration}, {num_blocks} blocks, {channels} channels")
    return network


def select_action_raw_policy(network, state, device, temperature=0.1):
    """Select action using raw network policy (no MCTS)."""
    legal = state.get_legal_actions()
    if not legal:
        return -1

    board = torch.from_numpy(state.get_nn_state()).unsqueeze(0).to(device)
    pieces = torch.from_numpy(make_pieces_remaining_vector(state)).unsqueeze(0).to(device)
    mask = torch.from_numpy(state.get_legal_actions_mask()).unsqueeze(0).to(device)

    with torch.no_grad():
        log_policy, _ = network(board, pieces, mask)

    # Apply temperature
    probs = torch.exp(log_policy).squeeze(0).cpu().numpy()
    legal_probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for a in legal:
        legal_probs[a] = probs[a]

    if temperature < 0.05:
        # Greedy
        return legal[int(np.argmax([legal_probs[a] for a in legal]))]
    else:
        legal_probs_subset = np.array([legal_probs[a] for a in legal])
        total = legal_probs_subset.sum()
        if total < 1e-10:
            return legal[np.random.randint(len(legal))]
        legal_probs_subset /= total
        # Apply temperature
        log_p = np.log(legal_probs_subset + 1e-10) / temperature
        log_p -= log_p.max()
        p = np.exp(log_p)
        p /= p.sum()
        idx = np.random.choice(len(legal), p=p)
        return legal[idx]


def select_action_random(state):
    """Select a random legal action."""
    legal = state.get_legal_actions()
    if not legal:
        return -1
    return legal[np.random.randint(len(legal))]


def play_game(network, device, mode='raw', mcts_sims=25, game_mode='dual'):
    """Play one game: trained agent (agent 0) vs random agent (agent 1).

    Returns: (winner, scores, num_moves, rewards)
        winner: 0 if trained wins, 1 if random wins, -1 if draw
        scores: dict of agent_idx -> score
        num_moves: total moves played
        rewards: dict of agent_idx -> reward
    """
    state = GameState.new_game(game_mode)
    mcts = None
    if mode == 'mcts':
        mcts = MCTS(
            network=network,
            c_puct=1.5,
            num_simulations=mcts_sims,
            dirichlet_alpha=0.0,  # No exploration noise for eval
            dirichlet_epsilon=0.0,
            temperature=0.1,
            device=device,
        )

    num_moves = 0
    while not state.is_terminal():
        agent_idx = state.get_current_agent()
        legal = state.get_legal_actions()

        if not legal:
            state = state.pass_turn()
            continue

        if agent_idx == 0:
            # Trained agent
            if mode == 'mcts':
                action, _, _ = mcts.select_action(state)
            else:
                action = select_action_raw_policy(network, state, device, temperature=0.1)
        else:
            # Random agent
            action = select_action_random(state)

        state = state.apply_action(action)
        num_moves += 1

    rewards = state.get_rewards()
    scores = state.get_scores()

    # Compute agent scores (dual mode: agent 0 = colors 0+2, agent 1 = colors 1+3)
    if game_mode == 'dual':
        agent_scores = {
            0: scores[0] + scores[2],
            1: scores[1] + scores[3],
        }
    else:
        agent_scores = scores

    if rewards[0] > rewards[1]:
        winner = 0
    elif rewards[1] > rewards[0]:
        winner = 1
    else:
        winner = -1

    return winner, agent_scores, num_moves, rewards


def run_evaluation(network, device, num_games, mode, mcts_sims, game_mode):
    """Run full evaluation and print results."""
    wins = 0
    losses = 0
    draws = 0
    total_score_diff = 0.0
    total_moves = 0
    total_reward_0 = 0.0
    total_reward_1 = 0.0

    start = time.time()
    for i in range(num_games):
        winner, agent_scores, num_moves, rewards = play_game(
            network, device, mode=mode, mcts_sims=mcts_sims, game_mode=game_mode
        )

        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

        total_score_diff += agent_scores[0] - agent_scores[1]
        total_moves += num_moves
        total_reward_0 += rewards[0]
        total_reward_1 += rewards[1]

        if (i + 1) % 100 == 0 or (i + 1) == num_games:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{num_games}] W={wins} L={losses} D={draws} "
                  f"WR={wins/(i+1)*100:.1f}% ({rate:.1f} games/s)")

    elapsed = time.time() - start
    n = num_games

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — {mode.upper()} policy"
          f"{f' ({mcts_sims} sims)' if mode == 'mcts' else ''}")
    print(f"{'='*60}")
    print(f"Games played:        {n}")
    print(f"Trained agent wins:  {wins} ({wins/n*100:.1f}%)")
    print(f"Random agent wins:   {losses} ({losses/n*100:.1f}%)")
    print(f"Draws:               {draws} ({draws/n*100:.1f}%)")
    print(f"Avg score diff:      {total_score_diff/n:+.1f} (trained - random)")
    print(f"Avg game length:     {total_moves/n:.1f} moves")
    print(f"Avg reward trained:  {total_reward_0/n:+.3f}")
    print(f"Avg reward random:   {total_reward_1/n:+.3f}")
    print(f"Total time:          {elapsed:.1f}s ({n/elapsed:.1f} games/s)")
    print(f"{'='*60}")

    return {
        'wins': wins, 'losses': losses, 'draws': draws,
        'win_rate': wins / n,
        'avg_score_diff': total_score_diff / n,
        'avg_game_length': total_moves / n,
        'avg_reward_trained': total_reward_0 / n,
        'avg_reward_random': total_reward_1 / n,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained network vs random agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--games', type=int, default=1000,
                        help='Number of games to play (default: 1000)')
    parser.add_argument('--mode', choices=['raw', 'mcts', 'both'], default='both',
                        help='Policy mode: raw network, MCTS, or both (default: both)')
    parser.add_argument('--sims', type=int, default=25,
                        help='MCTS simulations for mcts mode (default: 25)')
    parser.add_argument('--game-mode', choices=['dual', 'standard'], default='dual',
                        help='Game mode (default: dual)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda (default: auto)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    network = load_checkpoint(args.checkpoint, device)

    modes = []
    if args.mode in ('raw', 'both'):
        modes.append('raw')
    if args.mode in ('mcts', 'both'):
        modes.append('mcts')

    for m in modes:
        print(f"\n--- Running {m.upper()} evaluation ({args.games} games) ---")
        run_evaluation(network, device, args.games, m, args.sims, args.game_mode)


if __name__ == '__main__':
    main()
