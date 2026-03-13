#!/usr/bin/env python3
"""Comprehensive throughput and bottleneck analysis for Blokus AlphaZero training.

Profiles:
1. Per-simulation cost breakdown (legal move gen, NN inference, tree ops, overhead)
2. Full training iteration (self-play + gradient updates)
3. GPU utilization estimate
4. Max throughput projections
"""

import sys
import os
import time
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus.nn.network import BlokusNetwork, make_pieces_remaining_vector
from blokus.engine.game_state import GameState, ACTION_SPACE_SIZE
from blokus.agents.alpha_zero import self_play_game, AlphaZeroAgent, TrainingExample
from blokus.mcts.mcts import MCTS, MCTSNode


def profile_per_simulation(net, device):
    """Break down cost of a single MCTS simulation."""
    print("\n" + "=" * 60)
    print("1. PER-SIMULATION COST BREAKDOWN")
    print("=" * 60)

    state = GameState.new_game('dual')
    rng = random.Random(42)
    # Advance to mid-game for realistic profiling
    for _ in range(10):
        legal = state.get_legal_actions()
        if legal:
            state = state.apply_action(rng.choice(legal))
        else:
            state = state.pass_turn()

    N = 200  # repetitions for timing

    # 1a. Legal move generation
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        legal = state.get_legal_actions()
        times.append(time.perf_counter() - t0)
    legal_ms = np.mean(times) * 1000
    print(f"\n  Legal move gen:        {legal_ms:.3f} ms")

    # 1b. Legal actions mask
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        mask = state.get_legal_actions_mask()
        times.append(time.perf_counter() - t0)
    mask_ms = np.mean(times) * 1000
    print(f"  Legal actions mask:    {mask_ms:.3f} ms")

    # 1c. get_nn_state
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        nn_state = state.get_nn_state()
        times.append(time.perf_counter() - t0)
    nn_state_ms = np.mean(times) * 1000
    print(f"  get_nn_state:          {nn_state_ms:.3f} ms")

    # 1d. make_pieces_remaining_vector
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        pv = make_pieces_remaining_vector(state)
        times.append(time.perf_counter() - t0)
    pv_ms = np.mean(times) * 1000
    print(f"  pieces_remaining_vec:  {pv_ms:.3f} ms")

    # 1e. Tensor creation + transfer to device
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        bt = torch.from_numpy(state.get_nn_state()).unsqueeze(0).to(device)
        pt = torch.from_numpy(make_pieces_remaining_vector(state)).unsqueeze(0).to(device)
        mt = torch.from_numpy(state.get_legal_actions_mask()).unsqueeze(0).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    transfer_ms = np.mean(times) * 1000
    print(f"  Tensor create+xfer:   {transfer_ms:.3f} ms")

    # 1f. NN forward pass (batch=1 and batch=8)
    bt = torch.from_numpy(state.get_nn_state()).unsqueeze(0).to(device)
    pt = torch.from_numpy(make_pieces_remaining_vector(state)).unsqueeze(0).to(device)
    mt = torch.from_numpy(state.get_legal_actions_mask()).unsqueeze(0).to(device)
    # warmup
    for _ in range(20):
        with torch.no_grad():
            net(bt, pt, mt)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        with torch.no_grad():
            net(bt, pt, mt)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    nn_b1_ms = np.mean(times) * 1000
    print(f"  NN forward (batch=1):  {nn_b1_ms:.3f} ms")

    bt8 = torch.randn(8, 5, 20, 20).to(device)
    pt8 = torch.ones(8, 84).to(device)
    mt8 = torch.ones(8, 67200).to(device)
    for _ in range(20):
        with torch.no_grad():
            net(bt8, pt8, mt8)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        with torch.no_grad():
            net(bt8, pt8, mt8)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    nn_b8_ms = np.mean(times) * 1000
    print(f"  NN forward (batch=8):  {nn_b8_ms:.3f} ms total, {nn_b8_ms/8:.3f} ms/sample")

    # 1g. apply_action
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = state.apply_action(legal[0])
        times.append(time.perf_counter() - t0)
    apply_ms = np.mean(times) * 1000
    print(f"  apply_action:          {apply_ms:.3f} ms")

    # 1h. PUCT selection (simulate selecting from a node with children)
    # Build a fake expanded node
    mcts = MCTS(net, num_simulations=50, device=device, batch_size=8)
    root = MCTSNode(state)
    mcts._expand_single(root, add_noise=True)
    # Give children some visits
    for child in root.children.values():
        child.visit_count = random.randint(1, 20)
        child.total_value = random.random() * child.visit_count
    times = []
    for _ in range(N * 10):
        t0 = time.perf_counter()
        mcts._select_child(root)
        times.append(time.perf_counter() - t0)
    select_ms = np.mean(times) * 1000
    print(f"  PUCT selection:        {select_ms:.4f} ms")

    # 1i. Backup
    leaf = list(root.children.values())[0]
    times = []
    for _ in range(N * 10):
        t0 = time.perf_counter()
        mcts._backup(leaf, 0.5)
        times.append(time.perf_counter() - t0)
        # undo to keep state consistent
        leaf.visit_count -= 1
    backup_ms = np.mean(times) * 1000
    print(f"  Backup (1 level):      {backup_ms:.4f} ms")

    # Summary: estimate per-simulation cost
    # A simulation = select (tree traversal) + expand (legal moves + NN) + backup
    # With batching (batch=8): 8 sims share one NN call
    nn_per_sim = nn_b8_ms / 8
    # Tree depth averages ~3-5 levels in early search
    tree_depth = 4
    select_total = select_ms * tree_depth
    expand_cost = legal_ms + nn_state_ms + pv_ms + transfer_ms / 8  # transfer amortized over batch
    backup_cost = backup_ms * (tree_depth + 1)

    print(f"\n  --- Estimated per-simulation cost (batch=8) ---")
    print(f"  Tree traversal ({tree_depth} levels):  {select_total:.3f} ms")
    print(f"  Expand (legal+state):     {legal_ms + nn_state_ms + pv_ms:.3f} ms")
    print(f"  Tensor xfer (amortized):  {transfer_ms/8:.3f} ms")
    print(f"  NN forward (amortized):   {nn_per_sim:.3f} ms")
    print(f"  Backup ({tree_depth+1} levels):           {backup_cost:.3f} ms")
    total_est = select_total + expand_cost + nn_per_sim + backup_cost + transfer_ms / 8
    print(f"  TOTAL estimated:          {total_est:.3f} ms/sim")

    return {
        'legal_ms': legal_ms, 'nn_b1_ms': nn_b1_ms, 'nn_b8_ms': nn_b8_ms,
        'transfer_ms': transfer_ms, 'apply_ms': apply_ms,
        'select_ms': select_ms, 'backup_ms': backup_ms,
        'nn_state_ms': nn_state_ms, 'pv_ms': pv_ms, 'mask_ms': mask_ms,
    }


def profile_full_mcts_search(net, device):
    """Profile a complete MCTS search call with instrumented timing."""
    print("\n" + "=" * 60)
    print("2. FULL MCTS SEARCH PROFILING")
    print("=" * 60)

    state = GameState.new_game('dual')
    rng = random.Random(42)
    for _ in range(10):
        legal = state.get_legal_actions()
        if legal:
            state = state.apply_action(rng.choice(legal))
        else:
            state = state.pass_turn()

    for sims in [25, 50, 100]:
        mcts = MCTS(net, num_simulations=sims, device=device, batch_size=8)

        # Warmup
        mcts.search(state)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            policy, value = mcts.search(state)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        per_sim = avg_ms / sims
        print(f"  sims={sims:3d}: {avg_ms:.1f} ms total, {per_sim:.2f} ms/sim")


def profile_training_iteration(net, device):
    """Profile a full training iteration: self-play + training."""
    print("\n" + "=" * 60)
    print("3. FULL TRAINING ITERATION PROFILING")
    print("=" * 60)

    games_per_iter = 5
    sims = 100
    train_epochs = 5
    train_batch_size = 64

    # --- Self-play phase ---
    print(f"\n  Self-play: {games_per_iter} games, sims={sims}")
    all_examples = []
    t_selfplay_start = time.time()
    game_times = []
    for g in range(games_per_iter):
        t0 = time.time()
        examples = self_play_game(
            network=net, game_mode='dual',
            num_simulations=sims, device=device,
        )
        game_times.append(time.time() - t0)
        all_examples.extend(examples)
        print(f"    Game {g+1}: {game_times[-1]:.1f}s, {len(examples)} examples")
    t_selfplay = time.time() - t_selfplay_start
    print(f"  Self-play total: {t_selfplay:.1f}s ({len(all_examples)} examples)")

    # --- Training phase ---
    print(f"\n  Training: {train_epochs} epochs, batch_size={train_batch_size}")
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
    from train import train_on_examples

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    t_train_start = time.time()
    stats = train_on_examples(
        net, all_examples, optimizer,
        batch_size=train_batch_size, epochs=train_epochs, device=device,
    )
    t_train = time.time() - t_train_start

    print(f"  Training total: {t_train:.1f}s")
    print(f"    Policy loss: {stats['policy_loss']:.4f}")
    print(f"    Value loss:  {stats['value_loss']:.4f}")
    print(f"    Total loss:  {stats['total_loss']:.4f}")

    # --- Summary ---
    t_total = t_selfplay + t_train
    print(f"\n  --- Iteration Summary ---")
    print(f"  Self-play:  {t_selfplay:6.1f}s  ({t_selfplay/t_total*100:.1f}%)")
    print(f"  Training:   {t_train:6.1f}s  ({t_train/t_total*100:.1f}%)")
    print(f"  TOTAL:      {t_total:6.1f}s")

    return {
        't_selfplay': t_selfplay, 't_train': t_train,
        'n_examples': len(all_examples),
        'games_per_iter': games_per_iter,
        'game_times': game_times,
    }


def profile_training_step(net, device, examples):
    """Profile individual training step components."""
    print("\n" + "=" * 60)
    print("4. TRAINING STEP BREAKDOWN")
    print("=" * 60)

    batch_size = 64
    n = min(batch_size, len(examples))

    # Prepare a batch
    board_states = np.stack([ex.board_state for ex in examples[:n]])
    pieces_vecs = np.stack([ex.pieces_remaining for ex in examples[:n]])
    legal_masks = np.stack([ex.legal_mask for ex in examples[:n]])
    policy_targets = np.stack([ex.policy_target for ex in examples[:n]])
    value_targets = np.array([ex.value_target for ex in examples[:n]], dtype=np.float32)

    N = 50

    # Data loading (numpy to tensor + transfer)
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        bt = torch.from_numpy(board_states).to(device)
        pt = torch.from_numpy(pieces_vecs).to(device)
        mt = torch.from_numpy(legal_masks).to(device)
        pol_t = torch.from_numpy(policy_targets).to(device)
        val_t = torch.from_numpy(value_targets).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    data_ms = np.mean(times) * 1000
    print(f"  Data loading (batch={n}):  {data_ms:.2f} ms")

    # Forward pass
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    bt = torch.from_numpy(board_states).to(device)
    pt = torch.from_numpy(pieces_vecs).to(device)
    mt = torch.from_numpy(legal_masks).to(device)
    pol_t = torch.from_numpy(policy_targets).to(device)
    val_t = torch.from_numpy(value_targets).to(device)

    # warmup
    for _ in range(5):
        log_policy, value = net(bt, pt, mt)
        policy_loss = -(pol_t * log_policy).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(value, val_t)
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times_fwd = []
    times_loss = []
    times_bwd = []
    times_step = []
    for _ in range(N):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        log_policy, value = net(bt, pt, mt)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        policy_loss = -(pol_t * log_policy).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(value, val_t)
        loss = policy_loss + value_loss
        t2 = time.perf_counter()

        optimizer.zero_grad()
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        times_fwd.append(t1 - t0)
        times_loss.append(t2 - t1)
        times_bwd.append(t3 - t2)
        times_step.append(t4 - t3)

    fwd_ms = np.mean(times_fwd) * 1000
    loss_ms = np.mean(times_loss) * 1000
    bwd_ms = np.mean(times_bwd) * 1000
    step_ms = np.mean(times_step) * 1000
    total_ms = data_ms + fwd_ms + loss_ms + bwd_ms + step_ms

    print(f"  Forward pass:             {fwd_ms:.2f} ms")
    print(f"  Loss computation:         {loss_ms:.2f} ms")
    print(f"  Backward pass:            {bwd_ms:.2f} ms")
    print(f"  Optimizer step:           {step_ms:.2f} ms")
    print(f"  TOTAL per batch:          {total_ms:.2f} ms")


def gpu_utilization_estimate(net, device, iter_stats):
    """Estimate GPU utilization during self-play."""
    print("\n" + "=" * 60)
    print("5. GPU UTILIZATION ESTIMATE")
    print("=" * 60)

    if device.type != 'cuda':
        print("  Skipped (CPU mode)")
        return

    state = GameState.new_game('dual')
    rng = random.Random(42)
    for _ in range(10):
        legal = state.get_legal_actions()
        if legal:
            state = state.apply_action(rng.choice(legal))
        else:
            state = state.pass_turn()

    # Measure pure GPU compute time for a batch of 8
    bt = torch.randn(8, 5, 20, 20).to(device)
    pt = torch.ones(8, 84).to(device)
    mt = torch.ones(8, 67200).to(device)

    # Use CUDA events for precise GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(20):
        with torch.no_grad():
            net(bt, pt, mt)
    torch.cuda.synchronize()

    gpu_times = []
    for _ in range(100):
        start_event.record()
        with torch.no_grad():
            net(bt, pt, mt)
        end_event.record()
        torch.cuda.synchronize()
        gpu_times.append(start_event.elapsed_time(end_event))  # ms

    gpu_fwd_ms = np.mean(gpu_times)
    print(f"  GPU forward time (batch=8): {gpu_fwd_ms:.3f} ms (CUDA events)")

    # During self-play with sims=100, batch=8:
    # ~13 batched NN calls per move (100/8 ≈ 13)
    # Plus 1 root expansion (batch=1)
    nn_calls_per_move = 100 / 8 + 1
    gpu_time_per_move = nn_calls_per_move * gpu_fwd_ms  # ms

    avg_game_time = np.mean(iter_stats['game_times'])
    avg_moves = iter_stats['n_examples'] / iter_stats['games_per_iter']
    wall_time_per_move = avg_game_time / avg_moves * 1000  # ms

    gpu_util = gpu_time_per_move / wall_time_per_move * 100

    print(f"  NN calls per move (~sims=100, batch=8): {nn_calls_per_move:.0f}")
    print(f"  GPU compute per move: {gpu_time_per_move:.1f} ms")
    print(f"  Wall time per move:   {wall_time_per_move:.1f} ms")
    print(f"  GPU utilization:      {gpu_util:.1f}%")
    print(f"  CPU-bound overhead:   {100-gpu_util:.1f}%")


def throughput_projections(iter_stats):
    """Project max throughput numbers."""
    print("\n" + "=" * 60)
    print("6. THROUGHPUT PROJECTIONS")
    print("=" * 60)

    avg_game_time = np.mean(iter_stats['game_times'])
    n_examples = iter_stats['n_examples']
    games = iter_stats['games_per_iter']
    avg_examples_per_game = n_examples / games

    games_per_hour = 3600 / avg_game_time
    examples_per_hour = games_per_hour * avg_examples_per_game

    t_total = iter_stats['t_selfplay'] + iter_stats['t_train']
    iters_per_hour = 3600 / t_total

    print(f"\n  Single-process (sims=100):")
    print(f"    Avg game time:          {avg_game_time:.1f}s")
    print(f"    Avg examples/game:      {avg_examples_per_game:.0f}")
    print(f"    Games/hour:             {games_per_hour:.0f}")
    print(f"    Examples/hour:           {examples_per_hour:.0f}")
    print(f"    Iterations/hour (5g):   {iters_per_hour:.1f}")

    # Multi-process projections
    # Self-play is embarrassingly parallel (independent games)
    # But all share one GPU for NN inference
    print(f"\n  Multi-process projections (self-play parallelism):")
    for n_workers in [2, 4, 8]:
        # Assume linear speedup on self-play (CPU-bound parts)
        # GPU is underutilized so can handle more concurrent requests
        parallel_selfplay = iter_stats['t_selfplay'] / n_workers
        parallel_total = parallel_selfplay + iter_stats['t_train']
        parallel_iters = 3600 / parallel_total
        parallel_games = games_per_hour * n_workers  # approx
        print(f"    {n_workers} workers: ~{parallel_games:.0f} games/hr, "
              f"~{parallel_iters:.1f} iters/hr (5g/iter)")


def main():
    print("=" * 60)
    print("Blokus AlphaZero — Comprehensive Training Profiler")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        device = torch.device('cpu')
        print("No CUDA GPU — running on CPU")

    net = BlokusNetwork(num_blocks=5, channels=128).to(device)
    params = sum(p.numel() for p in net.parameters())
    print(f"Network: {params:,} params")

    # 1. Per-simulation breakdown
    comp_stats = profile_per_simulation(net, device)

    # 2. Full MCTS search timing
    profile_full_mcts_search(net, device)

    # 3. Full training iteration
    iter_stats = profile_training_iteration(net, device)

    # 4. Training step breakdown
    examples = self_play_game(
        network=net, game_mode='dual',
        num_simulations=50, max_moves=20, device=device,
    )
    profile_training_step(net, device, examples)

    # 5. GPU utilization
    gpu_utilization_estimate(net, device, iter_stats)

    # 6. Throughput projections
    throughput_projections(iter_stats)

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
