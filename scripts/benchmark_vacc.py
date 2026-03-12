#\!/usr/bin/env python3
"""Benchmark self-play on VACC: CPU vs CUDA, various sim counts and batch sizes."""

import sys
import os
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blokus.nn.network import BlokusNetwork, make_pieces_remaining_vector
from blokus.engine.game_state import GameState
from blokus.agents.alpha_zero import self_play_game
from blokus.mcts.mcts import MCTS

def profile_components(net, device, device_name):
    """Profile individual components on the given device."""
    print(f"\n=== Component Profiling ({device_name}) ===")
    
    state = GameState.new_game("dual")
    
    # Legal move gen (first move)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        legal = state.get_legal_actions()
        times.append(time.perf_counter() - t0)
    print(f"  Legal move gen (first move): {np.mean(times)*1000:.2f}ms (n={len(legal)})")
    
    # Advance to mid-game
    import random
    rng = random.Random(42)
    for _ in range(10):
        legal = state.get_legal_actions()
        if legal:
            state = state.apply_action(rng.choice(legal))
        else:
            state = state.pass_turn()
    
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        legal = state.get_legal_actions()
        times.append(time.perf_counter() - t0)
    print(f"  Legal move gen (mid-game): {np.mean(times)*1000:.2f}ms (n={len(legal)})")
    
    # NN inference batch=1
    board_t = torch.from_numpy(state.get_nn_state()).unsqueeze(0).to(device)
    pv_t = torch.from_numpy(make_pieces_remaining_vector(state)).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(state.get_legal_actions_mask()).unsqueeze(0).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            net(board_t, pv_t, mask_t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            net(board_t, pv_t, mask_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"  NN inference (batch=1): {np.mean(times)*1000:.2f}ms")
    
    # Batched inference
    for bs in [4, 8, 16, 32]:
        board_b = torch.randn(bs, 5, 20, 20).to(device)
        pv_b = torch.ones(bs, 84).to(device)
        mask_b = torch.ones(bs, 67200).to(device)
        for _ in range(5):
            with torch.no_grad():
                net(board_b, pv_b, mask_b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            with torch.no_grad():
                net(board_b, pv_b, mask_b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        print(f"  NN inference (batch={bs}): {np.mean(times)*1000:.2f}ms total, {np.mean(times)*1000/bs:.2f}ms/sample")
    
    # Tensor transfer
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        b = torch.from_numpy(state.get_nn_state()).unsqueeze(0).to(device)
        p = torch.from_numpy(make_pieces_remaining_vector(state)).unsqueeze(0).to(device)
        m = torch.from_numpy(state.get_legal_actions_mask()).unsqueeze(0).to(device)
        times.append(time.perf_counter() - t0)
    print(f"  Tensor creation + transfer: {np.mean(times)*1000:.2f}ms")


def benchmark_selfplay(net, device, device_name, sims_list=[25, 50, 100, 200]):
    """Benchmark full self-play games."""
    print(f"\n=== Self-Play Benchmark ({device_name}) ===")
    
    for sims in sims_list:
        t0 = time.time()
        examples = self_play_game(
            network=net,
            game_mode="dual",
            num_simulations=sims,
            device=device,
        )
        elapsed = time.time() - t0
        num_moves = len(examples)
        time_per_sim = elapsed / (num_moves * sims) * 1000 if num_moves > 0 else 0
        print(f"  sims={sims:3d}: {elapsed:6.1f}s | {num_moves} moves | "
              f"{elapsed/num_moves:.2f}s/move | {time_per_sim:.1f}ms/sim")


def main():
    print("=" * 60)
    print("Blokus AlphaZero — VACC Benchmark")
    print("=" * 60)
    
    # GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        print("No CUDA GPU available")
    
    net = BlokusNetwork(num_blocks=5, channels=128)
    params = sum(p.numel() for p in net.parameters())
    print(f"Network: 5 blocks, 128 channels, {params:,} params")
    
    # CPU benchmark
    cpu_device = torch.device("cpu")
    net_cpu = net.to(cpu_device)
    profile_components(net_cpu, cpu_device, "CPU")
    benchmark_selfplay(net_cpu, cpu_device, "CPU")
    
    # CUDA benchmark
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda:0")
        net_cuda = net.to(cuda_device)
        profile_components(net_cuda, cuda_device, f"CUDA ({torch.cuda.get_device_name(0)})")
        benchmark_selfplay(net_cuda, cuda_device, f"CUDA ({torch.cuda.get_device_name(0)})")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
