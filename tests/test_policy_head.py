"""Footprint policy head and corner planes line up with the engine's placements."""

import numpy as np
import pytest
import torch

from blokus.agents.bots import RandomBot
from blokus.engine.game_state import GameState, BOARD_SIZE, decode_action
from blokus.nn.checkpoint import load_network
from blokus.nn.network import (
    BlokusNetwork, POLICY_CHANNELS, MAX_ORIENTATIONS, corner_planes, footprint_masks,
    make_pieces_remaining_vector,
)


@pytest.fixture(scope='module')
def states():
    """Mid-game positions from a few random games."""
    found = []
    for seed in range(3):
        bot, state, ply = RandomBot(seed), GameState.new_game('dual'), 0
        while not state.is_terminal():
            if ply in (8, 20, 36) and state.get_legal_actions():
                found.append(state)
            actions = state.get_legal_actions()
            state = state.apply_action(bot.select_action(state)) if actions else state.pass_turn()
            ply += 1
    return found


def placed_cells(state, action):
    after = state.apply_action(action)
    return set(zip(*np.nonzero(after.board != state.board)))


def sample_actions(state, n=150, seed=0):
    actions = state.get_legal_actions()
    rng = np.random.default_rng(seed)
    return [actions[i] for i in rng.choice(len(actions), min(n, len(actions)), replace=False)]


@pytest.mark.parametrize('halo', [0, 1])
def test_masks_match_placed_cells(states, halo):
    masks = footprint_masks(halo).numpy()
    for state in states:
        for action in sample_actions(state):
            pid, oid, row, col = decode_action(action)
            cells = placed_cells(state, action)
            expect = {(r + dr, c + dc) for r, c in cells
                      for dr in range(-halo, halo + 1) for dc in range(-halo, halo + 1)}
            got = {(row + i - halo, col + j - halo)
                   for i, j in zip(*np.nonzero(masks[pid * MAX_ORIENTATIONS + oid]))}
            assert got == expect


@pytest.mark.parametrize('head', ['footprint5', 'footprint7'])
def test_logit_at_anchor_reads_covered_cells(states, head):
    """With unit weights and features marking one placement's cells plus one cell
    touching it diagonally, the logit at its anchor counts the covered cells, and
    the 7x7 head also the touching one."""
    net = BlokusNetwork(num_blocks=0, channels=1, policy_head=head)
    with torch.no_grad():
        net.policy_conv.weight.fill_(1.0)
    checked = 0
    for state in states[:4]:
        for action in sample_actions(state, n=40):
            pid, oid, row, col = decode_action(action)
            cells = placed_cells(state, action)
            ring = [(r + dr, c + dc) for r, c in cells for dr in (-1, 1) for dc in (-1, 1)
                    if 0 <= r + dr < BOARD_SIZE and 0 <= c + dc < BOARD_SIZE
                    and (r + dr, c + dc) not in cells]
            h = torch.zeros(1, 1, BOARD_SIZE, BOARD_SIZE)
            for r, c in list(cells) + ring[:1]:
                h[0, 0, r, c] = 1.0
            logit = net.footprint_logits(h)[0, pid * MAX_ORIENTATIONS + oid, row, col].item()
            assert logit == len(cells) + (len(ring[:1]) if net.halo else 0)
            checked += 1
    assert checked > 50


def test_corner_planes_cover_every_legal_placement(states):
    """Every legal move covers an open corner of the mover, and none covers a
    cell that touches the mover along an edge."""
    for state in states:
        if not state._has_played[state.current_color]:
            continue
        planes = corner_planes(torch.from_numpy(state.get_nn_state()[None]))[0, 0].numpy()
        own = state.board == state.current_color + 1
        edge = np.zeros_like(own)
        edge[1:] |= own[:-1]; edge[:-1] |= own[1:]; edge[:, 1:] |= own[:, :-1]; edge[:, :-1] |= own[:, 1:]
        for action in sample_actions(state):
            cells = placed_cells(state, action)
            assert any(planes[r, c] for r, c in cells)
            assert not any(edge[r, c] for r, c in cells)


@pytest.mark.parametrize('head,corners', [('footprint5', False), ('footprint7', True)])
def test_new_heads_run_and_round_trip(tmp_path, states, head, corners):
    net = BlokusNetwork(num_blocks=1, channels=16, policy_head=head, corner_planes=corners).eval()
    state = states[0]
    args = (torch.from_numpy(state.get_nn_state()[None]),
            torch.from_numpy(make_pieces_remaining_vector(state)[None]),
            torch.from_numpy(state.get_legal_actions_mask()[None].astype(np.float32)))
    log_policy, value = net(*args)
    assert log_policy.shape == (1, POLICY_CHANNELS * BOARD_SIZE * BOARD_SIZE)
    assert torch.isclose(log_policy.exp().sum(), torch.tensor(1.0), atol=1e-4)
    from blokus.engine.game_state import RULES_VERSION
    path = tmp_path / 'net.pt'
    torch.save({'model_state_dict': net.state_dict(), 'rules_version': RULES_VERSION,
                'config': {'network': {'num_blocks': 1, 'channels': 16,
                                       'policy_head': head, 'corner_planes': corners}}}, path)
    loaded, _ = load_network(str(path), torch.device('cpu'))
    assert torch.allclose(loaded(*args)[0], log_policy, atol=1e-5)
