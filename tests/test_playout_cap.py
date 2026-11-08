"""Playout-cap randomization: fast-search moves train the value head only."""

import os
import sys

import numpy as np
import pytest
import torch

from blokus.agents.alpha_zero import TrainingExample, self_play_game
from blokus.nn.network import BlokusNetwork

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import train  # noqa: E402


@pytest.fixture(scope='module')
def net():
    torch.manual_seed(0)
    return BlokusNetwork(num_blocks=1, channels=8).eval()


def play(net, **kwargs):
    np.random.seed(0)
    return self_play_game(network=net, num_simulations=6, top_k_actions=5, margin_scale=20.0,
                          dirichlet_alpha=0.1, **kwargs)


def test_self_play_marks_fast_moves(net):
    assert all(ex.policy_weight == 1.0 for ex in play(net))
    mixed = play(net, full_search_prob=0.25, fast_simulations=2)
    weights = [ex.policy_weight for ex in mixed]
    assert set(weights) == {0.0, 1.0}
    assert 0.1 < np.mean(weights) < 0.45          # About a quarter get the full search


def examples(n, weight):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        mask = np.zeros(67200, np.float32)
        mask[rng.choice(67200, 5, replace=False)] = 1
        out.append(TrainingExample(board_state=rng.random((5, 20, 20)).astype(np.float32),
                                   pieces_remaining=np.ones(84, np.float32), legal_mask=mask,
                                   policy_target=mask / 5, value_target=float(i % 3 - 1) / 2,
                                   policy_weight=weight(i)))
    return out


def test_buffer_keeps_weights_through_drops_and_saves():
    buf = train.ReplayBuffer(max_size=10)
    buf.add(examples(8, lambda i: float(i % 2)))
    buf.add(examples(6, lambda i: 1.0))             # Drops the 4 oldest
    expect = [0.0, 1.0, 0.0, 1.0] + [1.0] * 6
    assert buf.policy_weights_tensor().tolist() == expect
    again = train.ReplayBuffer(max_size=10)
    again.load_state_dict(buf.state_dict())
    assert again.policy_weights_tensor().tolist() == expect
    old = buf.state_dict()
    del old['policy_weights']                       # A buffer saved before this change
    legacy = train.ReplayBuffer(max_size=10)
    legacy.load_state_dict(old)
    assert legacy.policy_weights_tensor().tolist() == [1.0] * 10


def test_checkpoint_saves_weights(tmp_path, net):
    buf = train.ReplayBuffer(max_size=10)
    buf.add(examples(4, lambda i: float(i % 2)))
    mgr = train.CheckpointManager(save_dir=str(tmp_path))
    optimizer = torch.optim.Adam(net.parameters())
    mgr.save(1, net, optimizer, {'replay_buffer': buf.state_dict()}, {})
    saved = np.load(tmp_path / 'replay_buffer.npz')
    assert saved['policy_weights'].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_zero_weight_examples_do_not_train_the_policy():
    torch.manual_seed(0)
    network = BlokusNetwork(num_blocks=1, channels=8)
    buf = train.ReplayBuffer(max_size=100)
    buf.add(examples(40, lambda i: 0.0))
    before = network.policy_conv.weight.detach().clone()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)   # No weight decay
    train.train_on_examples(network, buf, optimizer, batch_size=8, epochs=1)
    assert torch.equal(network.policy_conv.weight, before)
    buf = train.ReplayBuffer(max_size=100)
    buf.add(examples(40, lambda i: 1.0))
    train.train_on_examples(network, buf, optimizer, batch_size=8, epochs=1)
    assert not torch.equal(network.policy_conv.weight, before)
