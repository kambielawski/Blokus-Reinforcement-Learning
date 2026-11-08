"""Replay buffers on disk: uint8 planes, compression, and the quota guard."""

import errno
import json
import os
import sys
import time

import numpy as np
import pytest
import torch

from blokus.agents.alpha_zero import TrainingExample
from blokus.nn.network import BlokusNetwork

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import train  # noqa: E402


def make_buffer(n=12):
    rng = np.random.default_rng(0)
    buf = train.ReplayBuffer(max_size=100)
    examples = []
    for i in range(n):
        mask = np.zeros(67200, np.float32)
        mask[rng.choice(67200, 6, replace=False)] = 1
        examples.append(TrainingExample(
            board_state=(rng.random((5, 20, 20)) < 0.3).astype(np.float32),
            pieces_remaining=(rng.random(84) < 0.5).astype(np.float32),
            legal_mask=mask, policy_target=mask / 6, value_target=float(rng.uniform(-1, 1)),
            policy_weight=float(i % 2)))
    buf.add(examples)
    return buf


def save(tmp_path, buf, iteration=1):
    mgr = train.CheckpointManager(save_dir=str(tmp_path))
    net = BlokusNetwork(num_blocks=1, channels=8)
    return mgr.save(iteration, net, torch.optim.Adam(net.parameters()),
                    {'replay_buffer': buf.state_dict()}, {})


def load(path):
    state = dict(np.load(path))
    state['max_size'], state['size'] = int(state['max_size']), int(state['size'])
    buf = train.ReplayBuffer(max_size=state['max_size'])
    buf.load_state_dict(state)
    return buf, state


def test_planes_saved_as_uint8_and_load_bit_identical(tmp_path):
    buf = make_buffer()
    save(tmp_path, buf)
    raw = np.load(tmp_path / 'replay_buffer.npz')
    assert raw['board_states'].dtype == np.uint8 and raw['pieces_remaining'].dtype == np.uint8
    loaded, _ = load(tmp_path / 'replay_buffer.npz')
    for got, want in zip(loaded.get_dense_tensors()[:3], buf.get_dense_tensors()[:3]):
        assert got.dtype == torch.float32 and torch.equal(got, want)
    assert torch.equal(loaded.policy_weights_tensor(), buf.policy_weights_tensor())
    for i in range(len(buf)):
        assert np.array_equal(loaded._legal_indices[i], buf._legal_indices[i])
        assert np.array_equal(loaded._policy_values[i], buf._policy_values[i])


def test_old_float32_buffers_still_load(tmp_path):
    buf = make_buffer()
    state = buf.state_dict()
    np.savez(tmp_path / 'old.npz', **{k: np.asarray(v) for k, v in state.items()})
    loaded, raw = load(tmp_path / 'old.npz')
    assert raw['board_states'].dtype == np.float32
    assert torch.equal(loaded.get_dense_tensors()[0], buf.get_dense_tensors()[0])


def test_non_binary_planes_stay_float32():
    planes = np.array([[0.0, 0.5, 1.0]], np.float32)
    assert train._binary_as_uint8(planes).dtype == np.float32
    assert train._binary_as_uint8(np.array([[0.0, 1.0]], np.float32)).dtype == np.uint8


def usage_file(tmp_path, used, age=0):
    path = tmp_path / 'usage.json'
    path.write_text(json.dumps({'bytes': used, 'time': time.time() - age}))
    return str(path)


def test_guard_skips_the_buffer_save_when_headroom_is_short(tmp_path, monkeypatch, capsys):
    run = tmp_path / 'run'
    buf = make_buffer()
    save(run, buf)
    before = (run / 'replay_buffer.npz').read_bytes()
    monkeypatch.setenv('BLOKUS_QUOTA_BYTES', str(10 ** 12))
    monkeypatch.setenv('BLOKUS_USAGE_FILE', usage_file(tmp_path, 10 ** 12 - 1000))
    buf.add(make_buffer(3).get_all())
    assert save(run, buf, iteration=2) is not None           # Checkpoint still saved
    assert 'REPLAY BUFFER NOT SAVED' in capsys.readouterr().out
    assert (run / 'replay_buffer.npz').read_bytes() == before
    assert not (run / 'replay_buffer.tmp.npz').exists()


def test_guard_ignores_a_stale_usage_reading(tmp_path, monkeypatch):
    monkeypatch.setenv('BLOKUS_QUOTA_BYTES', str(10 ** 12))
    monkeypatch.setenv('BLOKUS_USAGE_FILE', usage_file(tmp_path, 10 ** 12, age=2 * train.USAGE_MAX_AGE))
    assert train.quota_headroom() is None
    save(tmp_path / 'run', make_buffer())
    assert (tmp_path / 'run' / 'replay_buffer.npz').exists()


def test_quota_error_mid_write_does_not_crash(tmp_path, monkeypatch, capsys):
    run = tmp_path / 'run'
    buf = make_buffer()
    save(run, buf)
    before = (run / 'replay_buffer.npz').read_bytes()

    def full(path, **kwargs):
        open(path, 'wb').write(b'partial')
        raise OSError(errno.EDQUOT, 'Disk quota exceeded')

    monkeypatch.setattr(train.np, 'savez_compressed', full)
    save(run, buf, iteration=2)
    assert 'REPLAY BUFFER NOT SAVED' in capsys.readouterr().out
    assert (run / 'replay_buffer.npz').read_bytes() == before
    assert not (run / 'replay_buffer.tmp.npz').exists()

    def full_ckpt(payload, path):
        raise OSError(errno.EDQUOT, 'Disk quota exceeded')

    monkeypatch.setattr(train.torch, 'save', full_ckpt)
    assert save(run, buf, iteration=3) is None
    assert 'CHECKPOINT NOT SAVED' in capsys.readouterr().out
