"""Load a trained BlokusNetwork from a training checkpoint."""

from typing import Any, Dict, Tuple

import torch

from blokus.nn.network import (
    BlokusNetwork, BOARD_SIZE, PIECE_REMAINING_SIZE, SCORE_VECTOR_SIZE,
)


def load_network(path: str, device: torch.device) -> Tuple[BlokusNetwork, Dict[str, Any]]:
    """Build the network a checkpoint was trained with and load its weights.

    Architecture comes from the checkpoint itself (blocks, channels, and whether
    the value head takes a score vector), so older checkpoints load correctly.

    Returns (network in eval mode, info dict with iteration and config).
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    net_cfg = cfg.get('network', {})
    state_dict = ckpt['model_state_dict']

    base_size = BOARD_SIZE * BOARD_SIZE + PIECE_REMAINING_SIZE
    value_inputs = state_dict['value_fc1.weight'].shape[1]
    score_input = value_inputs == base_size + SCORE_VECTOR_SIZE

    network = BlokusNetwork(
        num_blocks=net_cfg.get('num_blocks', 5),
        channels=net_cfg.get('channels', 128),
        score_input=score_input,
    )
    network.load_state_dict(state_dict)
    network.legacy_action_encoding = ckpt.get('action_encoding', 1) < 2
    network.to(device).eval()

    info = {
        'iteration': ckpt.get('iteration'),
        'cumulative_games': ckpt.get('cumulative_games'),
        'game_mode': cfg.get('self_play', {}).get('game_mode'),
        'score_diff_targets': cfg.get('training', {}).get('score_diff_targets', False),
        'config': cfg,
    }
    return network, info
