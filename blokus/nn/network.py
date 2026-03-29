"""AlphaZero-style neural network for Blokus.

Architecture (Silver et al. 2018):
- ResNet backbone: initial conv + N residual blocks (default 5)
- Convolutional policy head: 1x1 conv to (168, 20, 20) = 67,200 logits
- Value head: 1x1 conv -> flatten -> FC layers -> tanh scalar

Input: (batch, 5, 20, 20) from GameState.get_nn_state()
Policy output: (batch, 67200) masked log-probabilities
Value output: (batch,) scalar in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from blokus.engine.game_state import (
    BOARD_SIZE, NUM_PIECES, MAX_ORIENTATIONS, NUM_COLORS, ACTION_SPACE_SIZE,
)

# Input channels: current player, 3 opponents, empty
INPUT_CHANNELS = NUM_COLORS + 1  # 5

# Policy head output channels: 21 pieces * 8 orientations = 168
POLICY_CHANNELS = NUM_PIECES * MAX_ORIENTATIONS  # 168

# Piece-remaining vector size: 21 pieces * 4 colors = 84
PIECE_REMAINING_SIZE = NUM_PIECES * NUM_COLORS  # 84

# Score vector size: 4 colors, normalized by max possible score
SCORE_VECTOR_SIZE = NUM_COLORS  # 4
MAX_SCORE_PER_COLOR = 89  # 21 pieces = 89 total squares


class ResidualBlock(nn.Module):
    """Single residual block: two 3x3 convs with batch norm and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class BlokusNetwork(nn.Module):
    """AlphaZero-style dual-headed network for Blokus.

    Args:
        num_blocks: Number of residual blocks in the backbone (default 5).
        channels: Number of filters in the backbone (default 128).
        value_fc_size: Hidden size for the value head FC layer (default 256).
    """

    def __init__(self, num_blocks: int = 5, channels: int = 128,
                 value_fc_size: int = 256, value_dropout: float = 0.0,
                 score_input: bool = False):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels
        self.score_input = score_input

        # --- Backbone ---
        self.input_conv = nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])

        # --- Policy head (convolutional) ---
        # 1x1 conv to POLICY_CHANNELS (168), giving (batch, 168, 20, 20) = 67,200 logits
        self.policy_conv = nn.Conv2d(channels, POLICY_CHANNELS, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(POLICY_CHANNELS)
        # Project piece-remaining info to per-channel bias (84 → 168)
        # This biases each piece×orientation channel based on piece availability
        self.policy_pieces_fc = nn.Linear(PIECE_REMAINING_SIZE, POLICY_CHANNELS)

        # --- Value head ---
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        value_extra_input = SCORE_VECTOR_SIZE if score_input else 0
        self.value_fc1 = nn.Linear(
            BOARD_SIZE * BOARD_SIZE + PIECE_REMAINING_SIZE + value_extra_input,
            value_fc_size,
        )
        self.value_dropout = nn.Dropout(p=value_dropout) if value_dropout > 0 else None
        self.value_fc2 = nn.Linear(value_fc_size, 1)

    def forward(self, board_state: torch.Tensor,
                pieces_remaining: torch.Tensor,
                legal_actions_mask: torch.Tensor,
                score_vector: torch.Tensor = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            board_state: (batch, 5, 20, 20) float32 from get_nn_state().
            pieces_remaining: (batch, 84) binary float32 — which pieces each
                player still has (21 pieces * 4 colors, in turn order starting
                from current player).
            legal_actions_mask: (batch, 67200) binary float32 — 1 for legal actions.
            score_vector: (batch, 4) float32 — normalized scores per color in
                turn order. Only used when score_input=True.

        Returns:
            policy: (batch, 67200) log-probabilities (masked softmax).
            value: (batch,) scalar predictions in [-1, 1].
        """
        # Backbone
        x = F.relu(self.input_bn(self.input_conv(board_state)))
        for block in self.res_blocks:
            x = block(x)

        # --- Policy head ---
        p = self.policy_bn(self.policy_conv(x))  # (batch, 168, 20, 20)
        # Add piece-remaining bias: project 84 → 168, broadcast over spatial dims
        piece_bias = self.policy_pieces_fc(pieces_remaining)  # (batch, 168)
        p = p + piece_bias.unsqueeze(-1).unsqueeze(-1)  # broadcast to (batch, 168, 20, 20)
        p = F.relu(p)
        p = p.view(p.size(0), -1)  # (batch, 67200)

        # Mask illegal actions with large negative value, then log-softmax
        p = p.masked_fill(legal_actions_mask == 0, -1e8)
        log_policy = F.log_softmax(p, dim=1)

        # --- Value head ---
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # (batch, 400)
        value_inputs = [v, pieces_remaining]
        if self.score_input and score_vector is not None:
            value_inputs.append(score_vector)
        v = torch.cat(value_inputs, dim=1)
        v = F.relu(self.value_fc1(v))
        if self.value_dropout is not None:
            v = self.value_dropout(v)
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)  # (batch,)

        return log_policy, v


def make_pieces_remaining_vector(state) -> np.ndarray:
    """Build the 84-dim piece-remaining vector from a GameState.

    Layout: 4 colors in turn order (starting from current player),
    each with 21 binary values indicating if that piece is still available.

    Returns: (84,) float32 numpy array.
    """
    vec = np.zeros(PIECE_REMAINING_SIZE, dtype=np.float32)
    current = state.current_color
    for i in range(NUM_COLORS):
        color = (current + i) % NUM_COLORS
        offset = i * NUM_PIECES
        for pid in state.pieces_remaining[color]:
            vec[offset + pid] = 1.0
    return vec


def make_score_vector(state) -> np.ndarray:
    """Build a 4-dim normalized score vector from a GameState.

    Layout: 4 colors in turn order (starting from current player),
    each score normalized by max possible score (89).

    Returns: (4,) float32 numpy array.
    """
    scores = state.get_scores()
    vec = np.zeros(SCORE_VECTOR_SIZE, dtype=np.float32)
    current = state.current_color
    for i in range(NUM_COLORS):
        color = (current + i) % NUM_COLORS
        vec[i] = scores.get(color, 0) / MAX_SCORE_PER_COLOR
    return vec
