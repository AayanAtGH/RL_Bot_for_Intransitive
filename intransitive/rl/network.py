"""Dual-headed Policy-Value ResNet for Intransitive (RPS V6)."""

from typing import List, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from intransitive.core.board import Board
from intransitive.core.constants import ACTION_SIZE, NUM_SQUARES
from intransitive.env.canonical import NUM_OBS_CHANNELS, canonical_action_mask, encode_board


class ResidualBlock(nn.Module):
    """Standard 2-layer 3x3 convolutional residual block with BatchNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class IntransitiveNet(nn.Module):
    """
    AlphaZero Policy-Value Network for the 9x9 Intransitive board.
    Input shape: (B, 7, 9, 9)
    Outputs:
      - policy_logits: (B, 648) in canonical action coordinates
      - value: (B, 1) in [-1, 1] from current player's perspective
    """

    def __init__(
        self,
        in_channels: int = NUM_OBS_CHANNELS,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.res_tower = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        # Policy head: 1x1 conv -> flatten -> linear (648)
        self.policy_conv = nn.Conv2d(hidden_channels, 16, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * NUM_SQUARES, ACTION_SIZE)

        # Value head: 1x1 conv -> flatten -> linear(64) -> tanh(1)
        self.value_conv = nn.Conv2d(hidden_channels, 4, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * NUM_SQUARES, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.stem(x)
        for block in self.res_tower:
            out = block(out)

        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value

    @torch.no_grad()
    def predict(
        self, board: Board, device: Union[str, torch.device] = "cpu"
    ) -> Tuple[np.ndarray, float]:
        """
        Evaluate a single `Board` instance.
        Returns:
          - action_probs: numpy float32 array of shape (648,) masked to legal canonical actions
          - value: float in [-1, 1] from `board.side`'s perspective
        """
        probs_list, vals_list = self.predict_batch([board], device=device)
        return probs_list[0], vals_list[0]

    @torch.no_grad()
    def predict_batch(
        self, boards: Sequence[Board], device: Union[str, torch.device] = "cpu"
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Evaluate a batch of `Board` instances in a single forward pass on `device` (e.g. XPU).
        Returns:
          - list of (648,) masked probability arrays
          - list of scalar value estimates
        """
        if not boards:
            return [], []

        was_training = self.training
        self.eval()

        obs_batch = np.stack([encode_board(b) for b in boards], axis=0)
        x = torch.from_numpy(obs_batch).to(device)
        logits_t, vals_t = self.forward(x)
        logits_np = logits_t.cpu().numpy()
        vals_np = vals_t.squeeze(-1).cpu().numpy()

        if was_training:
            self.train()

        probs_out: List[np.ndarray] = []
        vals_out: List[float] = []

        for i, b in enumerate(boards):
            mask = canonical_action_mask(b)
            val = float(vals_np[i]) if vals_np.ndim > 0 else float(vals_np)
            vals_out.append(val)

            if not np.any(mask):
                probs_out.append(np.zeros(ACTION_SIZE, dtype=np.float32))
                continue

            row_logits = logits_np[i]
            masked_logits = np.where(mask, row_logits, -1e9)
            max_logit = np.max(masked_logits[mask])
            exp_logits = np.where(mask, np.exp(masked_logits - max_logit), 0.0)
            total = float(np.sum(exp_logits))
            if total > 0:
                probs = (exp_logits / total).astype(np.float32)
            else:
                probs = (mask.astype(np.float32) / float(np.sum(mask))).astype(np.float32)
            probs_out.append(probs)

        return probs_out, vals_out
