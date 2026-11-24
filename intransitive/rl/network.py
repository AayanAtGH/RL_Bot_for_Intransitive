"""Dual-headed Policy-Value ResNets (v1 legacy & v2 SE-ResNet with Conv Policy + WDL Head)."""

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from intransitive.core.board import Board
from intransitive.core.constants import ACTION_SIZE, NUM_DIRECTIONS, NUM_SQUARES
from intransitive.env.canonical import (
    NUM_OBS_CHANNELS,
    NUM_OBS_CHANNELS_V2,
    canonical_action_mask,
    encode_board,
    encode_board_v2,
)
from intransitive.rl.device import resolve_device


class ResidualBlock(nn.Module):
    """Standard 2-layer 3x3 convolutional residual block with BatchNorm (used in v1)."""

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
        return F.relu(out + residual)


class IntransitiveNetV1(nn.Module):
    """
    Legacy v1 AlphaZero Policy-Value Network (7 channels, FC policy head, scalar tanh value).
    Preserved so existing v1 checkpoints load cleanly for play and benchmarking.
    """

    def __init__(
        self,
        in_channels: int = NUM_OBS_CHANNELS,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.arch_version = 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_res_blocks = num_res_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.res_tower = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        self.policy_conv = nn.Conv2d(hidden_channels, 16, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * NUM_SQUARES, ACTION_SIZE)

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
        probs_list, vals_list = self.predict_batch([board], device=device)
        return probs_list[0], vals_list[0]

    @torch.no_grad()
    def predict_batch(
        self, boards: Sequence[Board], device: Union[str, torch.device] = "cpu"
    ) -> Tuple[List[np.ndarray], List[float]]:
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

        return _mask_and_normalize_logits(logits_np, vals_np, boards)

    @torch.no_grad()
    def predict_batch_raw(
        self,
        obs_np: np.ndarray,
        masks_np: np.ndarray,
        device: Union[str, torch.device] = "cpu",
        use_amp: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compatibility raw-buffer inference for v1 checkpoints (slices first 7 channels)."""
        bsz = obs_np.shape[0]
        if bsz == 0:
            return np.zeros((0, ACTION_SIZE), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        dev = next(self.parameters()).device if (device is None or str(device).lower() == "auto") else resolve_device(device)
        was_training = self.training
        self.eval()

        obs_v1 = np.ascontiguousarray(obs_np[:, : self.in_channels], dtype=np.float32)
        x = torch.from_numpy(obs_v1).to(dev)
        mask_t = torch.from_numpy(masks_np.astype(bool, copy=False)).to(dev)

        logits_t, vals_t = self.forward(x)
        masked_logits = logits_t.float().masked_fill(~mask_t, -1e9)
        probs_t = F.softmax(masked_logits, dim=-1)
        legal_any = mask_t.any(dim=-1, keepdim=True)
        probs_t = torch.where(legal_any, probs_t, torch.zeros_like(probs_t))
        vals_out = vals_t.squeeze(-1).float().cpu().numpy().astype(np.float32, copy=False)
        probs_out = probs_t.cpu().numpy().astype(np.float32, copy=False)

        if was_training:
            self.train()

        return np.ascontiguousarray(probs_out), np.ascontiguousarray(vals_out)


class GlobalPoolSEBlock(nn.Module):
    """KataGo-style Global Pooling (mean + max) FiLM/Squeeze-and-Excitation conditioning."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(16, channels // reduction)
        self.fc1 = nn.Linear(channels * 2, mid)
        self.fc2 = nn.Linear(mid, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_p = x.mean(dim=(2, 3))
        max_p = x.amax(dim=(2, 3))
        pooled = torch.cat([mean_p, max_p], dim=-1)
        proj = self.fc2(F.relu(self.fc1(pooled)))
        gamma, beta = proj.chunk(2, dim=-1)
        scale = torch.sigmoid(gamma).unsqueeze(-1).unsqueeze(-1)
        bias = beta.unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias


class SEResidualBlock(nn.Module):
    """Pre/post-BN 3x3 Residual Block with zero-gamma bn2 init and optional GlobalPoolSE."""

    def __init__(self, channels: int, use_se: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = GlobalPoolSEBlock(channels) if use_se else None
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return F.relu(out + residual)


class IntransitiveNetV2(nn.Module):
    """
    v2 AlphaZero Policy-Value Network for 9x9 Intransitive:
      - Input: (B, 11, 9, 9) canonical features (6x symmetry invariant)
      - Trunk: 6 ResBlocks x 128 channels (default), GlobalPoolSE on every `se_every`-th block
      - Policy Head: Convolutional (B, 8, 9, 9) -> (B, 648) matching `from_sq * 8 + dir_idx`
      - Value Head: 3-class WDL logits (B, 3) ordered [LOSS=0, DRAW=1, WIN=2]
    """

    def __init__(
        self,
        in_channels: int = NUM_OBS_CHANNELS_V2,
        hidden_channels: int = 128,
        num_res_blocks: int = 6,
        se_every: int = 2,
    ) -> None:
        super().__init__()
        self.arch_version = 2
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_res_blocks = num_res_blocks
        self.se_every = se_every

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.res_tower = nn.ModuleList(
            [
                SEResidualBlock(
                    hidden_channels,
                    use_se=(se_every > 0 and (i + 1) % se_every == 0),
                )
                for i in range(num_res_blocks)
            ]
        )

        # Convolutional policy head: preserves spatial translation equivariance
        self.policy_conv1 = nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(32)
        self.policy_conv2 = nn.Conv2d(32, NUM_DIRECTIONS, kernel_size=1, bias=True)

        # Global + spatial WDL value head: logits ordered [LOSS=0, DRAW=1, WIN=2]
        self.value_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_spatial = nn.Conv2d(32, 4, kernel_size=1, bias=True)
        self.value_fc1 = nn.Linear(32 * 2 + 4 * NUM_SQUARES, 128)
        self.value_fc2 = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.stem(x)
        for block in self.res_tower:
            out = block(out)

        p = F.relu(self.policy_bn1(self.policy_conv1(out)))
        p = self.policy_conv2(p)  # (B, 8, 9, 9)
        policy_logits = p.permute(0, 2, 3, 1).reshape(p.size(0), ACTION_SIZE)

        v = F.relu(self.value_bn(self.value_conv(out)))
        v_mean = v.mean(dim=(2, 3))
        v_max = v.amax(dim=(2, 3))
        v_spat = self.value_spatial(v).reshape(v.size(0), -1)
        v_cat = torch.cat([v_mean, v_max, v_spat], dim=-1)
        v_hid = F.relu(self.value_fc1(v_cat))
        wdl_logits = self.value_fc2(v_hid)
        return policy_logits, wdl_logits

    @staticmethod
    def wdl_to_scalar(wdl_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(wdl_logits, dim=-1)
        return probs[:, 2] - probs[:, 0]

    @torch.no_grad()
    def predict(
        self, board: Board, device: Union[str, torch.device] = "cpu"
    ) -> Tuple[np.ndarray, float]:
        probs_list, vals_list = self.predict_batch([board], device=device)
        return probs_list[0], vals_list[0]

    @torch.no_grad()
    def predict_batch(
        self, boards: Sequence[Board], device: Union[str, torch.device] = "cpu"
    ) -> Tuple[List[np.ndarray], List[float]]:
        if not boards:
            return [], []

        was_training = self.training
        self.eval()

        encoder = encode_board_v2 if self.in_channels == NUM_OBS_CHANNELS_V2 else encode_board
        obs_batch = np.stack([encoder(b) for b in boards], axis=0)
        x = torch.from_numpy(obs_batch).to(device)
        logits_t, wdl_t = self.forward(x)
        vals_t = self.wdl_to_scalar(wdl_t)
        logits_np = logits_t.cpu().numpy()
        vals_np = vals_t.cpu().numpy()

        if was_training:
            self.train()

        return _mask_and_normalize_logits(logits_np, vals_np, boards)

    @torch.no_grad()
    def predict_batch_raw(
        self,
        obs_np: np.ndarray,
        masks_np: np.ndarray,
        device: Union[str, torch.device] = "cpu",
        use_amp: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast batched inference on pre-encoded `(B, 11, 9, 9)` float32 observations and `(B, 648)` uint8/bool masks.
        Returns `(masked_probs (B, 648) float32, scalar_vals (B,) float32)`.
        """
        bsz = obs_np.shape[0]
        if bsz == 0:
            return np.zeros((0, ACTION_SIZE), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        dev = next(self.parameters()).device if (device is None or str(device).lower() == "auto") else resolve_device(device)
        was_training = self.training
        self.eval()

        amp_ctx = nullcontext()
        if use_amp and dev.type == "cuda":
            amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)

        x = torch.from_numpy(obs_np).to(dev, non_blocking=False)
        mask_t = torch.from_numpy(masks_np.astype(bool, copy=False)).to(dev)

        with amp_ctx:
            logits_t, wdl_t = self.forward(x)

        logits_f = logits_t.float()
        masked_logits = logits_f.masked_fill(~mask_t, -1e9)
        probs_t = F.softmax(masked_logits, dim=-1)
        # Safety fallback if any row had zero legal moves
        legal_any = mask_t.any(dim=-1, keepdim=True)
        probs_t = torch.where(legal_any, probs_t, torch.zeros_like(probs_t))
        vals_t = self.wdl_to_scalar(wdl_t.float())

        probs_out = probs_t.cpu().numpy().astype(np.float32, copy=False)
        vals_out = vals_t.cpu().numpy().astype(np.float32, copy=False)

        if was_training:
            self.train()

        return np.ascontiguousarray(probs_out), np.ascontiguousarray(vals_out)


# Default alias points to IntransitiveNetV2
IntransitiveNet = IntransitiveNetV2


def _mask_and_normalize_logits(
    logits_np: np.ndarray, vals_np: np.ndarray, boards: Sequence[Board]
) -> Tuple[List[np.ndarray], List[float]]:
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


def load_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    prefer_ema: bool = True,
) -> Tuple[Union[IntransitiveNetV1, IntransitiveNetV2], Dict[str, Any]]:
    """
    Auto-detect checkpoint version (v1 legacy vs v2 SE-ResNet) and load weights cleanly.
    Returns `(model, checkpoint_dict)`.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = (
        ckpt["ema_state_dict"]
        if (prefer_ema and "ema_state_dict" in ckpt and ckpt["ema_state_dict"])
        else ckpt["model_state_dict"]
    )

    arch_version = ckpt.get("arch_version")
    if arch_version is None:
        arch_version = 1 if "policy_fc.weight" in state_dict else 2

    cfg = ckpt.get("config", {}) or {}
    if arch_version == 1:
        stem_w = state_dict["stem.0.weight"]
        in_ch = int(stem_w.shape[1])
        hid_ch = int(stem_w.shape[0])
        n_blocks = sum(
            1 for k in state_dict.keys() if k.startswith("res_tower.") and k.endswith(".conv1.weight")
        )
        net: Union[IntransitiveNetV1, IntransitiveNetV2] = IntransitiveNetV1(
            in_channels=in_ch,
            hidden_channels=hid_ch,
            num_res_blocks=n_blocks,
        )
    else:
        stem_w = state_dict["stem.0.weight"]
        in_ch = int(cfg.get("in_channels", stem_w.shape[1]))
        hid_ch = int(cfg.get("hidden_channels", stem_w.shape[0]))
        n_blocks = int(
            cfg.get(
                "num_res_blocks",
                sum(
                    1
                    for k in state_dict.keys()
                    if k.startswith("res_tower.") and k.endswith(".conv1.weight")
                ),
            )
        )
        se_every = int(cfg.get("se_every", 2))
        net = IntransitiveNetV2(
            in_channels=in_ch,
            hidden_channels=hid_ch,
            num_res_blocks=n_blocks,
            se_every=se_every,
        )

    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    return net, ckpt
