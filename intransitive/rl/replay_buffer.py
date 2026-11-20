"""Replay buffer for storing AlphaZero self-play training samples."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch


@dataclass
class TrainingSample:
    """Single training tuple: (canonical_obs, mcts_policy_probs, outcome_value)."""

    obs: np.ndarray          # float32 (7, 9, 9)
    policy: np.ndarray       # float32 (648,)
    value: float             # float in [-1, 1] from acting player's perspective


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer for AlphaZero training samples."""

    def __init__(self, capacity: int = 25_000, seed: Optional[int] = None) -> None:
        self.buffer: Deque[TrainingSample] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def extend(self, samples: Iterable[TrainingSample]) -> None:
        self.buffer.extend(samples)

    def sample_batch(
        self, batch_size: int, device: Union[str, torch.device] = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (obs_batch, policy_batch, value_batch) tensors on `device`."""
        n = len(self.buffer)
        if n == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer")
        actual_size = min(batch_size, n)
        indices = self.rng.choice(n, size=actual_size, replace=False)
        items: List[TrainingSample] = [self.buffer[int(i)] for i in indices]

        obs_arr = np.stack([it.obs for it in items], axis=0).astype(np.float32)
        pol_arr = np.stack([it.policy for it in items], axis=0).astype(np.float32)
        val_arr = np.array([[it.value] for it in items], dtype=np.float32)

        return (
            torch.from_numpy(obs_arr).to(device),
            torch.from_numpy(pol_arr).to(device),
            torch.from_numpy(val_arr).to(device),
        )
