"""Compact NumPy ring replay buffer with on-the-fly 6x (D1 x C3) symmetry augmentation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
import numpy as np
import torch

from intransitive.core.constants import ACTION_SIZE, NUM_SQUARES, QUIET_LIMIT
from intransitive.env.canonical import (
    _REFLECT_ACTION_MAP,
    _REFLECT_SQ_MAP,
    encode_canonical_grids_v2_batch,
)

MAX_POLICY_ENTRIES: int = 80


@dataclass
class TrainingSample:
    """Legacy single training tuple for backward compatibility with Python fallback."""

    obs: np.ndarray          # float32 (7, 9, 9) or (11, 9, 9)
    policy: np.ndarray       # float32 (648,)
    value: float             # float in [-1, 1]


class ReplayBuffer:
    """
    Memory-efficient NumPy ring replay buffer (~405 bytes/sample).
    Stores canonical int8 (81,) grids + sparse policy + WDL targets and applies
    random 6x (D1 reflection x C3 RPS rotation) symmetries vectorized at batch time.
    """

    def __init__(self, capacity: int = 400_000, seed: Optional[int] = None) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)

        self.grids = np.zeros((self.capacity, NUM_SQUARES), dtype=np.int8)
        self.quiets = np.zeros((self.capacity,), dtype=np.uint8)
        self.pol_actions = np.zeros((self.capacity, MAX_POLICY_ENTRIES), dtype=np.uint16)
        self.pol_probs = np.zeros((self.capacity, MAX_POLICY_ENTRIES), dtype=np.float16)
        self.pol_lens = np.zeros((self.capacity,), dtype=np.uint8)
        self.wdl_targets = np.zeros((self.capacity,), dtype=np.uint8)  # 0=LOSS, 1=DRAW, 2=WIN
        self.train_policy = np.zeros((self.capacity,), dtype=np.uint8)

        self.size: int = 0
        self.head: int = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        c_grids: np.ndarray,
        quiets: np.ndarray,
        pol_acts: np.ndarray,
        pol_probs: np.ndarray,
        pol_lens: np.ndarray,
        wdl: np.ndarray,
        train_pol: np.ndarray,
    ) -> int:
        """Insert a batch of finished self-play samples into the ring buffer."""
        n = int(c_grids.shape[0])
        if n <= 0:
            return 0
        if n >= self.capacity:
            c_grids = c_grids[-self.capacity :]
            quiets = quiets[-self.capacity :]
            pol_acts = pol_acts[-self.capacity :]
            pol_probs = pol_probs[-self.capacity :]
            pol_lens = pol_lens[-self.capacity :]
            wdl = wdl[-self.capacity :]
            train_pol = train_pol[-self.capacity :]
            n = self.capacity

        end = self.head + n
        if end <= self.capacity:
            sl = slice(self.head, end)
            self.grids[sl] = c_grids
            self.quiets[sl] = quiets
            self.pol_actions[sl] = pol_acts
            self.pol_probs[sl] = pol_probs.astype(np.float16, copy=False)
            self.pol_lens[sl] = pol_lens
            self.wdl_targets[sl] = wdl
            self.train_policy[sl] = train_pol
        else:
            first = self.capacity - self.head
            rem = n - first
            sl1 = slice(self.head, self.capacity)
            sl2 = slice(0, rem)

            self.grids[sl1] = c_grids[:first]
            self.quiets[sl1] = quiets[:first]
            self.pol_actions[sl1] = pol_acts[:first]
            self.pol_probs[sl1] = pol_probs[:first].astype(np.float16, copy=False)
            self.pol_lens[sl1] = pol_lens[:first]
            self.wdl_targets[sl1] = wdl[:first]
            self.train_policy[sl1] = train_pol[:first]

            self.grids[sl2] = c_grids[first:]
            self.quiets[sl2] = quiets[first:]
            self.pol_actions[sl2] = pol_acts[first:]
            self.pol_probs[sl2] = pol_probs[first:].astype(np.float16, copy=False)
            self.pol_lens[sl2] = pol_lens[first:]
            self.wdl_targets[sl2] = wdl[first:]
            self.train_policy[sl2] = train_pol[first:]

        self.head = (self.head + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        return n

    def extend(self, samples: Iterable[TrainingSample]) -> None:
        """Backward-compatible insertion for `TrainingSample` objects."""
        items = list(samples)
        if not items:
            return
        n = len(items)
        c_grids = np.zeros((n, NUM_SQUARES), dtype=np.int8)
        quiets = np.zeros((n,), dtype=np.uint8)
        pol_acts = np.zeros((n, MAX_POLICY_ENTRIES), dtype=np.uint16)
        pol_probs = np.zeros((n, MAX_POLICY_ENTRIES), dtype=np.float32)
        pol_lens = np.zeros((n,), dtype=np.uint8)
        wdl = np.zeros((n,), dtype=np.uint8)
        train_pol = np.ones((n,), dtype=np.uint8)

        for i, it in enumerate(items):
            obs = it.obs
            grid = (
                (obs[0] > 0.5).astype(np.int8) * 1
                + (obs[1] > 0.5).astype(np.int8) * 2
                + (obs[2] > 0.5).astype(np.int8) * 3
                - (obs[3] > 0.5).astype(np.int8) * 1
                - (obs[4] > 0.5).astype(np.int8) * 2
                - (obs[5] > 0.5).astype(np.int8) * 3
            ).reshape(NUM_SQUARES)
            c_grids[i] = grid
            quiets[i] = int(np.clip(round(float(obs[6, 0, 0]) * QUIET_LIMIT), 0, 255))

            nz = np.flatnonzero(it.policy > 0)
            if len(nz) > MAX_POLICY_ENTRIES:
                top_idx = np.argpartition(it.policy[nz], -MAX_POLICY_ENTRIES)[-MAX_POLICY_ENTRIES:]
                nz = nz[top_idx]
            k = len(nz)
            pol_lens[i] = k
            if k > 0:
                pol_acts[i, :k] = nz.astype(np.uint16)
                pol_probs[i, :k] = it.policy[nz]

            if it.value > 0.33:
                wdl[i] = 2
            elif it.value < -0.33:
                wdl[i] = 0
            else:
                wdl[i] = 1

        self.add_batch(c_grids, quiets, pol_acts, pol_probs, pol_lens, wdl, train_pol)

    def sample_batch(
        self,
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
        augment_6x: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a minibatch and encode vectorized 11-channel tensors with optional 6x (D1 x C3) symmetry.
        Returns: `(obs_t (B,11,9,9), policy_t (B,648), wdl_t (B,) int64, policy_mask_t (B,) float32)`.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer")
        actual_size = min(batch_size, self.size)
        indices = self.rng.integers(0, self.size, size=actual_size)

        grids_b = self.grids[indices].copy()
        quiets_b = self.quiets[indices]
        acts_b = self.pol_actions[indices].copy()
        probs_b = self.pol_probs[indices].astype(np.float32)
        lens_b = self.pol_lens[indices]
        wdl_b = self.wdl_targets[indices].astype(np.int64)
        mask_b = self.train_policy[indices].astype(np.float32)

        if augment_6x:
            sym_ids = self.rng.integers(0, 6, size=actual_size)
            refl_mask = (sym_ids // 3) == 1
            if np.any(refl_mask):
                grids_b[refl_mask] = grids_b[refl_mask][:, _REFLECT_SQ_MAP]
                acts_b[refl_mask] = _REFLECT_ACTION_MAP[acts_b[refl_mask]]

            shifts = sym_ids % 3
            for s in (1, 2):
                sm = shifts == s
                if np.any(sm):
                    sub = grids_b[sm]
                    pos = sub > 0
                    neg = sub < 0
                    if np.any(pos):
                        sub[pos] = ((sub[pos] - 1 + s) % 3) + 1
                    if np.any(neg):
                        sub[neg] = -(((np.abs(sub[neg]) - 1 + s) % 3) + 1)
                    grids_b[sm] = sub

        obs_np = encode_canonical_grids_v2_batch(grids_b, quiets_b)

        # Scatter sparse policy entries into (B, 648) dense target
        pol_dense = np.zeros((actual_size, ACTION_SIZE), dtype=np.float32)
        col_idx = np.arange(MAX_POLICY_ENTRIES, dtype=np.int64)[None, :]
        valid = col_idx < lens_b[:, None]
        rows = np.broadcast_to(np.arange(actual_size)[:, None], (actual_size, MAX_POLICY_ENTRIES))
        pol_dense[rows[valid], acts_b[valid]] = probs_b[valid]
        row_sums = pol_dense.sum(axis=1, keepdims=True)
        nz_rows = row_sums[:, 0] > 0.0
        pol_dense[nz_rows] /= row_sums[nz_rows]

        return (
            torch.from_numpy(obs_np).to(device),
            torch.from_numpy(pol_dense).to(device),
            torch.from_numpy(wdl_b).to(device),
            torch.from_numpy(mask_b).to(device),
        )

    def save_npz(self, path: Union[str, Path]) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        n = self.size
        np.savez(
            tmp,
            grids=self.grids[:n],
            quiets=self.quiets[:n],
            pol_actions=self.pol_actions[:n],
            pol_probs=self.pol_probs[:n],
            pol_lens=self.pol_lens[:n],
            wdl_targets=self.wdl_targets[:n],
            train_policy=self.train_policy[:n],
            size=np.array([self.size], dtype=np.int64),
            head=np.array([self.head], dtype=np.int64),
        )
        tmp.replace(out)

    def load_npz(self, path: Union[str, Path]) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        with np.load(p, allow_pickle=False) as data:
            n = min(self.capacity, int(data["grids"].shape[0]))
            self.grids[:n] = data["grids"][:n]
            self.quiets[:n] = data["quiets"][:n]
            self.pol_actions[:n] = data["pol_actions"][:n]
            self.pol_probs[:n] = data["pol_probs"][:n]
            self.pol_lens[:n] = data["pol_lens"][:n]
            self.wdl_targets[:n] = data["wdl_targets"][:n]
            self.train_policy[:n] = data["train_policy"][:n]
            self.size = n
            saved_head = int(data["head"][0]) if "head" in data else n
            self.head = saved_head % self.capacity if n < self.capacity else (saved_head % self.capacity)
        return self.size
