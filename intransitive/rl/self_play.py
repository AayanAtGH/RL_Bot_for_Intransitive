"""Batched C++ self-play runner and legacy episode generator for AlphaZero training."""

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from intransitive.baselines.base import Player
from intransitive.core.board import Board
from intransitive.core.constants import ACTION_SIZE, BLUE, START_FEN
from intransitive.core.move import uci_to_action
from intransitive.core.rules import GameResult
from intransitive.core.tablebase import TablebaseManager
from intransitive.env.canonical import (
    NUM_OBS_CHANNELS_V2,
    apply_symmetry_6x,
    encode_board_v2,
    from_canonical_action,
    to_canonical_action,
)
from intransitive.native.native_bridge import NativeEngine
from intransitive.rl.mcts import MCTS
from intransitive.rl.network import IntransitiveNetV2
from intransitive.rl.replay_buffer import ReplayBuffer, TrainingSample


class BatchedSelfPlayRunner:
    """
    Runs `num_games` concurrent self-play games inside the C++17 `BatchMCTSEngine`
    and batches leaf neural network forward passes onto GPU in FP16/BF16/FP32.
    """

    def __init__(
        self,
        num_games: int = 128,
        full_simulations: int = 320,
        fast_simulations: int = 64,
        full_search_prob: float = 0.25,
        temp_threshold_ply: int = 24,
        c_puct: float = 1.5,
        fpu_reduction: float = 0.25,
        root_fpu_reduction: float = 0.05,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        tablebase: Optional[TablebaseManager] = None,
        seed: int = 42,
    ) -> None:
        tb_h = tablebase.handle if (tablebase and tablebase.is_loaded()) else None
        self.handle = NativeEngine.sp_create(
            tb_handle=tb_h,
            num_games=num_games,
            full_simulations=full_simulations,
            fast_simulations=fast_simulations,
            full_search_prob=full_search_prob,
            temp_threshold_ply=temp_threshold_ply,
            c_puct=c_puct,
            fpu_reduction=fpu_reduction,
            root_fpu_reduction=root_fpu_reduction,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            seed=seed,
        )
        self._obs_buf: Optional[np.ndarray] = None
        self._mask_buf: Optional[np.ndarray] = None

    def close(self) -> None:
        if getattr(self, "handle", None) is not None and NativeEngine.is_available():
            NativeEngine.sp_destroy(self.handle)
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def clear_cache(self) -> None:
        if self.handle is not None:
            NativeEngine.sp_clear_cache(self.handle)

    def get_stats(self) -> Dict[str, int]:
        assert self.handle is not None
        return NativeEngine.sp_get_stats(self.handle)

    def collect_samples(
        self,
        network: IntransitiveNetV2,
        replay_buffer: ReplayBuffer,
        min_new_samples: int = 4096,
        device: Union[str, torch.device] = "auto",
        batch_size: int = 256,
        use_amp: bool = True,
    ) -> Dict[str, float]:
        """Run concurrent games until at least `min_new_samples` are pushed to `replay_buffer`."""
        assert self.handle is not None
        if self._obs_buf is None or self._obs_buf.shape[0] < batch_size:
            self._obs_buf = np.empty((batch_size, NUM_OBS_CHANNELS_V2, 9, 9), dtype=np.float32)
            self._mask_buf = np.empty((batch_size, ACTION_SIZE), dtype=np.uint8)
        assert self._mask_buf is not None

        stats_before = NativeEngine.sp_get_stats(self.handle)
        t0 = time.perf_counter()
        added_samples = 0
        nn_evals_here = 0

        while added_samples < min_new_samples:
            b_cnt = NativeEngine.sp_collect_batch(
                self.handle, self._obs_buf, self._mask_buf, batch_size
            )
            if b_cnt > 0:
                probs_np, vals_np = network.predict_batch_raw(
                    self._obs_buf[:b_cnt],
                    self._mask_buf[:b_cnt],
                    device=device,
                    use_amp=use_amp,
                )
                NativeEngine.sp_apply_evals(self.handle, probs_np, vals_np, b_cnt)
                nn_evals_here += b_cnt

            avail = NativeEngine.sp_finished_sample_count(self.handle)
            if avail > 0:
                popped = NativeEngine.sp_pop_finished_samples(self.handle, avail)
                if popped:
                    added_samples += replay_buffer.add_batch(
                        popped["c_grids"],
                        popped["quiets"],
                        popped["pol_acts"],
                        popped["pol_probs"],
                        popped["pol_lens"],
                        popped["wdl"],
                        popped["train_pol"],
                    )
            elif b_cnt == 0:
                break

        dt = max(1e-6, time.perf_counter() - t0)
        stats_after = NativeEngine.sp_get_stats(self.handle)
        games_delta = stats_after["games_finished"] - stats_before["games_finished"]
        plies_delta = stats_after["total_plies"] - stats_before["total_plies"]

        return {
            "samples_added": float(added_samples),
            "games_delta": float(games_delta),
            "avg_plies": float(plies_delta / max(1, games_delta)),
            "nn_evals_per_sec": float(nn_evals_here / dt),
            "games_per_hour": float(games_delta * 3600.0 / dt),
            "elapsed_sec": float(dt),
            "total_games": float(stats_after["games_finished"]),
            "blue_wins": float(stats_after["blue_wins"] - stats_before["blue_wins"]),
            "red_wins": float(stats_after["red_wins"] - stats_before["red_wins"]),
            "draws": float(stats_after["draws"] - stats_before["draws"]),
            "tb_terminations": float(stats_after["tb_terminations"] - stats_before["tb_terminations"]),
            "tb_node_hits": float(stats_after["tb_node_hits"] - stats_before["tb_node_hits"]),
            "prover_hits": float(stats_after["prover_hits"] - stats_before["prover_hits"]),
            "cache_hits": float(stats_after["cache_hits"] - stats_before["cache_hits"]),
            "avg_pv_depth": float(stats_after["avg_pv_depth"]),
        }


def generate_self_play_episode(
    mcts: MCTS,
    start_fen: str = START_FEN,
    temp_threshold_ply: int = 24,
    max_plies: Optional[int] = 400,
    augment_symmetry: bool = True,
    adjudicate_draws: bool = False,
    sparring_opponent: Optional[Player] = None,
    az_side: int = BLUE,
) -> Tuple[List[TrainingSample], GameResult, int]:
    """
    Play one complete game using `mcts` (pure self-play with natural termination).
    Kept for Python fallback / unit testing compatibility.
    """
    board = Board(fen=start_fen)
    if sparring_opponent is not None:
        sparring_opponent.reset()

    trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []
    seen_counts: Dict[Tuple[bytes, int], int] = {}

    while True:
        res = board.result()
        if res is not None:
            break
        key = (board.grid.tobytes(), board.side)
        seen_counts[key] = seen_counts.get(key, 0) + 1
        if seen_counts[key] >= 3:
            res = GameResult(winner=0, reason="threefold")
            break
        if max_plies is not None and board.ply >= max_plies:
            res = GameResult(winner=0, reason="max-plies")
            break

        acting_side = board.side
        obs = encode_board_v2(board)

        if sparring_opponent is not None and acting_side != az_side:
            opp_uci = sparring_opponent.select_move(board)
            real_action = uci_to_action(opp_uci)
            c_action = to_canonical_action(real_action, acting_side)
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[c_action] = 1.0
            trajectory.append((obs, policy, acting_side))
            board.push_action(real_action, validate=False)
        else:
            temp = 1.0 if board.ply < temp_threshold_ply else 0.15
            policy, _ = mcts.search(board, temperature=temp, add_noise=True)
            trajectory.append((obs, policy, acting_side))

            prob_sum = float(np.sum(policy))
            if prob_sum > 0:
                norm_p = policy / prob_sum
                c_action = int(mcts.rng.choice(ACTION_SIZE, p=norm_p))
            else:
                c_action = int(np.argmax(policy))

            real_action = from_canonical_action(c_action, acting_side)
            board.push_action(real_action, validate=False)

    assert res is not None
    _ = adjudicate_draws
    blue_outcome = 0.0 if res.winner == 0 else (1.0 if res.winner == BLUE else -1.0)

    samples: List[TrainingSample] = []
    for obs, policy, side in trajectory:
        val = blue_outcome if side == BLUE else -blue_outcome
        if augment_symmetry:
            for sym_id in range(6):
                s_obs, s_pol = apply_symmetry_6x(obs, policy, sym_id)
                samples.append(TrainingSample(obs=s_obs, policy=s_pol, value=float(val)))
        else:
            samples.append(TrainingSample(obs=obs, policy=policy, value=float(val)))

    return samples, res, board.ply
