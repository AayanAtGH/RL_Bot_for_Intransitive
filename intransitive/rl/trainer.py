"""Tabula-rasa AlphaZero training loop with C++ batched self-play, EMA, WDL loss, and gating."""

import argparse
from contextlib import nullcontext
import copy
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from intransitive.arena.evaluator import MatchStats
from intransitive.baselines.base import Player
from intransitive.baselines.minimax_player import MinimaxPlayer
from intransitive.core.board import Board
from intransitive.core.constants import BLUE, RED, START_FEN
from intransitive.core.rules import GameResult
from intransitive.core.tablebase import TablebaseManager
from intransitive.env.canonical import NUM_OBS_CHANNELS_V2
from intransitive.rl.device import (
    get_device_summary,
    resolve_device,
    synchronize_device,
)
from intransitive.rl.mcts import AlphaZeroPlayer
from intransitive.rl.network import IntransitiveNetV2, load_checkpoint
from intransitive.rl.replay_buffer import ReplayBuffer
from intransitive.rl.self_play import BatchedSelfPlayRunner


@dataclass
class TrainConfig:
    """Hyperparameters for v2 tabula-rasa AlphaZero training."""

    iterations: int = 10
    total_games: Optional[int] = None
    duration_minutes: Optional[float] = None
    continuous: bool = False

    # Network (6 x 128 SE-ResNet default)
    hidden_channels: int = 128
    num_res_blocks: int = 6
    se_every: int = 2

    # Batched C++ Self-Play + Playout Cap Randomization
    num_parallel_games: int = 128
    selfplay_batch_size: int = 256
    full_simulations: int = 320
    fast_simulations: int = 64
    full_search_prob: float = 0.25
    temp_threshold_ply: int = 24
    samples_per_iter: int = 4096

    # Optimization
    batch_size: int = 256
    replay_ratio: float = 1.5
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 1e-4
    ema_decay: float = 0.995
    buffer_capacity: int = 400_000
    use_amp: bool = True

    # Tablebase & Evaluation Gating
    tb_dir: Optional[str] = "data/tablebase"
    eval_every_games: int = 2000
    gating_games: int = 40
    gating_simulations: int = 80
    gating_threshold: float = 0.55

    # Persistence
    device: str = "auto"
    checkpoint_path: Optional[str] = "checkpoints/alphazero_v2_latest.pt"
    best_checkpoint_path: Optional[str] = "checkpoints/alphazero_v2_best.pt"
    buffer_path: Optional[str] = None
    save_every_minutes: float = 15.0
    seed: int = 42


def _diverse_play_series(
    player_a: Player,
    player_b: Player,
    num_games: int = 40,
    max_plies: int = 280,
    seed: int = 42,
) -> MatchStats:
    """Play `num_games` alternating colors with randomized 0..2 opening plies for diversity."""
    rng = random.Random(seed)
    stats = MatchStats(player_a=player_a.name, player_b=player_b.name)

    for i in range(num_games):
        # Generate shared randomized opening position for every pair of games (i//2)
        if i % 2 == 0:
            board = Board(START_FEN)
            rand_plies = rng.randint(0, 2)
            for _ in range(rand_plies):
                legal = board.legal_moves()
                if not legal:
                    break
                board.push_uci(rng.choice(legal), validate=False)
                if board.is_terminal():
                    board = Board(START_FEN)
                    break
            pair_fen = board.to_fen()
            blue_p, red_p = player_a, player_b
            a_is_blue = True
        else:
            blue_p, red_p = player_b, player_a
            a_is_blue = False

        b = Board(fen=pair_fen)
        blue_p.reset()
        red_p.reset()
        seen: Dict[Tuple[bytes, int], int] = {}

        res: Optional[GameResult] = None
        while True:
            res = b.result()
            if res is not None:
                break
            k = (b.grid.tobytes(), b.side)
            seen[k] = seen.get(k, 0) + 1
            if seen[k] >= 3:
                res = GameResult(winner=0, reason="threefold")
                break
            if b.ply >= max_plies:
                res = GameResult(winner=0, reason="max-plies")
                break
            active = blue_p if b.side == BLUE else red_p
            mv = active.select_move(b)
            b.push_uci(mv, validate=True)

        assert res is not None
        stats.total_plies += b.ply
        if res.winner == 0:
            stats.draws += 1
        elif (res.winner == BLUE and a_is_blue) or (res.winner == RED and not a_is_blue):
            stats.a_wins += 1
        else:
            stats.b_wins += 1
    return stats


class AlphaZeroTrainer:
    """Manages batched C++ self-play, EMA optimization, and arena gating."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self.device = resolve_device(config.device)

        self.tablebase: Optional[TablebaseManager] = None
        if config.tb_dir and Path(config.tb_dir).exists():
            tb = TablebaseManager(config.tb_dir)
            if tb.is_loaded():
                self.tablebase = tb

        self.network = IntransitiveNetV2(
            in_channels=NUM_OBS_CHANNELS_V2,
            hidden_channels=config.hidden_channels,
            num_res_blocks=config.num_res_blocks,
            se_every=config.se_every,
        ).to(self.device)

        self.start_iter: int = 0
        self.total_games_done: int = 0
        self.global_grad_steps: int = 0

        if config.checkpoint_path and Path(config.checkpoint_path).exists():
            try:
                loaded_net, ckpt = load_checkpoint(config.checkpoint_path, device="cpu", prefer_ema=False)
                if isinstance(loaded_net, IntransitiveNetV2) and (
                    loaded_net.hidden_channels == config.hidden_channels
                    and loaded_net.num_res_blocks == config.num_res_blocks
                ):
                    self.network.load_state_dict(ckpt["model_state_dict"])
                    self.start_iter = int(ckpt.get("iteration", ckpt.get("epoch", 0)))
                    self.total_games_done = int(ckpt.get("total_games", 0))
                    self.global_grad_steps = int(ckpt.get("grad_steps", 0))
                    print(
                        f"Resumed v2 checkpoint from {config.checkpoint_path} "
                        f"(iter={self.start_iter}, games={self.total_games_done})"
                    )
                else:
                    print(
                        f"Existing {config.checkpoint_path} has incompatible arch; "
                        "starting fresh v2 weights."
                    )
            except Exception as exc:
                print(f"Starting with fresh v2 weights ({exc})")

        self.ema_network = copy.deepcopy(self.network).to(self.device)
        if config.checkpoint_path and Path(config.checkpoint_path).exists():
            try:
                ckpt_ema = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
                if (
                    ckpt_ema.get("arch_version") == 2
                    and "ema_state_dict" in ckpt_ema
                    and ckpt_ema["ema_state_dict"]
                ):
                    self.ema_network.load_state_dict(ckpt_ema["ema_state_dict"])
            except Exception:
                pass
        self.ema_network.eval()
        for p in self.ema_network.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.use_cuda_amp = bool(config.use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_cuda_amp)

        self.buffer = ReplayBuffer(capacity=config.buffer_capacity, seed=config.seed)
        if config.buffer_path and Path(config.buffer_path).exists():
            loaded_n = self.buffer.load_npz(config.buffer_path)
            print(f"Loaded replay buffer from {config.buffer_path} ({loaded_n:,} samples)")

        self.sp_runner = BatchedSelfPlayRunner(
            num_games=config.num_parallel_games,
            full_simulations=config.full_simulations,
            fast_simulations=config.fast_simulations,
            full_search_prob=config.full_search_prob,
            temp_threshold_ply=config.temp_threshold_ply,
            tablebase=self.tablebase,
            seed=config.seed,
        )

    def _update_lr(self, progress_fraction: float) -> float:
        if self.global_grad_steps < self.config.warmup_steps:
            frac = float(self.global_grad_steps + 1) / float(max(1, self.config.warmup_steps))
            curr_lr = self.config.min_lr + (self.config.lr - self.config.min_lr) * frac
        else:
            prog = max(0.0, min(1.0, progress_fraction))
            cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
            curr_lr = self.config.min_lr + (self.config.lr - self.config.min_lr) * cosine
        for group in self.optimizer.param_groups:
            group["lr"] = curr_lr
        return curr_lr

    def _update_ema(self) -> None:
        decay = self.config.ema_decay
        with torch.no_grad():
            for p_ema, p_net in zip(self.ema_network.parameters(), self.network.parameters()):
                p_ema.mul_(decay).add_(p_net.data, alpha=1.0 - decay)
            for b_ema, b_net in zip(self.ema_network.buffers(), self.network.buffers()):
                b_ema.copy_(b_net)

    def train_step(self) -> Dict[str, float]:
        self.network.train()
        obs_b, target_pi, target_wdl, pol_mask = self.buffer.sample_batch(
            self.config.batch_size, device=self.device, augment_6x=True
        )

        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_cuda_amp
            else nullcontext()
        )
        with amp_ctx:
            logits_pi, logits_wdl = self.network(obs_b)

        logits_pi_f = logits_pi.float()
        logits_wdl_f = logits_wdl.float()

        log_probs = F.log_softmax(logits_pi_f, dim=-1)
        per_sample_pol = -torch.sum(target_pi * log_probs, dim=-1)
        pol_weight_sum = pol_mask.sum().clamp(min=1.0)
        policy_loss = (per_sample_pol * pol_mask).sum() / pol_weight_sum

        wdl_loss = F.cross_entropy(logits_wdl_f, target_wdl)
        total_loss = policy_loss + wdl_loss

        self.optimizer.zero_grad(set_to_none=True)
        if self.use_cuda_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=2.0)
            self.optimizer.step()

        self._update_ema()
        self.global_grad_steps += 1

        with torch.no_grad():
            val_acc = (logits_wdl_f.argmax(dim=-1) == target_wdl).float().mean()

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "wdl_loss": float(wdl_loss.item()),
            "val_acc": float(val_acc.item()),
        }

    def save_checkpoint(self, path: str | Path, iteration: int, total_mins: float, is_best: bool = False) -> None:
        ckpt_file = Path(path)
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = ckpt_file.with_suffix(".tmp.pt")
        payload = {
            "arch_version": 2,
            "iteration": iteration,
            "epoch": iteration,
            "total_games": self.total_games_done,
            "grad_steps": self.global_grad_steps,
            "total_minutes": total_mins,
            "is_best": is_best,
            "model_state_dict": self.network.state_dict(),
            "ema_state_dict": self.ema_network.state_dict(),
            "config": {
                "in_channels": NUM_OBS_CHANNELS_V2,
                "hidden_channels": self.config.hidden_channels,
                "num_res_blocks": self.config.num_res_blocks,
                "se_every": self.config.se_every,
            },
        }
        torch.save(payload, tmp_file)
        tmp_file.replace(ckpt_file)

    def _run_gating_eval(self, iteration: int, total_mins: float) -> None:
        cand_player = AlphaZeroPlayer(
            network=self.ema_network,
            num_simulations=self.config.gating_simulations,
            temperature=0.0,
            device=self.device,
            tablebase=self.tablebase,
            min_pv_depth=6,
            name=f"Cand(it={iteration})",
        )

        # Benchmark against Minimax(d=2)
        mm2 = MinimaxPlayer(depth=2, seed=self.config.seed + 500)
        vs_mm = _diverse_play_series(cand_player, mm2, num_games=10, seed=iteration + 100)
        print(
            f"  >>> [Arena @ Iter {iteration:04d}] vs Minimax(d=2): "
            f"+{vs_mm.a_wins}-{vs_mm.b_wins}={vs_mm.draws} ({vs_mm.a_win_rate * 100:.1f}%)"
        )

        best_path = self.config.best_checkpoint_path
        if not best_path:
            return
        best_file = Path(best_path)
        if not best_file.exists():
            self.save_checkpoint(best_file, iteration, total_mins, is_best=True)
            print(f"  >>> Initialized best checkpoint at {best_file}")
            return

        best_net, best_ckpt = load_checkpoint(best_file, device=self.device, prefer_ema=True)
        best_player = AlphaZeroPlayer(
            network=best_net,
            num_simulations=self.config.gating_simulations,
            temperature=0.0,
            device=self.device,
            tablebase=self.tablebase,
            min_pv_depth=6,
            name=f"Best(v{best_ckpt.get('arch_version', 1)})",
        )
        vs_best = _diverse_play_series(
            cand_player,
            best_player,
            num_games=self.config.gating_games,
            seed=iteration + 2000,
        )
        print(
            f"  >>> [Gating vs {best_player.name}] "
            f"+{vs_best.a_wins}-{vs_best.b_wins}={vs_best.draws} "
            f"(Score={vs_best.a_win_rate * 100:.1f}%, thresh={self.config.gating_threshold * 100:.0f}%)"
        )
        if vs_best.a_win_rate >= self.config.gating_threshold:
            self.save_checkpoint(best_file, iteration, total_mins, is_best=True)
            print(f"  >>> PROMOTED new best model to {best_file}!")

    def run_training(self) -> List[Dict[str, float]]:
        summary = get_device_summary(self.device)
        tb_desc = f"Loaded (max {self.tablebase.max_pieces}p)" if self.tablebase else "None"
        print(
            f"=== AlphaZero v2 Trainer (Tabula-Rasa RL) ===\n"
            f"  Device       : {summary['active_device']} | AMP={self.use_cuda_amp} | TB={tb_desc}\n"
            f"  Network      : {self.config.num_res_blocks}x{self.config.hidden_channels} SE-ResNet "
            f"({sum(p.numel() for p in self.network.parameters()):,} params)\n"
            f"  Self-Play    : {self.config.num_parallel_games} parallel games | "
            f"PlayoutCap {self.config.full_simulations}/{self.config.fast_simulations} (p={self.config.full_search_prob})"
        )
        sys.stdout.flush()

        history: List[Dict[str, float]] = []
        start_wall = time.perf_counter()
        last_save_wall = start_wall
        next_eval_games = self.total_games_done + self.config.eval_every_games
        session_iter = 0

        try:
            while True:
                session_iter += 1
                iteration = self.start_iter + session_iter
                elapsed_mins = (time.perf_counter() - start_wall) / 60.0

                if not self.config.continuous:
                    if self.config.duration_minutes is not None:
                        if session_iter > 1 and elapsed_mins >= self.config.duration_minutes:
                            print(f"\nReached duration target ({elapsed_mins:.2f}m).")
                            break
                        prog_frac = elapsed_mins / max(1e-3, self.config.duration_minutes)
                    elif self.config.total_games is not None:
                        if session_iter > 1 and self.total_games_done >= self.config.total_games:
                            print(f"\nReached total_games target ({self.total_games_done}).")
                            break
                        prog_frac = float(self.total_games_done) / float(max(1, self.config.total_games))
                    else:
                        if session_iter > self.config.iterations:
                            break
                        prog_frac = float(session_iter - 1) / float(max(1, self.config.iterations))
                else:
                    prog_frac = 0.0

                curr_lr = self._update_lr(prog_frac)

                # 1. Batched C++ Self-Play using smooth EMA weights
                self.sp_runner.clear_cache()
                sp_stats = self.sp_runner.collect_samples(
                    network=self.ema_network,
                    replay_buffer=self.buffer,
                    min_new_samples=self.config.samples_per_iter,
                    device=self.device,
                    batch_size=self.config.selfplay_batch_size,
                    use_amp=self.use_cuda_amp,
                )
                self.total_games_done += int(sp_stats["games_delta"])

                # 2. Gradient updates proportional to replay_ratio
                steps = max(
                    4,
                    int(round(sp_stats["samples_added"] * self.config.replay_ratio / max(1, self.config.batch_size))),
                )
                losses = {"total_loss": 0.0, "policy_loss": 0.0, "wdl_loss": 0.0, "val_acc": 0.0}
                synchronize_device(self.device)
                t_train0 = time.perf_counter()
                for _ in range(steps):
                    m = self.train_step()
                    for k in losses:
                        losses[k] += m[k] / steps
                synchronize_device(self.device)
                t_train = time.perf_counter() - t_train0

                total_mins = (time.perf_counter() - start_wall) / 60.0
                row = {
                    "iteration": float(iteration),
                    "total_games": float(self.total_games_done),
                    "total_minutes": total_mins,
                    "buffer_size": float(len(self.buffer)),
                    "lr": curr_lr,
                    "train_sec": t_train,
                    **sp_stats,
                    **losses,
                }
                history.append(row)

                print(
                    f"[Iter {iteration:04d} | {total_mins:6.2f}m | Games={self.total_games_done:7d}] "
                    f"G/hr={sp_stats['games_per_hour']:6.0f} | NN/s={sp_stats['nn_evals_per_sec']:7.0f} | "
                    f"PVd={sp_stats['avg_pv_depth']:2.0f} | TBterm={int(sp_stats['tb_terminations']):3d} | "
                    f"Plies={sp_stats['avg_plies']:5.1f} | Buf={len(self.buffer):6d} | "
                    f"Pol={losses['policy_loss']:.3f} WDL={losses['wdl_loss']:.3f} Acc={losses['val_acc']*100:4.1f}% | "
                    f"LR={curr_lr:.5f}"
                )
                sys.stdout.flush()

                # Save latest checkpoint every iter (fast ~10 MB) and buffer periodically
                if self.config.checkpoint_path:
                    self.save_checkpoint(self.config.checkpoint_path, iteration, total_mins, is_best=False)

                now_wall = time.perf_counter()
                if self.config.buffer_path and (now_wall - last_save_wall) / 60.0 >= self.config.save_every_minutes:
                    self.buffer.save_npz(self.config.buffer_path)
                    last_save_wall = now_wall
                    print(f"  >>> Saved replay buffer snapshot to {self.config.buffer_path}")

                # Periodic Arena Evaluation & Gating
                if self.config.eval_every_games > 0 and self.total_games_done >= next_eval_games:
                    self._run_gating_eval(iteration, total_mins)
                    next_eval_games = self.total_games_done + self.config.eval_every_games
        finally:
            if self.config.buffer_path and len(self.buffer) > 0:
                self.buffer.save_npz(self.config.buffer_path)
            self.sp_runner.close()

        return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaZero v2 RL Bot for Intransitive")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--total-games", type=int, default=None)
    parser.add_argument("--duration-minutes", type=float, default=None)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--blocks", type=int, default=6, help="ResNet blocks (default: 6)")
    parser.add_argument("--channels", type=int, default=128, help="ResNet channels (default: 128)")
    parser.add_argument("--parallel-games", type=int, default=128)
    parser.add_argument("--sp-batch", type=int, default=256)
    parser.add_argument("--full-sims", type=int, default=320)
    parser.add_argument("--fast-sims", type=int, default=64)
    parser.add_argument("--samples-per-iter", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--replay-ratio", type=float, default=1.5)
    parser.add_argument("--buffer-capacity", type=int, default=400_000)
    parser.add_argument("--tb-dir", type=str, default="data/tablebase")
    parser.add_argument("--eval-every-games", type=int, default=2000)
    parser.add_argument("--gating-games", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/alphazero_v2_latest.pt")
    parser.add_argument("--best-checkpoint", type=str, default="checkpoints/alphazero_v2_best.pt")
    parser.add_argument("--buffer-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        iterations=args.iterations,
        total_games=args.total_games,
        duration_minutes=args.duration_minutes,
        continuous=args.continuous,
        hidden_channels=args.channels,
        num_res_blocks=args.blocks,
        num_parallel_games=args.parallel_games,
        selfplay_batch_size=args.sp_batch,
        full_simulations=args.full_sims,
        fast_simulations=args.fast_sims,
        samples_per_iter=args.samples_per_iter,
        batch_size=args.batch_size,
        lr=args.lr,
        replay_ratio=args.replay_ratio,
        buffer_capacity=args.buffer_capacity,
        tb_dir=args.tb_dir if args.tb_dir else None,
        eval_every_games=args.eval_every_games,
        gating_games=args.gating_games,
        device=args.device,
        use_amp=not args.no_amp,
        checkpoint_path=args.checkpoint,
        best_checkpoint_path=args.best_checkpoint,
        buffer_path=args.buffer_path,
        seed=args.seed,
    )
    trainer = AlphaZeroTrainer(cfg)
    trainer.run_training()


if __name__ == "__main__":
    main()
