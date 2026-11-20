"""AlphaZero training loop with Intel Extension for PyTorch (IPEX / XPU) and curriculum sparring."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F

from intransitive.arena.evaluator import play_series
from intransitive.baselines.greedy_player import GreedyPlayer
from intransitive.baselines.minimax_player import MinimaxPlayer
from intransitive.baselines.random_player import RandomPlayer
from intransitive.core.constants import BLUE, RED
from intransitive.rl.device import (
    get_device_summary,
    optimize_with_ipex,
    resolve_device,
    synchronize_device,
)
from intransitive.rl.mcts import AlphaZeroPlayer, MCTS
from intransitive.rl.network import IntransitiveNet
from intransitive.rl.replay_buffer import ReplayBuffer
from intransitive.rl.self_play import generate_self_play_episode


@dataclass
class TrainConfig:
    """Hyperparameters for AlphaZero self-play and network optimization."""

    epochs: int = 2
    duration_minutes: Optional[float] = None
    continuous: bool = False
    episodes_per_epoch: int = 4
    num_simulations: int = 40
    mcts_batch_size: Optional[int] = None
    max_plies_per_game: int = 120
    batch_size: int = 64
    train_steps_per_epoch: int = 24
    lr: float = 4e-4
    min_lr: float = 1e-4
    weight_decay: float = 1e-4
    buffer_capacity: int = 60_000
    hidden_channels: int = 64
    num_res_blocks: int = 4
    device: str = "auto"
    use_ipex_optimize: bool = False
    use_curriculum_sparring: bool = True
    eval_every_epochs: int = 15
    checkpoint_path: Optional[str] = "checkpoints/alphazero_latest.pt"
    best_checkpoint_path: Optional[str] = "checkpoints/alphazero_best.pt"
    seed: int = 42


class AlphaZeroTrainer:
    """Manages AlphaZero self-play generation and neural network training."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self.device = resolve_device(config.device)

        self.network: torch.nn.Module = IntransitiveNet(
            hidden_channels=config.hidden_channels,
            num_res_blocks=config.num_res_blocks,
        ).to(self.device)

        self.start_epoch: int = 0
        self.best_eval_score: float = -1.0

        # Resume from existing checkpoint if available when doing a long or continuous run
        if config.checkpoint_path and Path(config.checkpoint_path).exists() and (config.continuous or (config.duration_minutes or 0) > 0.1):
            try:
                ckpt = torch.load(config.checkpoint_path, map_location=self.device, weights_only=False)
                self.network.load_state_dict(ckpt["model_state_dict"])
                self.start_epoch = int(ckpt.get("epoch", 0))
                print(f"Resumed weights from {config.checkpoint_path} (completed epoch {self.start_epoch})")
            except Exception as exc:
                print(f"Starting with fresh weights ({exc})")

        if config.best_checkpoint_path and Path(config.best_checkpoint_path).exists() and (config.continuous or (config.duration_minutes or 0) > 0.1):
            try:
                b_ckpt = torch.load(config.best_checkpoint_path, map_location="cpu", weights_only=False)
                self.best_eval_score = float(b_ckpt.get("eval_score", -1.0))
            except Exception:
                pass

        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        if config.use_ipex_optimize:
            opt_net, opt_optim = optimize_with_ipex(self.network, self.optimizer)
            self.network = opt_net
            if opt_optim is not None:
                self.optimizer = opt_optim

        self.buffer = ReplayBuffer(capacity=config.buffer_capacity, seed=config.seed)
        self.mcts = MCTS(
            network=self.network,  # type: ignore[arg-type]
            num_simulations=config.num_simulations,
            device=self.device,
            batch_size=config.mcts_batch_size,
            seed=config.seed,
        )
        self.greedy_sparring = GreedyPlayer(seed=config.seed + 100)
        self.minimax_sparring = MinimaxPlayer(depth=2, seed=config.seed + 200)
        self.random_eval = RandomPlayer(seed=config.seed + 300)

    def _update_lr(self, progress_fraction: float) -> float:
        """Cosine learning rate schedule between `config.lr` and `config.min_lr`."""
        prog = max(0.0, min(1.0, progress_fraction))
        cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
        curr_lr = self.config.min_lr + (self.config.lr - self.config.min_lr) * cosine
        for group in self.optimizer.param_groups:
            group["lr"] = curr_lr
        return curr_lr

    def train_step(self) -> Dict[str, float]:
        """Sample a batch from the replay buffer and perform one gradient update."""
        self.network.train()
        obs_b, target_pi, target_v = self.buffer.sample_batch(
            self.config.batch_size, device=self.device
        )

        logits_pi, pred_v = self.network(obs_b)
        log_probs = F.log_softmax(logits_pi, dim=-1)

        policy_loss = -torch.sum(target_pi * log_probs, dim=-1).mean()
        value_loss = F.mse_loss(pred_v, target_v)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=2.0)
        self.optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def run_training(self) -> List[Dict[str, float]]:
        """Execute training for `config.duration_minutes` (if set) or `config.epochs`."""
        summary = get_device_summary(self.device)
        if self.config.continuous:
            target_desc = "Continuous (runs until stopped)"
        elif self.config.duration_minutes:
            target_desc = f"{self.config.duration_minutes:.1f} minutes"
        else:
            target_desc = f"{self.config.epochs} epochs"

        print(
            f"=== AlphaZero Trainer Initialized (Target: {target_desc}) ===\n"
            f"  Active Device : {summary['active_device']} "
            f"({summary['xpu_device_name'] or 'CPU/Standard'})\n"
            f"  PyTorch       : {summary['torch_version']} | "
            f"IPEX Installed: {summary['ipex_installed']} ({summary['ipex_version']})\n"
            f"  MCTS BatchSz  : {self.mcts.batch_size} | Simulations/Move: {self.config.num_simulations} | "
            f"Episodes/Epoch: {self.config.episodes_per_epoch} | StartEpoch: {self.start_epoch}"
        )
        sys.stdout.flush()

        history: List[Dict[str, float]] = []
        start_wall_time = time.perf_counter()
        session_step = 0

        while True:
            session_step += 1
            epoch = self.start_epoch + session_step

            if not self.config.continuous:
                if self.config.duration_minutes is not None:
                    elapsed_mins = (time.perf_counter() - start_wall_time) / 60.0
                    if session_step > 1 and elapsed_mins >= self.config.duration_minutes:
                        print(
                            f"\nReached target training duration ({elapsed_mins:.2f} / "
                            f"{self.config.duration_minutes:.1f} mins) at Epoch {epoch - 1}."
                        )
                        sys.stdout.flush()
                        break
                    prog_frac = elapsed_mins / max(1e-3, self.config.duration_minutes)
                else:
                    if session_step > self.config.epochs:
                        break
                    prog_frac = (session_step - 1) / max(1, self.config.epochs)
                curr_lr = self._update_lr(prog_frac)
            else:
                curr_lr = self.config.lr

            synchronize_device(self.device)
            t0 = time.perf_counter()

            blue_wins = 0
            red_wins = 0
            draws = 0
            total_plies = 0

            for ep_idx in range(self.config.episodes_per_epoch):
                sparring = None
                az_side = BLUE if (epoch + ep_idx) % 2 == 0 else RED
                if self.config.use_curriculum_sparring and self.config.episodes_per_epoch >= 3:
                    if ep_idx == self.config.episodes_per_epoch - 2:
                        sparring = self.greedy_sparring
                    elif ep_idx == self.config.episodes_per_epoch - 1:
                        sparring = self.minimax_sparring

                samples, res, plies = generate_self_play_episode(
                    self.mcts,
                    max_plies=self.config.max_plies_per_game,
                    augment_symmetry=True,
                    adjudicate_draws=True,
                    sparring_opponent=sparring,
                    az_side=az_side,
                )
                self.buffer.extend(samples)
                total_plies += plies
                if res.winner > 0:
                    blue_wins += 1
                elif res.winner < 0:
                    red_wins += 1
                else:
                    draws += 1

            losses = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
            for _ in range(self.config.train_steps_per_epoch):
                step_metrics = self.train_step()
                for k in losses:
                    losses[k] += step_metrics[k] / self.config.train_steps_per_epoch

            synchronize_device(self.device)
            elapsed = time.perf_counter() - t0
            total_mins = (time.perf_counter() - start_wall_time) / 60.0

            epoch_stats = {
                "epoch": float(epoch),
                "elapsed_sec": elapsed,
                "total_minutes": total_mins,
                "buffer_size": float(len(self.buffer)),
                "avg_plies": total_plies / max(1, self.config.episodes_per_epoch),
                "blue_wins": float(blue_wins),
                "red_wins": float(red_wins),
                "draws": float(draws),
                "lr": curr_lr,
                **losses,
            }
            history.append(epoch_stats)

            print(
                f"[Epoch {epoch:04d} | {total_mins:6.2f}m] "
                f"Time={elapsed:5.2f}s | Buf={len(self.buffer):5d} | "
                f"AvgPlies={epoch_stats['avg_plies']:5.1f} (B:{blue_wins}/R:{red_wins}/D:{draws}) | "
                f"Loss={losses['total_loss']:.4f} "
                f"(Pol={losses['policy_loss']:.4f}, Val={losses['value_loss']:.4f}) | LR={curr_lr:.5f}"
            )
            sys.stdout.flush()

            if self.config.checkpoint_path:
                ckpt_file = Path(self.config.checkpoint_path)
                ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "total_minutes": total_mins,
                        "model_state_dict": self.network.state_dict(),
                        "config": self.config.__dict__,
                    },
                    ckpt_file,
                )

            # Periodic Arena Evaluation against Random, Greedy, and Minimax(d=2)
            if self.config.eval_every_epochs > 0 and session_step % self.config.eval_every_epochs == 0:
                az_eval = AlphaZeroPlayer(
                    network=self.network,  # type: ignore[arg-type]
                    num_simulations=max(20, self.config.num_simulations),
                    device=self.device,
                    name=f"AZ(ep={epoch})",
                )
                vs_rand = play_series(az_eval, self.random_eval, num_games=4, max_plies=140)
                vs_greedy = play_series(az_eval, self.greedy_sparring, num_games=4, max_plies=140)
                vs_minimax = play_series(az_eval, self.minimax_sparring, num_games=2, max_plies=140)
                combined_score = (
                    0.2 * vs_rand.a_win_rate
                    + 0.4 * vs_greedy.a_win_rate
                    + 0.4 * vs_minimax.a_win_rate
                )
                print(
                    f"  >>> [Eval @ Epoch {epoch:04d}] "
                    f"vs Random: +{vs_rand.a_wins}-{vs_rand.b_wins}={vs_rand.draws} ({vs_rand.a_win_rate*100:.0f}%) | "
                    f"vs Greedy: +{vs_greedy.a_wins}-{vs_greedy.b_wins}={vs_greedy.draws} ({vs_greedy.a_win_rate*100:.0f}%) | "
                    f"vs Minimax: +{vs_minimax.a_wins}-{vs_minimax.b_wins}={vs_minimax.draws} ({vs_minimax.a_win_rate*100:.0f}%)"
                )
                if combined_score >= self.best_eval_score and self.config.best_checkpoint_path:
                    self.best_eval_score = combined_score
                    best_file = Path(self.config.best_checkpoint_path)
                    best_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "eval_score": combined_score,
                            "model_state_dict": self.network.state_dict(),
                            "config": self.config.__dict__,
                        },
                        best_file,
                    )
                    print(f"  >>> Saved new best model to {best_file} (Score={combined_score*100:.1f}%)")
                sys.stdout.flush()

        return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaZero RL Bot for Intransitive")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (if --duration-minutes not set)")
    parser.add_argument("--duration-minutes", type=float, default=None, help="Wall-clock training duration in minutes (e.g. 360)")
    parser.add_argument("--continuous", action="store_true", help="Run continuous training indefinitely until stopped")
    parser.add_argument("--episodes", type=int, default=6, help="Self-play games per epoch")
    parser.add_argument("--simulations", type=int, default=40, help="MCTS simulations per move")
    parser.add_argument("--mcts-batch", type=int, default=None, help="MCTS virtual-loss batch size")
    parser.add_argument("--max-plies", type=int, default=120, help="Max plies per self-play episode")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--steps", type=int, default=24, help="Gradient steps per epoch")
    parser.add_argument("--eval-every", type=int, default=15, help="Evaluate against baselines every N epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'xpu', 'cuda', or 'cpu'")
    parser.add_argument("--ipex-optimize", action="store_true", help="Apply ipex.optimize() to network and optimizer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/alphazero_latest.pt")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        duration_minutes=args.duration_minutes,
        continuous=args.continuous,
        episodes_per_epoch=args.episodes,
        num_simulations=args.simulations,
        mcts_batch_size=args.mcts_batch,
        max_plies_per_game=args.max_plies,
        batch_size=args.batch_size,
        train_steps_per_epoch=args.steps,
        eval_every_epochs=args.eval_every,
        device=args.device,
        use_ipex_optimize=args.ipex_optimize,
        checkpoint_path=args.checkpoint,
    )
    trainer = AlphaZeroTrainer(cfg)
    trainer.run_training()


if __name__ == "__main__":
    main()
