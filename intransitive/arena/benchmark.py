"""Round-robin tournament and Elo benchmark across baselines, v1, and v2 AlphaZero."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from intransitive.arena.evaluator import play_series, update_elo
from intransitive.baselines.base import Player
from intransitive.baselines.greedy_player import GreedyPlayer
from intransitive.baselines.minimax_player import MinimaxPlayer
from intransitive.baselines.random_player import RandomPlayer
from intransitive.core.tablebase import TablebaseManager
from intransitive.rl.mcts import AlphaZeroPlayer
from intransitive.rl.network import IntransitiveNet, load_checkpoint


def run_benchmark(
    num_games_per_pair: int = 4,
    checkpoint_path: str = "checkpoints/alphazero_v2_best.pt",
    v1_checkpoint_path: Optional[str] = "checkpoints/alphazero_best.pt",
    tb_dir: Optional[str] = "data/tablebase",
    simulations: int = 100,
) -> Dict[str, float]:
    tb: Optional[TablebaseManager] = None
    if tb_dir and Path(tb_dir).exists():
        cand = TablebaseManager(tb_dir)
        if cand.is_loaded():
            tb = cand

    players: List[Player] = [
        RandomPlayer(seed=1, name="RandomBot"),
        GreedyPlayer(seed=2, name="GreedyBot"),
        MinimaxPlayer(depth=2, seed=3, name="Minimax(d=2)"),
    ]

    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        net, meta = load_checkpoint(ckpt, device="cpu", prefer_ema=True)
        tag = meta.get("iteration", meta.get("epoch", "?"))
        az_name = f"AZ-v{meta.get('arch_version', 1)}({tag})"
        players.append(AlphaZeroPlayer(network=net, num_simulations=simulations, tablebase=tb, name=az_name))
    else:
        players.append(
            AlphaZeroPlayer(network=IntransitiveNet(), num_simulations=simulations, tablebase=tb, name="AZ-v2(init)")
        )

    if v1_checkpoint_path and Path(v1_checkpoint_path).exists() and Path(v1_checkpoint_path).resolve() != ckpt.resolve():
        net_v1, meta_v1 = load_checkpoint(v1_checkpoint_path, device="cpu", prefer_ema=True)
        tag_v1 = meta_v1.get("epoch", "?")
        players.append(
            AlphaZeroPlayer(
                network=net_v1,
                num_simulations=simulations,
                tablebase=tb,
                name=f"AZ-v1(ep={tag_v1})",
            )
        )

    elos: Dict[str, float] = {p.name: 1200.0 for p in players}
    print(f"=== Running Intransitive Arena Benchmark ({num_games_per_pair} games/pair) ===")

    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pa, pb = players[i], players[j]
            stats = play_series(pa, pb, num_games=num_games_per_pair, max_plies=200)
            avg_plies = stats.total_plies / max(1, stats.games)
            print(
                f"  {pa.name:18s} vs {pb.name:18s} -> "
                f"+{stats.a_wins} -{stats.b_wins} ={stats.draws} "
                f"(WinRate: {stats.a_win_rate * 100:5.1f}%, AvgPlies: {avg_plies:.1f})"
            )
            for _ in range(stats.a_wins):
                elos[pa.name], elos[pb.name] = update_elo(elos[pa.name], elos[pb.name], 1.0)
            for _ in range(stats.b_wins):
                elos[pa.name], elos[pb.name] = update_elo(elos[pa.name], elos[pb.name], 0.0)
            for _ in range(stats.draws):
                elos[pa.name], elos[pb.name] = update_elo(elos[pa.name], elos[pb.name], 0.5)

    print("\n=== Final Elo Standings ===")
    for rank, (name, rating) in enumerate(sorted(elos.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  {rank}. {name:20s} : {rating:6.1f} Elo")

    return elos


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Intransitive Arena Benchmark")
    parser.add_argument("--games", type=int, default=4, help="Games per pair")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/alphazero_v2_best.pt")
    parser.add_argument("--v1-checkpoint", type=str, default="checkpoints/alphazero_best.pt")
    parser.add_argument("--tb-dir", type=str, default="data/tablebase")
    parser.add_argument("--simulations", type=int, default=100)
    args = parser.parse_args()
    run_benchmark(
        num_games_per_pair=args.games,
        checkpoint_path=args.checkpoint,
        v1_checkpoint_path=args.v1_checkpoint if args.v1_checkpoint else None,
        tb_dir=args.tb_dir if args.tb_dir else None,
        simulations=args.simulations,
    )


if __name__ == "__main__":
    main()
