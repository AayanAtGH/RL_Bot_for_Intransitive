"""Round-robin tournament and Elo benchmark across baselines and AlphaZero."""

import argparse
from pathlib import Path
from typing import Dict, List
import torch

from intransitive.arena.evaluator import play_series, update_elo
from intransitive.baselines.base import Player
from intransitive.baselines.greedy_player import GreedyPlayer
from intransitive.baselines.minimax_player import MinimaxPlayer
from intransitive.baselines.random_player import RandomPlayer
from intransitive.rl.mcts import AlphaZeroPlayer
from intransitive.rl.network import IntransitiveNet


def run_benchmark(num_games_per_pair: int = 4, checkpoint_path: str = "checkpoints/alphazero_latest.pt") -> Dict[str, float]:
    net = IntransitiveNet()
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(state["model_state_dict"])
        az_name = f"AlphaZero(ep={state.get('epoch', '?')})"
    else:
        az_name = "AlphaZero(init)"

    players: List[Player] = [
        RandomPlayer(seed=1, name="RandomBot"),
        GreedyPlayer(seed=2, name="GreedyBot"),
        MinimaxPlayer(depth=2, seed=3, name="Minimax(d=2)"),
        AlphaZeroPlayer(network=net, num_simulations=15, name=az_name),
    ]

    elos: Dict[str, float] = {p.name: 1200.0 for p in players}
    print(f"=== Running Intransitive Arena Benchmark ({num_games_per_pair} games/pair) ===")

    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pa, pb = players[i], players[j]
            stats = play_series(pa, pb, num_games=num_games_per_pair, max_plies=150)
            avg_plies = stats.total_plies / max(1, stats.games)
            print(
                f"  {pa.name:16s} vs {pb.name:16s} -> "
                f"+{stats.a_wins} -{stats.b_wins} ={stats.draws} "
                f"(WinRate: {stats.a_win_rate * 100:5.1f}%, AvgPlies: {avg_plies:.1f})"
            )
            # Apply Elo updates across games
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
    parser.add_argument("--checkpoint", type=str, default="checkpoints/alphazero_latest.pt")
    args = parser.parse_args()
    run_benchmark(num_games_per_pair=args.games, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
