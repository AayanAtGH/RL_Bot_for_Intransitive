"""Interactive terminal game runner for Intransitive (Human vs Bot or Bot vs Bot)."""

import argparse
from pathlib import Path
import torch

from intransitive.baselines.base import Player
from intransitive.baselines.greedy_player import GreedyPlayer
from intransitive.baselines.minimax_player import MinimaxPlayer
from intransitive.baselines.random_player import RandomPlayer
from intransitive.core.board import Board
from intransitive.core.constants import BLUE, START_FEN
from intransitive.core.move import to_notation, to_uci
from intransitive.rl.mcts import AlphaZeroPlayer
from intransitive.rl.network import IntransitiveNet


class HumanPlayer(Player):
    """Interactive CLI human player."""

    def __init__(self) -> None:
        super().__init__(name="Human")

    def select_move(self, board: Board) -> str:
        legal = board.legal_moves()
        while True:
            raw = input(f"Enter move (e.g. {legal[0]}, or 'moves' to list all {len(legal)}): ").strip()
            if raw.lower() == "moves":
                print("Legal moves:", ", ".join(legal))
                continue
            try:
                uci = to_uci(raw)
                if uci in legal:
                    return uci
                print(f"Illegal move {uci!r}. Type 'moves' to see legal moves.")
            except ValueError as exc:
                print(f"Invalid format: {exc}")


def build_agent(kind: str, checkpoint: str = "checkpoints/alphazero_latest.pt") -> Player:
    kind = kind.lower()
    if kind == "human":
        return HumanPlayer()
    if kind == "random":
        return RandomPlayer()
    if kind == "greedy":
        return GreedyPlayer()
    if kind == "minimax":
        return MinimaxPlayer(depth=2)
    if kind == "alphazero":
        net = IntransitiveNet()
        ckpt = Path(checkpoint)
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            net.load_state_dict(state["model_state_dict"])
        return AlphaZeroPlayer(network=net, num_simulations=30)
    raise ValueError(f"Unknown player type: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Intransitive in the terminal")
    parser.add_argument("--blue", type=str, default="human", choices=["human", "random", "greedy", "minimax", "alphazero"])
    parser.add_argument("--red", type=str, default="alphazero", choices=["human", "random", "greedy", "minimax", "alphazero"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/alphazero_latest.pt")
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args()

    blue_p = build_agent(args.blue, args.checkpoint)
    red_p = build_agent(args.red, args.checkpoint)

    board = Board(fen=START_FEN)
    print(board.render_ascii())

    while not board.is_terminal() and board.ply < args.max_plies:
        active = blue_p if board.side == BLUE else red_p
        move = active.select_move(board)
        notation = to_notation(board.grid, move)
        board.push_uci(move)
        print(f"\n[{active.name}] played {notation} ({move}):")
        print(board.render_ascii())

    res = board.result()
    if res is not None:
        print(f"\nGame Over! Winner: {res.winner_str.upper()} (Reason: {res.reason})")
    else:
        print("\nGame reached max plies.")


if __name__ == "__main__":
    main()
