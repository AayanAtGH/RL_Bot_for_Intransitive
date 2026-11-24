"""Interactive terminal game runner for Intransitive (Human vs Bot or Bot vs Bot)."""

import argparse
from pathlib import Path
from typing import Optional

from intransitive.baselines.base import Player
from intransitive.baselines.greedy_player import GreedyPlayer
from intransitive.baselines.minimax_player import MinimaxPlayer
from intransitive.baselines.random_player import RandomPlayer
from intransitive.core.board import Board
from intransitive.core.constants import BLUE, START_FEN
from intransitive.core.move import to_notation, to_uci
from intransitive.core.tablebase import TablebaseManager
from intransitive.rl.mcts import AlphaZeroPlayer
from intransitive.rl.network import IntransitiveNet, load_checkpoint


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


def _load_optional_tb(tb_dir: str) -> Optional[TablebaseManager]:
    p = Path(tb_dir)
    if p.exists():
        tb = TablebaseManager(p)
        if tb.is_loaded():
            return tb
    return None


def build_agent(
    kind: str,
    checkpoint: str = "checkpoints/alphazero_v2_best.pt",
    tb_dir: str = "data/tablebase",
    simulations: int = 200,
) -> Player:
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
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            fallback = Path("checkpoints/alphazero_best.pt")
            if fallback.exists():
                ckpt = fallback
        if ckpt.exists():
            net, _ = load_checkpoint(ckpt, device="cpu", prefer_ema=True)
        else:
            net = IntransitiveNet()
        tb = _load_optional_tb(tb_dir)
        return AlphaZeroPlayer(network=net, num_simulations=simulations, tablebase=tb)
    raise ValueError(f"Unknown player type: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Intransitive in the terminal")
    parser.add_argument("--blue", type=str, default="human", choices=["human", "random", "greedy", "minimax", "alphazero"])
    parser.add_argument("--red", type=str, default="alphazero", choices=["human", "random", "greedy", "minimax", "alphazero"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/alphazero_v2_best.pt")
    parser.add_argument("--tb-dir", type=str, default="data/tablebase")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args()

    blue_p = build_agent(args.blue, args.checkpoint, args.tb_dir, args.simulations)
    red_p = build_agent(args.red, args.checkpoint, args.tb_dir, args.simulations)

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
