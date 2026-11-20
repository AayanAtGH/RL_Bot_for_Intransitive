"""UCI / JSON stdio bridge for querying the AlphaZero Intransitive engine."""

import sys
from pathlib import Path
import torch

from intransitive.core.board import Board
from intransitive.core.constants import START_FEN
from intransitive.rl.mcts import AlphaZeroPlayer
from intransitive.rl.network import IntransitiveNet


def run_uci_loop(checkpoint: str = "checkpoints/alphazero_latest.pt") -> None:
    net = IntransitiveNet()
    ckpt = Path(checkpoint)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(state["model_state_dict"])

    agent = AlphaZeroPlayer(network=net, num_simulations=40)
    board = Board(fen=START_FEN)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            print("id name Intransitive-AlphaZero")
            print("id author Antigravity")
            print("uciok")
            sys.stdout.flush()
        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()
        elif cmd == "ucinewgame":
            board = Board(fen=START_FEN)
        elif cmd == "position":
            # Supports: position startpos [moves m1 m2 ...] or position fen <fen> [moves m1 m2 ...]
            if "startpos" in tokens:
                board = Board(fen=START_FEN)
                if "moves" in tokens:
                    mv_idx = tokens.index("moves") + 1
                    for mv in tokens[mv_idx:]:
                        board.push_uci(mv)
            elif "fen" in tokens:
                fen_idx = tokens.index("fen") + 1
                if "moves" in tokens:
                    mv_idx = tokens.index("moves")
                    fen_str = " ".join(tokens[fen_idx:mv_idx])
                    board = Board(fen=fen_str)
                    for mv in tokens[mv_idx + 1 :]:
                        board.push_uci(mv)
                else:
                    fen_str = " ".join(tokens[fen_idx:])
                    board = Board(fen=fen_str)
        elif cmd == "go":
            best = agent.select_move(board)
            print(f"bestmove {best}")
            sys.stdout.flush()
        elif cmd == "quit":
            break


if __name__ == "__main__":
    run_uci_loop()
