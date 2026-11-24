"""UCI / JSON stdio bridge for querying the AlphaZero Intransitive engine."""

from pathlib import Path
import sys
from typing import Optional

from intransitive.core.board import Board
from intransitive.core.constants import START_FEN
from intransitive.core.tablebase import TablebaseManager
from intransitive.rl.mcts import AlphaZeroPlayer
from intransitive.rl.network import IntransitiveNet, load_checkpoint


def run_uci_loop(
    checkpoint: str = "checkpoints/alphazero_v2_best.pt",
    tb_dir: str = "data/tablebase",
    simulations: int = 200,
) -> None:
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        alt = Path("checkpoints/alphazero_best.pt")
        if alt.exists():
            ckpt = alt
    if ckpt.exists():
        net, _ = load_checkpoint(ckpt, device="cpu", prefer_ema=True)
    else:
        net = IntransitiveNet()

    tb: Optional[TablebaseManager] = None
    if Path(tb_dir).exists():
        cand = TablebaseManager(tb_dir)
        if cand.is_loaded():
            tb = cand

    agent = AlphaZeroPlayer(network=net, num_simulations=simulations, tablebase=tb)
    board = Board(fen=START_FEN)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            print("id name Intransitive-AlphaZero-v2")
            print("id author Antigravity")
            print("uciok")
            sys.stdout.flush()
        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()
        elif cmd == "ucinewgame":
            board = Board(fen=START_FEN)
            agent.reset()
        elif cmd == "position":
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
