"""Uniform random baseline player."""

from typing import Optional
import numpy as np

from intransitive.baselines.base import Player
from intransitive.core.board import Board


class RandomPlayer(Player):
    """Selects uniformly at random from all legal moves."""

    def __init__(self, seed: Optional[int] = None, name: str = "RandomBot") -> None:
        super().__init__(name=name)
        self.rng = np.random.default_rng(seed)

    def select_move(self, board: Board) -> str:
        moves = board.legal_moves()
        if not moves:
            raise ValueError("No legal moves available")
        idx = int(self.rng.integers(0, len(moves)))
        return moves[idx]
