"""Greedy heuristic player for Intransitive."""

from typing import Optional
import numpy as np

from intransitive.baselines.base import Player
from intransitive.core.board import Board
from intransitive.core.constants import (
    BEATS_TYPE,
    BLUE,
    BLUE_GOAL,
    BOARD_SIZE,
    DIRECTIONS,
    EMPTY,
    RED_GOAL,
)
from intransitive.core.move import uci_to_squares


def _chebyshev_dist(sq1: int, sq2: int) -> int:
    r1, f1 = sq1 // BOARD_SIZE, sq1 % BOARD_SIZE
    r2, f2 = sq2 // BOARD_SIZE, sq2 % BOARD_SIZE
    return max(abs(r1 - r2), abs(f1 - f2))


def _is_square_threatened_for_piece(grid: np.ndarray, sq: int, piece: int, from_sq: int) -> bool:
    """Return True if an adjacent enemy piece can capture `piece` when it stands on `sq`."""
    r, f = sq // BOARD_SIZE, sq % BOARD_SIZE
    # Which enemy piece type beats `piece`?
    # If piece is R(1), P(2) beats it. More generally, k such that BEATS_TYPE[k] == abs(piece)
    abs_p = abs(piece)
    nemesis_type = 2 if abs_p == 1 else (3 if abs_p == 2 else 1)
    enemy_nemesis = -nemesis_type if piece > 0 else nemesis_type

    for dr, df in DIRECTIONS:
        nr, nf = r + dr, f + df
        if 0 <= nr < BOARD_SIZE and 0 <= nf < BOARD_SIZE:
            nsq = nr * BOARD_SIZE + nf
            if nsq == from_sq:
                continue
            if int(grid[nsq]) == enemy_nemesis:
                return True
    return False


class GreedyPlayer(Player):
    """
    Heuristic agent that prioritizes:
    1. Immediate corner victory
    2. Safe captures of opponent pieces
    3. Avoiding stepping into immediate capture threats
    4. Advancing closer to the opponent's corner
    """

    def __init__(self, seed: Optional[int] = None, name: str = "GreedyBot") -> None:
        super().__init__(name=name)
        self.rng = np.random.default_rng(seed)

    def select_move(self, board: Board) -> str:
        moves = board.legal_moves()
        if not moves:
            raise ValueError("No legal moves available")

        goal_sq = BLUE_GOAL if board.side == BLUE else RED_GOAL
        best_score = -1e9
        best_moves = []

        for mv in moves:
            from_sq, to_sq = uci_to_squares(mv)
            piece = int(board.grid[from_sq])
            victim = int(board.grid[to_sq])

            # 1. Immediate corner win
            if to_sq == goal_sq:
                return mv

            score = 0.0
            # 2. Capture bonus
            if victim != EMPTY:
                score += 100.0

            # 3. Distance to target corner improvement
            dist_before = _chebyshev_dist(from_sq, goal_sq)
            dist_after = _chebyshev_dist(to_sq, goal_sq)
            score += (dist_before - dist_after) * 12.0
            # Prefer advancing pieces that are already closer to goal
            score += (8 - dist_after) * 1.5

            # 4. Penalty for stepping into enemy nemesis range
            if _is_square_threatened_for_piece(board.grid, to_sq, piece, from_sq):
                score -= 110.0

            # Bonus for escaping a currently threatened square
            if _is_square_threatened_for_piece(board.grid, from_sq, piece, from_sq):
                score += 45.0

            if score > best_score + 1e-6:
                best_score = score
                best_moves = [mv]
            elif abs(score - best_score) <= 1e-6:
                best_moves.append(mv)

        idx = int(self.rng.integers(0, len(best_moves)))
        return best_moves[idx]
