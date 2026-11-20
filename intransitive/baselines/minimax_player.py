"""Alpha-Beta Minimax search player for Intransitive."""

from typing import List, Optional, Tuple
import numpy as np

from intransitive.baselines.base import Player
from intransitive.core.board import Board
from intransitive.core.constants import (
    BLUE,
    BLUE_GOAL,
    BOARD_SIZE,
    EMPTY,
    NUM_SQUARES,
    RED,
    RED_GOAL,
)
from intransitive.core.move import uci_to_squares

WIN_SCORE = 100_000.0

# Precompute Chebyshev distance to Blue goal (80) and Red goal (0)
_DIST_TO_BLUE_GOAL = np.array(
    [max(abs((sq // BOARD_SIZE) - 8), abs((sq % BOARD_SIZE) - 8)) for sq in range(NUM_SQUARES)],
    dtype=np.float32,
)
_DIST_TO_RED_GOAL = np.array(
    [max(sq // BOARD_SIZE, sq % BOARD_SIZE) for sq in range(NUM_SQUARES)],
    dtype=np.float32,
)


def evaluate_board(board: Board) -> float:
    """
    Static evaluation from the perspective of `board.side`.
    Combines:
    - Material balance (100 cp per piece)
    - Closest runner distance to goal corner (strong weight for endgame race)
    - Overall advancement toward opponent corner
    """
    res = board.result()
    if res is not None:
        if res.winner == 0:
            return 0.0
        return WIN_SCORE if res.winner == board.side else -WIN_SCORE

    grid = board.grid
    blue_mask = grid > 0
    red_mask = grid < 0

    blue_count = int(np.sum(blue_mask))
    red_count = int(np.sum(red_mask))
    material_blue = (blue_count - red_count) * 100.0

    # Advancement and lead runner bonus
    if blue_count > 0:
        blue_dists = _DIST_TO_BLUE_GOAL[blue_mask]
        min_blue_dist = float(np.min(blue_dists))
        adv_blue = float(np.sum(8.0 - blue_dists)) * 4.0 + (8.0 - min_blue_dist) ** 2 * 6.0
    else:
        adv_blue = -500.0

    if red_count > 0:
        red_dists = _DIST_TO_RED_GOAL[red_mask]
        min_red_dist = float(np.min(red_dists))
        adv_red = float(np.sum(8.0 - red_dists)) * 4.0 + (8.0 - min_red_dist) ** 2 * 6.0
    else:
        adv_red = -500.0

    score_blue = material_blue + (adv_blue - adv_red)
    return score_blue if board.side == BLUE else -score_blue


def _order_moves(board: Board, moves: List[str]) -> List[str]:
    """Order moves for alpha-beta efficiency: corner wins -> captures -> advancements."""
    goal_sq = BLUE_GOAL if board.side == BLUE else RED_GOAL
    dist_arr = _DIST_TO_BLUE_GOAL if board.side == BLUE else _DIST_TO_RED_GOAL
    grid = board.grid

    def move_priority(mv: str) -> float:
        from_sq, to_sq = uci_to_squares(mv)
        if to_sq == goal_sq:
            return 1000.0
        prio = 0.0
        if int(grid[to_sq]) != EMPTY:
            prio += 100.0
        prio += float(dist_arr[from_sq] - dist_arr[to_sq]) * 10.0
        return prio

    return sorted(moves, key=move_priority, reverse=True)


class MinimaxPlayer(Player):
    """Negamax Alpha-Beta search agent with move ordering."""

    def __init__(self, depth: int = 2, seed: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(name=name or f"Minimax(d={depth})")
        self.depth = depth
        self.rng = np.random.default_rng(seed)
        self.nodes_searched = 0

    def _negamax(self, board: Board, depth: int, alpha: float, beta: float) -> float:
        self.nodes_searched += 1
        res = board.result()
        if res is not None:
            if res.winner == 0:
                return 0.0
            # Prefer faster wins and slower losses
            base = WIN_SCORE + depth * 100.0
            return base if res.winner == board.side else -base

        if depth <= 0:
            return evaluate_board(board)

        moves = _order_moves(board, board.legal_moves())
        best_val = -1e12
        for mv in moves:
            child = board.copy()
            child.push_uci(mv, validate=False)
            val = -self._negamax(child, depth - 1, -beta, -alpha)
            if val > best_val:
                best_val = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        return best_val

    def select_move(self, board: Board) -> str:
        self.nodes_searched = 0
        moves = _order_moves(board, board.legal_moves())
        if not moves:
            raise ValueError("No legal moves available")

        best_val = -1e12
        best_moves: List[str] = []
        alpha = -1e12
        beta = 1e12

        for mv in moves:
            child = board.copy()
            child.push_uci(mv, validate=False)
            val = -self._negamax(child, self.depth - 1, -beta, -alpha)
            if val > best_val + 1e-5:
                best_val = val
                best_moves = [mv]
            elif abs(val - best_val) <= 1e-5:
                best_moves.append(mv)
            if val > alpha:
                alpha = val

        idx = int(self.rng.integers(0, len(best_moves)))
        return best_moves[idx]
