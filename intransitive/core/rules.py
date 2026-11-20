"""Rules, move generation, and terminal state detection for Intransitive (RPS V6)."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from intransitive.core.constants import (
    ACTION_SIZE,
    BEATS_TYPE,
    BLUE,
    BLUE_GOAL,
    BOARD_SIZE,
    DIRECTIONS,
    EMPTY,
    NUM_DIRECTIONS,
    NUM_SQUARES,
    QUIET_LIMIT,
    RED,
    RED_GOAL,
)
from intransitive.core.move import square_label, to_uci, uci_to_squares

# Precompute valid 1-step king transitions for every square 0..80:
# Each entry is a list of (to_sq, action_idx, uci_str)
_SQUARE_TRANSITIONS: List[List[Tuple[int, int, str]]] = [[] for _ in range(NUM_SQUARES)]
for _sq in range(NUM_SQUARES):
    _r, _f = _sq // BOARD_SIZE, _sq % BOARD_SIZE
    _from_lbl = square_label(_sq)
    for _d_idx, (_dr, _df) in enumerate(DIRECTIONS):
        _nr, _nf = _r + _dr, _f + _df
        if 0 <= _nr < BOARD_SIZE and 0 <= _nf < BOARD_SIZE:
            _to_sq = _nr * BOARD_SIZE + _nf
            _act = _sq * NUM_DIRECTIONS + _d_idx
            _uci = _from_lbl + square_label(_to_sq)
            _SQUARE_TRANSITIONS[_sq].append((_to_sq, _act, _uci))


@dataclass(frozen=True)
class GameResult:
    """Represents a terminal game outcome."""

    winner: int  # BLUE (1), RED (-1), or 0 for draw
    reason: str  # 'corner', 'no-moves', or 'quiet-limit'

    @property
    def winner_str(self) -> str:
        if self.winner == BLUE:
            return "b"
        if self.winner == RED:
            return "r"
        return "draw"


def can_capture(attacker_piece: int, victim_piece: int) -> bool:
    """Return True if attacker_piece can legally capture victim_piece."""
    if attacker_piece == EMPTY or victim_piece == EMPTY:
        return False
    # Must belong to opposite sides
    if (attacker_piece > 0) == (victim_piece > 0):
        return False
    return BEATS_TYPE[abs(attacker_piece)] == abs(victim_piece)


def pseudo_moves(board: np.ndarray, side: int) -> List[str]:
    """Generate all pseudo-legal UCI moves before terminal rules are checked."""
    moves: List[str] = []
    for from_sq in range(NUM_SQUARES):
        piece = int(board[from_sq])
        if piece == EMPTY or (piece > 0 if side == RED else piece < 0):
            continue
        target_victim_type = BEATS_TYPE[abs(piece)]
        target_victim = -target_victim_type if side == BLUE else target_victim_type
        for to_sq, _, uci in _SQUARE_TRANSITIONS[from_sq]:
            victim = int(board[to_sq])
            if victim == EMPTY or victim == target_victim:
                moves.append(uci)
    return moves


def pseudo_actions(board: np.ndarray, side: int) -> List[int]:
    """Generate all pseudo-legal flat action indices (0..647) before terminal checks."""
    actions: List[int] = []
    for from_sq in range(NUM_SQUARES):
        piece = int(board[from_sq])
        if piece == EMPTY or (piece > 0 if side == RED else piece < 0):
            continue
        target_victim_type = BEATS_TYPE[abs(piece)]
        target_victim = -target_victim_type if side == BLUE else target_victim_type
        for to_sq, act, _ in _SQUARE_TRANSITIONS[from_sq]:
            victim = int(board[to_sq])
            if victim == EMPTY or victim == target_victim:
                actions.append(act)
    return actions


def has_any_pseudo_move(board: np.ndarray, side: int) -> bool:
    """Fast check for whether `side` has at least one pseudo-legal move."""
    for from_sq in range(NUM_SQUARES):
        piece = int(board[from_sq])
        if piece == EMPTY or (piece > 0 if side == RED else piece < 0):
            continue
        target_victim_type = BEATS_TYPE[abs(piece)]
        target_victim = -target_victim_type if side == BLUE else target_victim_type
        for to_sq, _, _ in _SQUARE_TRANSITIONS[from_sq]:
            victim = int(board[to_sq])
            if victim == EMPTY or victim == target_victim:
                return True
    return False


def has_immediate_corner_win(board: np.ndarray, side: int) -> bool:
    """Return True if `side` has at least one legal move reaching their goal corner."""
    goal_sq = BLUE_GOAL if side == BLUE else RED_GOAL
    victim = int(board[goal_sq])
    r_goal, f_goal = goal_sq // BOARD_SIZE, goal_sq % BOARD_SIZE
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            nr, nf = r_goal + dr, f_goal + df
            if 0 <= nr < BOARD_SIZE and 0 <= nf < BOARD_SIZE:
                sq = nr * BOARD_SIZE + nf
                piece = int(board[sq])
                if piece == EMPTY:
                    continue
                if (piece > 0 and side == BLUE) or (piece < 0 and side == RED):
                    if victim == EMPTY:
                        return True
                    if (piece > 0) != (victim > 0) and BEATS_TYPE[abs(piece)] == abs(victim):
                        return True
    return False


def game_result(board: np.ndarray, side: int, quiet: int = 0) -> Optional[GameResult]:
    """
    Determine terminal state, or None if game is live.
    Rules match Hushhush / meaf.us V6 specification:
    1. A Blue piece standing on i9 (80) wins immediately for Blue.
    2. A Red piece standing on a1 (0) wins immediately for Red.
    3. If side to move has no pseudo-legal moves, opponent wins ('no-moves').
    4. If quiet >= 200, game is a draw ('quiet-limit').
    """
    if int(board[BLUE_GOAL]) > 0:
        return GameResult(winner=BLUE, reason="corner")
    if int(board[RED_GOAL]) < 0:
        return GameResult(winner=RED, reason="corner")
    if not has_any_pseudo_move(board, side):
        return GameResult(winner=-side, reason="no-moves")
    if quiet >= QUIET_LIMIT:
        return GameResult(winner=0, reason="quiet-limit")
    return None


def legal_moves(board: np.ndarray, side: int, quiet: int = 0) -> List[str]:
    """Return all legal UCI moves (empty list if game is already terminal)."""
    if game_result(board, side, quiet) is not None:
        return []
    return pseudo_moves(board, side)


def legal_actions(board: np.ndarray, side: int, quiet: int = 0) -> List[int]:
    """Return all legal flat action indices 0..647 (empty if terminal)."""
    if game_result(board, side, quiet) is not None:
        return []
    return pseudo_actions(board, side)


def action_mask(board: np.ndarray, side: int, quiet: int = 0) -> np.ndarray:
    """Return boolean mask of shape (648,) where True indicates a legal action."""
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    for act in legal_actions(board, side, quiet):
        mask[act] = True
    return mask


def play_move(
    board: np.ndarray, side: int, move: str, quiet: int = 0, validate: bool = True
) -> Tuple[np.ndarray, int, int, bool]:
    """
    Execute a move on a copy of `board`.
    Returns (next_board, next_side, next_quiet, was_capture).
    """
    uci = to_uci(move)
    if validate and uci not in legal_moves(board, side, quiet):
        raise ValueError(f"Illegal V6 move: {uci}")

    from_sq, to_sq = uci_to_squares(uci)
    next_board = board.copy()
    was_capture = int(board[to_sq]) != EMPTY
    next_board[to_sq] = next_board[from_sq]
    next_board[from_sq] = EMPTY
    next_side = -side
    next_quiet = 0 if was_capture else quiet + 1
    return next_board, next_side, next_quiet, was_capture
