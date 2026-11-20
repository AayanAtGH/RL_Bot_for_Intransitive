"""Move encoding, square conversion, notation, and action index mapping."""

import re
from typing import Tuple
import numpy as np

from intransitive.core.constants import (
    BOARD_SIZE,
    DELTA_TO_DIR_IDX,
    DIRECTIONS,
    EMPTY,
    NUM_DIRECTIONS,
    NUM_SQUARES,
    PIECE_TO_CHAR,
)

_MOVE_REGEX = re.compile(r"^[RPSrps]?([a-i][1-9])[-x]?(?:[RPSrps])?([a-i][1-9])$")


def square_index(label: str) -> int:
    """Convert algebraic square label (e.g. 'd4') to flat index 0..80."""
    if len(label) != 2:
        raise ValueError(f"Invalid square label: {label!r}")
    file_idx = ord(label[0]) - ord("a")
    rank_idx = int(label[1]) - 1
    if not (0 <= file_idx < BOARD_SIZE and 0 <= rank_idx < BOARD_SIZE):
        raise ValueError(f"Square out of bounds: {label!r}")
    return rank_idx * BOARD_SIZE + file_idx


def square_label(index: int) -> str:
    """Convert flat index 0..80 to algebraic square label (e.g. 30 -> 'd4')."""
    if not (0 <= index < NUM_SQUARES):
        raise ValueError(f"Square index out of bounds: {index}")
    file_idx = index % BOARD_SIZE
    rank_idx = index // BOARD_SIZE
    return f"{chr(ord('a') + file_idx)}{rank_idx + 1}"


def square_coords(index: int) -> Tuple[int, int]:
    """Return (rank, file) tuple in 0..8 for flat index 0..80."""
    return index // BOARD_SIZE, index % BOARD_SIZE


def to_uci(move: str) -> str:
    """Accept 'd3d4', 'Rd3-d4', or 'Rd3xd4' and return 'd3d4'."""
    match = _MOVE_REGEX.match(move.strip())
    if not match:
        raise ValueError(f"Invalid move format: {move!r}")
    return match.group(1) + match.group(2)


def uci_to_squares(move: str) -> Tuple[int, int]:
    """Parse move string into (from_sq, to_sq) flat indices."""
    uci = to_uci(move)
    return square_index(uci[:2]), square_index(uci[2:4])


def squares_to_uci(from_sq: int, to_sq: int) -> str:
    """Convert (from_sq, to_sq) flat indices into UCI move string."""
    return square_label(from_sq) + square_label(to_sq)


def to_notation(board_array: np.ndarray, move: str) -> str:
    """Format a move in record notation: 'Rd3-d4' or 'Rd3xd4'."""
    uci = to_uci(move)
    from_sq = square_index(uci[:2])
    to_sq = square_index(uci[2:4])
    piece = int(board_array[from_sq])
    if piece == EMPTY:
        raise ValueError(f"No piece on {uci[:2]}")
    piece_char = PIECE_TO_CHAR[piece].upper()
    sep = "-" if int(board_array[to_sq]) == EMPTY else "x"
    return f"{piece_char}{uci[:2]}{sep}{uci[2:4]}"


def uci_to_action(move: str) -> int:
    """Convert a 1-step king move into flat action index in 0..647."""
    from_sq, to_sq = uci_to_squares(move)
    r1, f1 = from_sq // BOARD_SIZE, from_sq % BOARD_SIZE
    r2, f2 = to_sq // BOARD_SIZE, to_sq % BOARD_SIZE
    delta = (r2 - r1, f2 - f1)
    if delta not in DELTA_TO_DIR_IDX:
        raise ValueError(f"Move {move!r} is not a 1-step king move (delta={delta})")
    dir_idx = DELTA_TO_DIR_IDX[delta]
    return from_sq * NUM_DIRECTIONS + dir_idx


def action_to_squares(action: int) -> Tuple[int, int]:
    """Convert flat action index 0..647 into (from_sq, to_sq). Raises ValueError if off-board."""
    if not (0 <= action < NUM_SQUARES * NUM_DIRECTIONS):
        raise ValueError(f"Action index out of range: {action}")
    from_sq = action // NUM_DIRECTIONS
    dir_idx = action % NUM_DIRECTIONS
    r1, f1 = from_sq // BOARD_SIZE, from_sq % BOARD_SIZE
    dr, df = DIRECTIONS[dir_idx]
    r2, f2 = r1 + dr, f1 + df
    if not (0 <= r2 < BOARD_SIZE and 0 <= f2 < BOARD_SIZE):
        raise ValueError(f"Action {action} steps off board from {square_label(from_sq)}")
    to_sq = r2 * BOARD_SIZE + f2
    return from_sq, to_sq


def action_to_uci(action: int) -> str:
    """Convert flat action index 0..647 into UCI string (e.g. 'd3d4')."""
    from_sq, to_sq = action_to_squares(action)
    return squares_to_uci(from_sq, to_sq)


def rotate_square(sq: int) -> int:
    """Rotate square index 180 degrees around the center of the 9x9 board."""
    return (NUM_SQUARES - 1) - sq


def rotate_action(action: int) -> int:
    """Rotate flat action index 180 degrees (involutive: rotate_action(rotate_action(a)) == a)."""
    from_sq = action // NUM_DIRECTIONS
    dir_idx = action % NUM_DIRECTIONS
    rot_sq = (NUM_SQUARES - 1) - from_sq
    rot_dir = (dir_idx + 4) % NUM_DIRECTIONS
    return rot_sq * NUM_DIRECTIONS + rot_dir
