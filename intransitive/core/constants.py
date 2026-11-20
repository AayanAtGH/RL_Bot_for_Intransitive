"""Constants and definitions for Intransitive (RPS V6)."""

from typing import Dict, List, Tuple

BOARD_SIZE: int = 9
NUM_SQUARES: int = 81
NUM_DIRECTIONS: int = 8
ACTION_SIZE: int = NUM_SQUARES * NUM_DIRECTIONS  # 81 * 8 = 648

# Sides
BLUE: int = 1   # 'b' in record FEN, moves first, uppercase pieces (R, P, S)
RED: int = -1   # 'r' in record FEN, moves second, lowercase pieces (r, p, s)

SIDE_CHAR_TO_INT: Dict[str, int] = {"b": BLUE, "r": RED}
SIDE_INT_TO_CHAR: Dict[int, str] = {BLUE: "b", RED: "r"}

# Piece absolute types (positive for Blue, negative for Red)
EMPTY: int = 0
ROCK: int = 1
PAPER: int = 2
SCISSORS: int = 3

CHAR_TO_PIECE: Dict[str, int] = {
    ".": EMPTY,
    "R": ROCK,
    "P": PAPER,
    "S": SCISSORS,
    "r": -ROCK,
    "p": -PAPER,
    "s": -SCISSORS,
}

PIECE_TO_CHAR: Dict[int, str] = {v: k for k, v in CHAR_TO_PIECE.items()}

# Intransitive capture hierarchy on absolute piece types:
# Rock (1) beats Scissors (3), Paper (2) beats Rock (1), Scissors (3) beats Paper (2)
BEATS_TYPE: Dict[int, int] = {
    ROCK: SCISSORS,
    PAPER: ROCK,
    SCISSORS: PAPER,
}

# Corner square indices:
# Index 0 is a1 (rank 0, file 0) -> Blue's home base, Red's winning goal
# Index 80 is i9 (rank 8, file 8) -> Red's home base, Blue's winning goal
BLUE_CORNER: int = 0
RED_CORNER: int = 80
BLUE_GOAL: int = RED_CORNER   # i9 (index 80)
RED_GOAL: int = BLUE_CORNER   # a1 (index 0)

# Draw clock: 200 plies without a capture is a draw
QUIET_LIMIT: int = 200

# Standard V6 opening FEN (rank 1 first, Blue to move)
START_FEN: str = "9/3RP4/2RPS4/1RPS5/1PS3sp1/5spr1/4spr2/4pr3/9 b"

# 8 King movement directions ordered clockwise in (dr, df) so that
# a 180-degree rotation (-dr, -df) corresponds to (dir_idx + 4) % 8:
# 0: E  (0, +1)
# 1: NE (+1, +1)
# 2: N  (+1, 0)
# 3: NW (+1, -1)
# 4: W  (0, -1)
# 5: SW (-1, -1)
# 6: S  (-1, 0)
# 7: SE (-1, +1)
DIRECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
]

DELTA_TO_DIR_IDX: Dict[Tuple[int, int], int] = {
    delta: idx for idx, delta in enumerate(DIRECTIONS)
}
