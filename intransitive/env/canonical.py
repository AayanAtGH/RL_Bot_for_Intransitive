"""Canonical perspective transforms (180-degree rotation + color inversion) for symmetric RL."""

from typing import List
import numpy as np

from intransitive.core.board import Board
from intransitive.core.constants import (
    ACTION_SIZE,
    BLUE,
    BOARD_SIZE,
    DELTA_TO_DIR_IDX,
    DIRECTIONS,
    NUM_DIRECTIONS,
    NUM_SQUARES,
    PAPER,
    QUIET_LIMIT,
    ROCK,
    SCISSORS,
)
from intransitive.core.move import rotate_action

NUM_OBS_CHANNELS: int = 7

# Precompute main-diagonal reflection mapping (rank <-> file) for all 648 actions
_REFLECT_ACTION_MAP = np.zeros(ACTION_SIZE, dtype=np.int64)
for _sq in range(NUM_SQUARES):
    _r, _f = _sq // BOARD_SIZE, _sq % BOARD_SIZE
    _ref_sq = _f * BOARD_SIZE + _r
    for _d_idx, (_dr, _df) in enumerate(DIRECTIONS):
        _ref_d_idx = DELTA_TO_DIR_IDX[(_df, _dr)]
        _REFLECT_ACTION_MAP[_sq * NUM_DIRECTIONS + _d_idx] = _ref_sq * NUM_DIRECTIONS + _ref_d_idx


def reflect_obs_and_policy(obs: np.ndarray, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reflect a canonical (7, 9, 9) observation and (648,) policy across the a1-i9 main diagonal
    (swapping rank and file). Preserves corners a1 (0,0) and i9 (8,8) and RPS rules.
    """
    ref_obs = np.swapaxes(obs, 1, 2).copy()
    ref_policy = np.empty_like(policy)
    ref_policy[_REFLECT_ACTION_MAP] = policy
    return ref_obs, ref_policy


def canonical_grid(grid: np.ndarray, side: int) -> np.ndarray:
    """
    Return 81-element board array from the perspective of `side`.
    If side == BLUE, returns a copy of `grid`.
    If side == RED, rotates 180 degrees (reverses flat array) and negates piece signs
    so the active player always owns positive pieces (1, 2, 3) aiming for square 80 (i9).
    """
    if side == BLUE:
        return grid.copy()
    return (-grid[::-1]).copy()


def encode_board(board: Board) -> np.ndarray:
    """
    Encode `Board` into a canonical (7, 9, 9) float32 tensor:
      - Ch 0: Current player's Rocks
      - Ch 1: Current player's Papers
      - Ch 2: Current player's Scissors
      - Ch 3: Opponent's Rocks
      - Ch 4: Opponent's Papers
      - Ch 5: Opponent's Scissors
      - Ch 6: Normalized quiet clock (quiet / 200.0)
    """
    c_grid = canonical_grid(board.grid, board.side).reshape(BOARD_SIZE, BOARD_SIZE)
    obs = np.zeros((NUM_OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    obs[0] = (c_grid == ROCK).astype(np.float32)
    obs[1] = (c_grid == PAPER).astype(np.float32)
    obs[2] = (c_grid == SCISSORS).astype(np.float32)
    obs[3] = (c_grid == -ROCK).astype(np.float32)
    obs[4] = (c_grid == -PAPER).astype(np.float32)
    obs[5] = (c_grid == -SCISSORS).astype(np.float32)
    obs[6].fill(min(1.0, float(board.quiet) / float(QUIET_LIMIT)))
    return obs


def to_canonical_action(real_action: int, side: int) -> int:
    """Convert a real board action (0..647) into canonical perspective for `side`."""
    return real_action if side == BLUE else rotate_action(real_action)


def from_canonical_action(canonical_action: int, side: int) -> int:
    """Convert a canonical action (0..647) back into real board coordinates for `side`."""
    return canonical_action if side == BLUE else rotate_action(canonical_action)


def canonical_legal_actions(board: Board) -> List[int]:
    """Return list of legal actions in canonical coordinates for the current side to move."""
    real_acts = board.legal_actions()
    if board.side == BLUE:
        return real_acts
    return [rotate_action(a) for a in real_acts]


def canonical_action_mask(board: Board) -> np.ndarray:
    """Return boolean mask of shape (648,) in canonical coordinates."""
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    for act in canonical_legal_actions(board):
        mask[act] = True
    return mask
