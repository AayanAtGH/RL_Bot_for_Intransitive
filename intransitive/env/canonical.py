"""Canonical perspective transforms (180-degree rotation + color inversion) and 6x symmetries."""

from typing import List, Tuple
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
NUM_OBS_CHANNELS_V2: int = 11

# Precompute goal proximity maps in canonical coordinates (own goal = i9, opp goal = a1)
_RANKS = np.arange(BOARD_SIZE, dtype=np.float32)[:, None]
_FILES = np.arange(BOARD_SIZE, dtype=np.float32)[None, :]
_GOAL_PROX_OWN: np.ndarray = (
    1.0 - np.maximum(8.0 - _RANKS, 8.0 - _FILES) / 8.0
).astype(np.float32)
_GOAL_PROX_OPP: np.ndarray = (
    1.0 - np.maximum(_RANKS, _FILES) / 8.0
).astype(np.float32)

# Precompute main-diagonal reflection mapping (rank <-> file) for squares and all 648 actions
_REFLECT_SQ_MAP = np.zeros(NUM_SQUARES, dtype=np.int64)
for _sq in range(NUM_SQUARES):
    _r, _f = _sq // BOARD_SIZE, _sq % BOARD_SIZE
    _REFLECT_SQ_MAP[_sq] = _f * BOARD_SIZE + _r

_REFLECT_ACTION_MAP = np.zeros(ACTION_SIZE, dtype=np.int64)
for _sq in range(NUM_SQUARES):
    _r, _f = _sq // BOARD_SIZE, _sq % BOARD_SIZE
    _ref_sq = _f * BOARD_SIZE + _r
    for _d_idx, (_dr, _df) in enumerate(DIRECTIONS):
        _ref_d_idx = DELTA_TO_DIR_IDX[(_df, _dr)]
        _REFLECT_ACTION_MAP[_sq * NUM_DIRECTIONS + _d_idx] = _ref_sq * NUM_DIRECTIONS + _ref_d_idx


def _dilate_king_grid(plane: np.ndarray) -> np.ndarray:
    """Dilate a (..., 9, 9) float/bool plane by 1 king step in all 8 directions."""
    padded = np.pad(plane, (*((0, 0),) * (plane.ndim - 2), (1, 1), (1, 1)), mode="constant")
    h, w = BOARD_SIZE, BOARD_SIZE
    out = np.zeros_like(plane, dtype=bool)
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            out |= padded[..., 1 + dr : 1 + dr + h, 1 + df : 1 + df + w].astype(bool)
    return out


def reflect_obs_and_policy(obs: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reflect a canonical (*, 9, 9) observation and (648,) policy across the a1-i9 main diagonal
    (swapping rank and file). Preserves corners a1 (0,0) and i9 (8,8) and RPS rules.
    """
    ref_obs = np.swapaxes(obs, 1, 2).copy()
    ref_policy = np.empty_like(policy)
    ref_policy[_REFLECT_ACTION_MAP] = policy
    return ref_obs, ref_policy


def cycle_rps_obs(obs: np.ndarray, shift: int) -> np.ndarray:
    """
    Cyclically permute Rock/Paper/Scissors channels by `shift` in {0, 1, 2}
    (pt -> (pt + shift) % 3) on a (C, 9, 9) or (B, C, 9, 9) observation.
    Policy actions and spatial/tactical channels are invariant under cyclic RPS shift.
    """
    s = shift % 3
    if s == 0:
        return obs.copy()
    out = obs.copy()
    if obs.ndim == 3:
        for pt in range(3):
            npt = (pt + s) % 3
            out[npt] = obs[pt]
            out[3 + npt] = obs[3 + pt]
    else:
        for pt in range(3):
            npt = (pt + s) % 3
            out[:, npt] = obs[:, pt]
            out[:, 3 + npt] = obs[:, 3 + pt]
    return out


def apply_symmetry_6x(
    obs: np.ndarray, policy: np.ndarray, sym_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one of the 6 exact D1 x C3 symmetries (`sym_id` in 0..5)."""
    reflect = (sym_id // 3) == 1
    shift = sym_id % 3
    o, p = (reflect_obs_and_policy(obs, policy) if reflect else (obs.copy(), policy.copy()))
    if shift != 0:
        o = cycle_rps_obs(o, shift)
    return o, p


def apply_symmetry_6x_grid(c_grid: np.ndarray, sym_id: int) -> np.ndarray:
    """Apply symmetry `sym_id` in 0..5 directly to a canonical int8 (81,) grid."""
    out = c_grid[_REFLECT_SQ_MAP].copy() if (sym_id // 3) == 1 else c_grid.copy()
    shift = sym_id % 3
    if shift != 0:
        pos_mask = out > 0
        neg_mask = out < 0
        if np.any(pos_mask):
            out[pos_mask] = ((out[pos_mask] - 1 + shift) % 3) + 1
        if np.any(neg_mask):
            out[neg_mask] = -(((np.abs(out[neg_mask]) - 1 + shift) % 3) + 1)
    return out


def apply_symmetry_6x_actions(actions: np.ndarray, sym_id: int) -> np.ndarray:
    """Apply symmetry `sym_id` in 0..5 to canonical action indices (only D1 affects actions)."""
    if (sym_id // 3) == 1:
        return _REFLECT_ACTION_MAP[actions]
    return actions.copy()


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


def encode_canonical_grid_v2(c_grid: np.ndarray, quiet: int) -> np.ndarray:
    """
    Encode a single canonical int8 grid (81,) and `quiet` clock into an (11, 9, 9) float32 tensor:
      - Ch 0..2: Current player R, P, S
      - Ch 3..5: Opponent R, P, S
      - Ch 6: Normalized quiet clock (quiet / 200.0)
      - Ch 7: Normalized proximity to own goal (i9)
      - Ch 8: Normalized proximity to opponent goal (a1)
      - Ch 9: Own pieces immediately threatened by adjacent enemy nemesis
      - Ch 10: Opponent pieces immediately capturable by adjacent own piece
    """
    g2d = c_grid.reshape(BOARD_SIZE, BOARD_SIZE)
    obs = np.zeros((NUM_OBS_CHANNELS_V2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    own_r = g2d == ROCK
    own_p = g2d == PAPER
    own_s = g2d == SCISSORS
    opp_r = g2d == -ROCK
    opp_p = g2d == -PAPER
    opp_s = g2d == -SCISSORS

    obs[0] = own_r
    obs[1] = own_p
    obs[2] = own_s
    obs[3] = opp_r
    obs[4] = opp_p
    obs[5] = opp_s
    obs[6].fill(min(1.0, float(quiet) / float(QUIET_LIMIT)))
    obs[7] = _GOAL_PROX_OWN
    obs[8] = _GOAL_PROX_OPP

    own_threatened = (
        (own_r & _dilate_king_grid(opp_p))
        | (own_p & _dilate_king_grid(opp_s))
        | (own_s & _dilate_king_grid(opp_r))
    )
    opp_capturable = (
        (opp_s & _dilate_king_grid(own_r))
        | (opp_r & _dilate_king_grid(own_p))
        | (opp_p & _dilate_king_grid(own_s))
    )
    obs[9] = own_threatened
    obs[10] = opp_capturable
    return obs


def encode_board_v2(board: Board) -> np.ndarray:
    """Encode `Board` into canonical 11-channel float32 tensor (11, 9, 9)."""
    c_grid = canonical_grid(board.grid, board.side)
    return encode_canonical_grid_v2(c_grid, board.quiet)


def encode_canonical_grids_v2_batch(c_grids: np.ndarray, quiets: np.ndarray) -> np.ndarray:
    """Vectorized batch encoding of canonical int8 grids (B, 81) and quiets (B,) -> (B, 11, 9, 9)."""
    bsz = c_grids.shape[0]
    g2d = c_grids.reshape(bsz, BOARD_SIZE, BOARD_SIZE)
    obs = np.zeros((bsz, NUM_OBS_CHANNELS_V2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    own_r = g2d == ROCK
    own_p = g2d == PAPER
    own_s = g2d == SCISSORS
    opp_r = g2d == -ROCK
    opp_p = g2d == -PAPER
    opp_s = g2d == -SCISSORS

    obs[:, 0] = own_r
    obs[:, 1] = own_p
    obs[:, 2] = own_s
    obs[:, 3] = opp_r
    obs[:, 4] = opp_p
    obs[:, 5] = opp_s
    q_norm = np.minimum(1.0, quiets.astype(np.float32) / float(QUIET_LIMIT))
    obs[:, 6] = q_norm[:, None, None]
    obs[:, 7] = _GOAL_PROX_OWN[None, :, :]
    obs[:, 8] = _GOAL_PROX_OPP[None, :, :]

    obs[:, 9] = (
        (own_r & _dilate_king_grid(opp_p))
        | (own_p & _dilate_king_grid(opp_s))
        | (own_s & _dilate_king_grid(opp_r))
    )
    obs[:, 10] = (
        (opp_s & _dilate_king_grid(own_r))
        | (opp_r & _dilate_king_grid(own_p))
        | (opp_p & _dilate_king_grid(own_s))
    )
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
