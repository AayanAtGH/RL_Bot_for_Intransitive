"""Comprehensive unit tests for Intransitive rules, captures, and terminal conditions."""

import pytest

from intransitive.core.board import Board
from intransitive.core.constants import BLUE, QUIET_LIMIT, RED, START_FEN
from intransitive.core.rules import game_result, legal_moves


def test_start_position_piece_counts_and_moves() -> None:
    board = Board(START_FEN)
    assert board.side == BLUE
    # Blue has 3 R, 4 P, 3 S = 10 pieces; Red has 10 pieces
    assert int((board.grid > 0).sum()) == 10
    assert int((board.grid < 0).sum()) == 10
    moves = board.legal_moves()
    assert len(moves) > 0
    assert board.result() is None


def test_rps_capture_hierarchy() -> None:
    # Place Blue R at e5, surrounded by:
    # - Red s at e6 (capturable: R beats s)
    # - Red r at d5 (blocked: R == r)
    # - Red p at f5 (blocked: p beats R)
    # - Blue S at e4 (blocked: friendly piece)
    fen = "9/9/9/4S4/3rRp3/4s4/9/9/9 b"
    board = Board(fen)
    moves = board.legal_moves()

    # From e5:
    e5_moves = [m for m in moves if m.startswith("e5")]
    # Can capture e6 (s)
    assert "e5e6" in e5_moves
    # Cannot move onto d5 (r), f5 (p), or e4 (S)
    assert "e5d5" not in e5_moves
    assert "e5f5" not in e5_moves
    assert "e5e4" not in e5_moves
    # Can move to empty diagonals d6, f6, d4, f4
    assert "e5d6" in e5_moves
    assert "e5f6" in e5_moves


def test_capture_resets_quiet_clock() -> None:
    fen = "9/9/9/9/4R4/4s4/9/9/9 b"
    board = Board(fen, quiet=45)
    was_cap = board.push_uci("e5e6")
    assert was_cap is True
    assert board.quiet == 0
    assert board.side == RED


def test_blue_corner_victory() -> None:
    # Blue R at h8 steps to i9
    fen = "9/9/2r6/9/9/9/9/7R1/9 b"
    board = Board(fen)
    assert board.result() is None
    board.push_uci("h8i9")
    res = board.result()
    assert res is not None
    assert res.winner == BLUE
    assert res.reason == "corner"
    assert board.legal_moves() == []


def test_red_corner_victory() -> None:
    # Red s at b2 steps to a1
    fen = "9/1s7/9/9/9/9/6R2/9/9 r"
    board = Board(fen)
    assert board.result() is None
    board.push_uci("b2a1")
    res = board.result()
    assert res is not None
    assert res.winner == RED
    assert res.reason == "corner"
    assert board.legal_moves() == []


def test_no_moves_loss() -> None:
    # Blue has a single Rock at a9 (top-left, index 72) completely surrounded by
    # Red rocks/papers at b9, a8, b8 so Blue has 0 legal moves
    fen = "9/9/9/9/9/9/9/rp7/Rp7 b"
    board = Board(fen)
    assert legal_moves(board.grid, board.side, board.quiet) == []
    res = board.result()
    assert res is not None
    assert res.winner == RED
    assert res.reason == "no-moves"


def test_quiet_limit_draw_and_corner_precedence() -> None:
    board = Board(START_FEN, quiet=QUIET_LIMIT)
    res = board.result()
    assert res is not None
    assert res.winner == 0
    assert res.reason == "quiet-limit"

    # Corner win takes precedence over quiet == 200
    win_fen = "9/9/2r6/9/9/9/9/9/8R r"
    win_board = Board(win_fen, quiet=QUIET_LIMIT)
    win_res = game_result(win_board.grid, win_board.side, win_board.quiet)
    assert win_res is not None
    assert win_res.winner == BLUE
    assert win_res.reason == "corner"


def test_illegal_move_raises() -> None:
    board = Board(START_FEN)
    with pytest.raises(ValueError):
        board.push_uci("a1a2")  # No piece on a1
