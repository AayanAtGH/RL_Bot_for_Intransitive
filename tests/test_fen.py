"""Tests for FEN serialization, UCI/record notation, replay, and canonical symmetries."""

from intransitive.core.board import Board, replay
from intransitive.core.constants import ACTION_SIZE, NUM_SQUARES, START_FEN
from intransitive.core.move import (
    action_to_uci,
    rotate_action,
    rotate_square,
    square_index,
    square_label,
    to_notation,
    to_uci,
    uci_to_action,
)
from intransitive.env.canonical import (
    canonical_action_mask,
    encode_board,
    from_canonical_action,
    to_canonical_action,
)


def test_start_fen_roundtrip() -> None:
    board = Board(START_FEN)
    assert board.to_fen() == START_FEN
    engine_fen = board.to_engine_fen()
    # Engine dialect writes rank 9 first and Blue as 'w'
    assert engine_fen == "9/4pr3/4spr2/5spr1/1PS3sp1/1RPS5/2RPS4/3RP4/9 w - - 0 1"


def test_square_and_action_roundtrip() -> None:
    assert square_index("a1") == 0
    assert square_index("i9") == 80
    assert square_index("d4") == 30
    for sq in range(NUM_SQUARES):
        lbl = square_label(sq)
        assert square_index(lbl) == sq
        assert rotate_square(rotate_square(sq)) == sq

    for act in range(ACTION_SIZE):
        assert rotate_action(rotate_action(act)) == act


def test_notation_and_replay() -> None:
    board = Board(START_FEN)
    assert to_notation(board.grid, "d2c1") == "Rd2-c1"
    assert to_uci("Rd2-c1") == "d2c1"
    assert to_uci("Sd5xf5") == "d5f5"

    positions = replay(START_FEN, moves=["d2c1", "f8g9"])
    assert len(positions) == 3
    assert positions[1]["notation"] == "Rd2-c1"
    assert positions[1]["quiet"] == 1
    assert positions[2]["notation"] == "Rf8-g9"
    assert positions[2]["quiet"] == 2


def test_canonical_symmetry_between_blue_and_red() -> None:
    # Starting position is 180-degree symmetric between Blue and Red!
    board_blue = Board(START_FEN)
    # Create identical position with Red to move
    red_fen = START_FEN[:-1] + "r"
    board_red = Board(red_fen)

    obs_blue = encode_board(board_blue)
    obs_red = encode_board(board_red)

    # Because START_FEN has 180-degree rotational piece symmetry (R<->r, P<->p, S<->s),
    # the canonical observation tensor for Blue and Red must be identical!
    assert (obs_blue == obs_red).all()

    mask_blue = canonical_action_mask(board_blue)
    mask_red = canonical_action_mask(board_red)
    assert (mask_blue == mask_red).all()

    # Check canonical action roundtrip
    for mv in board_red.legal_moves():
        real_a = uci_to_action(mv)
        c_a = to_canonical_action(real_a, board_red.side)
        assert mask_red[c_a]
        back_a = from_canonical_action(c_a, board_red.side)
        assert action_to_uci(back_a) == mv
