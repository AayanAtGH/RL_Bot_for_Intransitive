"""Fuzz and invariance tests verifying C++17 FastBoard rules and 11-ch features against Python."""

import numpy as np

from intransitive.core.board import Board
from intransitive.core.constants import BLUE, RED, START_FEN
from intransitive.core.move import action_to_uci, uci_to_action
from intransitive.env.canonical import (
    apply_symmetry_6x,
    apply_symmetry_6x_actions,
    apply_symmetry_6x_grid,
    canonical_grid,
    encode_board_v2,
    encode_canonical_grid_v2,
)
from intransitive.native.native_bridge import NativeEngine


def test_native_library_loaded() -> None:
    assert NativeEngine.is_available() is True


def test_cpp_vs_python_rules_and_features_fuzz() -> None:
    rng = np.random.default_rng(12345)
    total_plies = 0

    for game_idx in range(2000):
        board = Board(START_FEN)
        # Alternate between early-game random walks and sparse endgame probes
        max_steps = 40 if game_idx < 1500 else 120
        for _ in range(max_steps):
            py_acts = sorted(board.legal_actions())
            cpp_acts = sorted(NativeEngine.legal_actions(board.grid, board.side, board.quiet).tolist())
            assert py_acts == cpp_acts

            py_res = board.result()
            cpp_term = NativeEngine.terminal_result(board.grid, board.side, board.quiet)
            if py_res is None:
                assert cpp_term == 0
            elif py_res.winner == 0:
                assert cpp_term == 2
            else:
                assert cpp_term == py_res.winner

            py_obs = encode_board_v2(board)
            cpp_obs = NativeEngine.encode_features_11ch(board.grid, board.side, board.quiet)
            assert py_obs.shape == (11, 9, 9)
            assert cpp_obs.shape == (11, 9, 9)
            assert np.allclose(py_obs, cpp_obs, atol=1e-6)

            if py_res is not None or not py_acts:
                break
            chosen = py_acts[int(rng.integers(0, len(py_acts)))]
            board.push_action(chosen, validate=False)
            total_plies += 1

    assert total_plies > 10_000


def test_6x_symmetry_invariance() -> None:
    rng = np.random.default_rng(77)
    for _ in range(50):
        board = Board(START_FEN)
        for _ in range(int(rng.integers(0, 16))):
            acts = board.legal_actions()
            if not acts:
                break
            board.push_action(acts[int(rng.integers(0, len(acts)))], validate=False)
        if board.is_terminal():
            continue

        obs = encode_board_v2(board)
        c_grid = canonical_grid(board.grid, board.side)
        mask_orig = np.zeros(648, dtype=np.float32)
        for m in board.legal_moves():
            a = uci_to_action(m)
            from intransitive.env.canonical import to_canonical_action
            mask_orig[to_canonical_action(a, board.side)] = 1.0

        for sym_id in range(6):
            s_obs, s_mask = apply_symmetry_6x(obs, mask_orig, sym_id)
            s_grid = apply_symmetry_6x_grid(c_grid, sym_id)
            re_obs = encode_canonical_grid_v2(s_grid, board.quiet)
            assert np.allclose(s_obs, re_obs, atol=1e-6)

            # Build transformed board as Blue and verify legal actions match s_mask
            b_sym = Board(_board_array=s_grid, _side=BLUE, quiet=board.quiet, ply=board.ply)
            sym_legal = np.array(b_sym.legal_actions(), dtype=np.int64)
            expected_legal = np.flatnonzero(s_mask > 0.5)
            assert np.array_equal(np.sort(sym_legal), np.sort(expected_legal))

            # Also verify apply_symmetry_6x_actions matches
            orig_legal = np.flatnonzero(mask_orig > 0.5)
            mapped = apply_symmetry_6x_actions(orig_legal, sym_id)
            assert np.array_equal(np.sort(mapped), np.sort(expected_legal))
            assert all(len(action_to_uci(int(a))) == 4 for a in mapped)
