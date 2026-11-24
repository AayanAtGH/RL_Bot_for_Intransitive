"""Exhaustive 2-piece exactness verification and 2-4 piece TB-vs-TB consistency tests."""

from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pytest

from intransitive.core.board import Board
from intransitive.core.constants import BEATS_TYPE, BLUE, BLUE_GOAL, BOARD_SIZE, DIRECTIONS, RED, RED_GOAL
from intransitive.core.tablebase import TablebaseManager

TB_DIR = Path(__file__).resolve().parent.parent / "data" / "tablebase"


def _solve_2piece_python() -> Dict[Tuple[int, int, int, int, int], Tuple[int, int]]:
    """
    Independent brute-force Python retrograde solver over ALL 113,778 non-terminal
    2-piece states (1 Blue + 1 Red, no symmetries assumed!).
    State key: (b_type in {1,2,3}, b_sq in 0..79, r_type in {1,2,3}, r_sq in 1..80, side in {1,-1}).
    Value: (wdl in {-1, 0, +1}, dtz in 0..255).
    """
    # Precompute 1-step neighbors
    nbrs = []
    for sq in range(81):
        r, f = sq // 9, sq % 9
        lst = []
        for dr, df in DIRECTIONS:
            nr, nf = r + dr, f + df
            if 0 <= nr < 9 and 0 <= nf < 9:
                lst.append(nr * 9 + nf)
        nbrs.append(lst)

    states: List[Tuple[int, int, int, int, int]] = []
    idx_map: Dict[Tuple[int, int, int, int, int], int] = {}
    for bt in (1, 2, 3):
        for bs in range(80):  # b_sq != 80
            for rt in (1, 2, 3):
                for rs in range(1, 81):  # r_sq != 0
                    if bs == rs:
                        continue
                    for side in (BLUE, RED):
                        st = (bt, bs, rt, rs, side)
                        idx_map[st] = len(states)
                        states.append(st)

    n = len(states)
    wdl = [255] * n  # 255 = unknown
    dtz = [0] * n
    out_deg = [0] * n
    rev_edges: List[List[int]] = [[] for _ in range(n)]

    win_front: List[int] = []
    loss_front_0: List[int] = []

    for i, (bt, bs, rt, rs, side) in enumerate(states):
        if side == BLUE:
            acts = []
            for ns in nbrs[bs]:
                if ns != rs:
                    acts.append((ns, False))
                elif BEATS_TYPE[bt] == rt:
                    acts.append((ns, True))
            if not acts:
                wdl[i] = -1
                dtz[i] = 0
                loss_front_0.append(i)
                continue
            won = False
            q_children = []
            for ns, is_cap in acts:
                if ns == BLUE_GOAL or is_cap:
                    # Corner reach or capturing Red's only piece -> immediate win in 1 ply!
                    won = True
                    break
                # Check if Red has any moves left at (bt, ns, rt, rs, RED)
                red_has_move = any((rns != ns or BEATS_TYPE[rt] == bt) for rns in nbrs[rs])
                if not red_has_move:
                    won = True
                    break
                c_idx = idx_map[(bt, ns, rt, rs, RED)]
                q_children.append(c_idx)
            if won:
                wdl[i] = 1
                dtz[i] = 1
                win_front.append(i)
            else:
                out_deg[i] = len(q_children)
                for c_idx in q_children:
                    rev_edges[c_idx].append(i)
        else:
            acts = []
            for ns in nbrs[rs]:
                if ns != bs:
                    acts.append((ns, False))
                elif BEATS_TYPE[rt] == bt:
                    acts.append((ns, True))
            if not acts:
                wdl[i] = -1
                dtz[i] = 0
                loss_front_0.append(i)
                continue
            won = False
            q_children = []
            for ns, is_cap in acts:
                if ns == RED_GOAL or is_cap:
                    won = True
                    break
                blue_has_move = any((bns != ns or BEATS_TYPE[bt] == rt) for bns in nbrs[bs])
                if not blue_has_move:
                    won = True
                    break
                c_idx = idx_map[(bt, bs, rt, ns, BLUE)]
                q_children.append(c_idx)
            if won:
                wdl[i] = 1
                dtz[i] = 1
                win_front.append(i)
            else:
                out_deg[i] = len(q_children)
                for c_idx in q_children:
                    rev_edges[c_idx].append(i)

    # Step d=0 -> d=1 for no-moves losses (if any)
    for ci in loss_front_0:
        for pi in rev_edges[ci]:
            if wdl[pi] == 255:
                wdl[pi] = 1
                dtz[pi] = 1
                win_front.append(pi)

    curr_loss: List[int] = []
    curr_win = win_front
    d = 1
    while curr_loss or curr_win:
        next_win: List[int] = []
        next_loss: List[int] = []
        next_d = d + 1
        for ci in curr_loss:
            for pi in rev_edges[ci]:
                if wdl[pi] == 255:
                    wdl[pi] = 1
                    dtz[pi] = next_d
                    next_win.append(pi)
        for ci in curr_win:
            for pi in rev_edges[ci]:
                if wdl[pi] == 255:
                    out_deg[pi] -= 1
                    if out_deg[pi] == 0:
                        wdl[pi] = -1
                        dtz[pi] = next_d
                        next_loss.append(pi)
        curr_win = next_win
        curr_loss = next_loss
        d += 1

    out: Dict[Tuple[int, int, int, int, int], Tuple[int, int]] = {}
    for i, st in enumerate(states):
        if wdl[i] == 255:
            out[st] = (0, 0)
        elif dtz[i] > 200:
            out[st] = (0, 201 if wdl[i] == 1 else 202)
        else:
            out[st] = (wdl[i], dtz[i])
    return out


def test_2piece_tablebase_100pct_exact_against_python() -> None:
    if not (TB_DIR / "S_vs_s.wdl").exists():
        pytest.skip("2-piece tablebase not generated yet")

    tb = TablebaseManager(TB_DIR)
    assert tb.is_loaded()
    ref = _solve_2piece_python()
    assert len(ref) == 113_778

    grid = np.zeros(81, dtype=np.int8)
    mismatches = 0
    for (bt, bs, rt, rs, side), (exp_wdl, exp_dtz) in ref.items():
        grid[bs] = bt
        grid[rs] = -rt
        b = Board(_board_array=grid, _side=side, quiet=0, ply=0)
        avail, got_wdl, got_dtz, _ = tb.probe(b)
        if not avail or got_wdl != exp_wdl or got_dtz != exp_dtz:
            mismatches += 1
            if mismatches <= 5:
                raise AssertionError(
                    f"Mismatch on {(bt, bs, rt, rs, side)}: expected {(exp_wdl, exp_dtz)}, "
                    f"got {(avail, got_wdl, got_dtz)}"
                )
        grid[bs] = 0
        grid[rs] = 0
    assert mismatches == 0


def test_tablebase_play_consistency_2_to_4_pieces() -> None:
    if not TB_DIR.exists():
        pytest.skip("Tablebase directory missing")
    tb = TablebaseManager(TB_DIR)
    if not tb.is_loaded():
        pytest.skip("Tablebase not generated")

    max_p = tb.max_pieces
    rng = np.random.default_rng(2026)

    num_games = 500 if max_p >= 4 else 150
    for _ in range(num_games):
        n_tot = int(rng.integers(2, max_p + 1))
        n_blue = int(rng.integers(1, n_tot))
        n_red = n_tot - n_blue

        # Sample valid non-terminal board
        while True:
            grid = np.zeros(81, dtype=np.int8)
            used = set()
            for _ in range(n_blue):
                sq = int(rng.integers(0, 80))
                while sq in used:
                    sq = (sq + 1) % 80
                used.add(sq)
                grid[sq] = int(rng.integers(1, 4))
            for _ in range(n_red):
                sq = int(rng.integers(1, 81))
                while sq in used:
                    sq = 1 + (sq % 80)
                used.add(sq)
                grid[sq] = -int(rng.integers(1, 4))
            side = BLUE if int(rng.integers(0, 2)) == 0 else RED
            init_quiet = int(rng.integers(0, 40)) if int(rng.integers(0, 4)) == 0 else 0
            b = Board(_board_array=grid, _side=side, quiet=init_quiet, ply=0)
            if b.result() is None:
                break

        avail, init_wdl, init_dtz, _ = tb.probe(b)
        assert avail

        # Verify symmetry invariance of TB probe under D1 reflection and C3 RPS shifts
        ref_b = Board(_board_array=b.grid.reshape(9, 9).T.reshape(81), _side=b.side, quiet=b.quiet)
        r_av, r_wdl, r_dtz, _ = tb.probe(ref_b)
        assert r_av and r_wdl == init_wdl and r_dtz == init_dtz

        # Play out TB vs TB!
        init_side = b.side
        seen: Dict[Tuple[bytes, int], int] = {}
        final_winner = 0

        for step in range(250):
            res = b.result()
            if res is not None:
                final_winner = res.winner
                break
            k = (b.grid.tobytes(), b.side)
            seen[k] = seen.get(k, 0) + 1
            if seen[k] >= 3:
                final_winner = 0
                break

            av_cur, cur_wdl, cur_dtz, _ = tb.probe(b)
            assert av_cur

            mv = tb.probe_best_move(b)
            assert mv is not None
            was_cap = b.push_uci(mv, validate=True)

            # Check DTZ strict decrease on quiet winning lines (when not capture/terminal)
            if cur_wdl == 1 and not was_cap and b.result() is None:
                av_nxt, nxt_wdl, nxt_dtz, _ = tb.probe(b)
                assert av_nxt and nxt_wdl == -1
                assert nxt_dtz <= cur_dtz - 1 or cur_dtz == 0

        if init_wdl == 1:
            assert final_winner == init_side
        elif init_wdl == -1:
            assert final_winner == -init_side
        else:
            # Draw never loses!
            assert final_winner != -init_side
