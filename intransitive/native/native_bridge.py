"""Zero-copy ctypes wrapper around libintransitive_core (.dll / .so)."""

import ctypes
from pathlib import Path
import platform
from typing import Dict, Optional, Tuple
import numpy as np

from intransitive.core.constants import ACTION_SIZE, NUM_SQUARES


def _find_library() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    lib_name = "libintransitive_core.dll" if platform.system() == "Windows" else "libintransitive_core.so"
    cand = here / lib_name
    return cand if cand.exists() else None


class NativeEngine:
    """Singleton ctypes binding to the compiled C++17 bitboard, TB, and MCTS engine."""

    _lib: Optional[ctypes.CDLL] = None
    _load_attempted: bool = False

    @classmethod
    def _ensure_loaded(cls) -> Optional[ctypes.CDLL]:
        if cls._lib is not None:
            return cls._lib
        if cls._load_attempted:
            return None
        cls._load_attempted = True
        lib_path = _find_library()
        if lib_path is None:
            return None

        lib = ctypes.CDLL(str(lib_path))
        c_i8_p = ctypes.POINTER(ctypes.c_int8)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)
        c_u16_p = ctypes.POINTER(ctypes.c_uint16)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)
        c_u64_p = ctypes.POINTER(ctypes.c_uint64)
        c_f32_p = ctypes.POINTER(ctypes.c_float)
        c_f64_p = ctypes.POINTER(ctypes.c_double)
        c_vp = ctypes.c_void_p

        # Board helpers
        lib.core_legal_actions.argtypes = [c_i8_p, ctypes.c_int32, ctypes.c_int32, c_u16_p]
        lib.core_legal_actions.restype = ctypes.c_int32

        lib.core_terminal_result.argtypes = [c_i8_p, ctypes.c_int32, ctypes.c_int32]
        lib.core_terminal_result.restype = ctypes.c_int32

        lib.core_encode_features_11ch.argtypes = [c_i8_p, ctypes.c_int32, ctypes.c_int32, c_f32_p]
        lib.core_encode_features_11ch.restype = None

        lib.core_prove_corner_win_3ply.argtypes = [c_i8_p, ctypes.c_int32, ctypes.c_int32]
        lib.core_prove_corner_win_3ply.restype = ctypes.c_int32

        # Tablebase API
        lib.tb_create_and_load.argtypes = [ctypes.c_char_p, c_i32_p]
        lib.tb_create_and_load.restype = c_vp

        lib.tb_destroy.argtypes = [c_vp]
        lib.tb_destroy.restype = None

        lib.tb_probe_state.argtypes = [
            c_vp, c_i8_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, c_i32_p, c_i32_p, c_i32_p
        ]
        lib.tb_probe_state.restype = ctypes.c_int32

        lib.tb_probe_best_action.argtypes = [
            c_vp, c_i8_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
        ]
        lib.tb_probe_best_action.restype = ctypes.c_int32

        # Single-game MCTS
        lib.mcts_create.argtypes = [
            c_vp, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.c_uint64,
        ]
        lib.mcts_create.restype = c_vp

        lib.mcts_destroy.argtypes = [c_vp]
        lib.mcts_destroy.restype = None

        lib.mcts_set_tablebase.argtypes = [c_vp, c_vp]
        lib.mcts_set_tablebase.restype = None

        lib.mcts_start_search.argtypes = [
            c_vp, c_i8_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            c_u64_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ]
        lib.mcts_start_search.restype = ctypes.c_int32

        lib.mcts_advance_root.argtypes = [c_vp, ctypes.c_uint16]
        lib.mcts_advance_root.restype = None

        lib.mcts_collect_batch.argtypes = [
            c_vp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, c_f32_p, c_u8_p, c_i32_p
        ]
        lib.mcts_collect_batch.restype = ctypes.c_int32

        lib.mcts_apply_batch.argtypes = [
            c_vp, c_f32_p, c_f32_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
        ]
        lib.mcts_apply_batch.restype = None

        lib.mcts_get_result.argtypes = [
            c_vp, ctypes.c_float, c_f32_p, c_f64_p, c_i32_p, c_i32_p, c_i32_p, c_i32_p, c_i32_p
        ]
        lib.mcts_get_result.restype = None

        # Multi-game Batched Self-Play
        lib.sp_create.argtypes = [
            c_vp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
            ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_uint64,
        ]
        lib.sp_create.restype = c_vp

        lib.sp_destroy.argtypes = [c_vp]
        lib.sp_destroy.restype = None

        lib.sp_clear_cache.argtypes = [c_vp]
        lib.sp_clear_cache.restype = None

        lib.sp_collect_batch.argtypes = [c_vp, c_f32_p, c_u8_p, ctypes.c_int32]
        lib.sp_collect_batch.restype = ctypes.c_int32

        lib.sp_apply_evals.argtypes = [c_vp, c_f32_p, c_f32_p, ctypes.c_int32]
        lib.sp_apply_evals.restype = None

        lib.sp_finished_sample_count.argtypes = [c_vp]
        lib.sp_finished_sample_count.restype = ctypes.c_int32

        lib.sp_pop_finished_samples.argtypes = [
            c_vp, c_i8_p, c_u8_p, c_u16_p, c_f32_p, c_u8_p, c_u8_p, c_u8_p, ctypes.c_int32
        ]
        lib.sp_pop_finished_samples.restype = ctypes.c_int32

        lib.sp_get_stats.argtypes = [c_vp, c_u64_p]
        lib.sp_get_stats.restype = None

        cls._lib = lib
        return cls._lib

    @classmethod
    def is_available(cls) -> bool:
        return cls._ensure_loaded() is not None

    @classmethod
    def require(cls) -> ctypes.CDLL:
        lib = cls._ensure_loaded()
        if lib is None:
            raise RuntimeError(
                "Native C++ library not found. Build it first with: python scripts/build_cpp.py"
            )
        return lib

    @classmethod
    def _grid_ptr(cls, grid: np.ndarray):
        arr = np.ascontiguousarray(grid, dtype=np.int8)
        return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))

    @classmethod
    def legal_actions(cls, grid: np.ndarray, side: int, quiet: int) -> np.ndarray:
        lib = cls.require()
        arr, ptr = cls._grid_ptr(grid)
        out = np.empty(80, dtype=np.uint16)
        n = lib.core_legal_actions(ptr, int(side), int(quiet), out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
        _ = arr
        return out[:n]

    @classmethod
    def terminal_result(cls, grid: np.ndarray, side: int, quiet: int) -> int:
        lib = cls.require()
        arr, ptr = cls._grid_ptr(grid)
        res = int(lib.core_terminal_result(ptr, int(side), int(quiet)))
        _ = arr
        return res

    @classmethod
    def encode_features_11ch(cls, grid: np.ndarray, side: int, quiet: int) -> np.ndarray:
        lib = cls.require()
        arr, ptr = cls._grid_ptr(grid)
        out = np.empty((11, 9, 9), dtype=np.float32)
        lib.core_encode_features_11ch(
            ptr, int(side), int(quiet), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        _ = arr
        return out

    @classmethod
    def prove_corner_win_3ply(cls, grid: np.ndarray, side: int, quiet: int) -> int:
        lib = cls.require()
        arr, ptr = cls._grid_ptr(grid)
        act = int(lib.core_prove_corner_win_3ply(ptr, int(side), int(quiet)))
        _ = arr
        return act

    @classmethod
    def tb_load(cls, tb_dir: str) -> Tuple[Optional[int], int]:
        lib = cls.require()
        max_p = ctypes.c_int32(0)
        handle = lib.tb_create_and_load(tb_dir.encode("utf-8"), ctypes.byref(max_p))
        return (int(handle) if handle else None), int(max_p.value)

    @classmethod
    def tb_destroy(cls, handle: int) -> None:
        lib = cls.require()
        lib.tb_destroy(ctypes.c_void_p(handle))

    @classmethod
    def tb_probe(cls, handle: int, grid: np.ndarray, side: int, quiet: int, ply: int) -> Tuple[bool, int, int, int]:
        lib = cls.require()
        arr, ptr = cls._grid_ptr(grid)
        wdl = ctypes.c_int32(0)
        dtz = ctypes.c_int32(0)
        raw_wdl = ctypes.c_int32(1)
        ok = lib.tb_probe_state(
            ctypes.c_void_p(handle),
            ptr,
            int(side),
            int(quiet),
            int(ply),
            ctypes.byref(wdl),
            ctypes.byref(dtz),
            ctypes.byref(raw_wdl),
        )
        _ = arr
        return bool(ok), int(wdl.value), int(dtz.value), int(raw_wdl.value)

    @classmethod
    def tb_probe_best_action(cls, handle: int, grid: np.ndarray, side: int, quiet: int, ply: int) -> int:
        lib = cls.require()
        arr, ptr = cls._grid_ptr(grid)
        act = int(lib.tb_probe_best_action(ctypes.c_void_p(handle), ptr, int(side), int(quiet), int(ply)))
        _ = arr
        return act

    @classmethod
    def mcts_create(
        cls,
        tb_handle: Optional[int],
        c_puct: float,
        fpu_reduction: float,
        root_fpu_reduction: float,
        dirichlet_alpha: float,
        dirichlet_eps: float,
        use_tactical_prover: bool,
        seed: int,
    ) -> int:
        lib = cls.require()
        h = lib.mcts_create(
            ctypes.c_void_p(tb_handle) if tb_handle else None,
            float(c_puct),
            float(fpu_reduction),
            float(root_fpu_reduction),
            float(dirichlet_alpha),
            float(dirichlet_eps),
            1 if use_tactical_prover else 0,
            int(seed),
        )
        return int(h)

    @classmethod
    def mcts_destroy(cls, handle: int) -> None:
        lib = cls.require()
        lib.mcts_destroy(ctypes.c_void_p(handle))

    @classmethod
    def mcts_set_tablebase(cls, handle: int, tb_handle: Optional[int]) -> None:
        lib = cls.require()
        lib.mcts_set_tablebase(
            ctypes.c_void_p(handle), ctypes.c_void_p(tb_handle) if tb_handle else None
        )

    @classmethod
    def mcts_start_search(
        cls,
        handle: int,
        grid: np.ndarray,
        side: int,
        quiet: int,
        ply: int,
        try_reuse: bool,
        add_noise: bool,
        history_hashes: Optional[np.ndarray] = None,
    ) -> bool:
        lib = cls.require()
        arr, ptr = cls._grid_ptr(grid)
        hist_ptr = None
        nh = 0
        if history_hashes is not None and len(history_hashes) > 0:
            harr = np.ascontiguousarray(history_hashes, dtype=np.uint64)
            hist_ptr = harr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
            nh = len(harr)
        reused = lib.mcts_start_search(
            ctypes.c_void_p(handle),
            ptr,
            int(side),
            int(quiet),
            int(ply),
            hist_ptr,
            int(nh),
            1 if try_reuse else 0,
            1 if add_noise else 0,
        )
        _ = arr
        return bool(reused)

    @classmethod
    def mcts_advance_root(cls, handle: int, canon_action: int) -> None:
        lib = cls.require()
        lib.mcts_advance_root(ctypes.c_void_p(handle), int(canon_action))

    @classmethod
    def mcts_collect_batch(
        cls,
        handle: int,
        max_batch: int,
        sims_remaining: int,
        add_noise: bool,
        obs_buf: np.ndarray,
        mask_buf: np.ndarray,
    ) -> Tuple[int, int]:
        lib = cls.require()
        resolved = ctypes.c_int32(0)
        cnt = lib.mcts_collect_batch(
            ctypes.c_void_p(handle),
            int(max_batch),
            int(sims_remaining),
            1 if add_noise else 0,
            obs_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mask_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(resolved),
        )
        return int(cnt), int(resolved.value)

    @classmethod
    def mcts_apply_batch(
        cls,
        handle: int,
        probs: np.ndarray,
        vals: np.ndarray,
        batch_count: int,
        used_vl: bool,
        add_noise: bool,
    ) -> None:
        lib = cls.require()
        p_arr = np.ascontiguousarray(probs, dtype=np.float32)
        v_arr = np.ascontiguousarray(vals, dtype=np.float32)
        lib.mcts_apply_batch(
            ctypes.c_void_p(handle),
            p_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(batch_count),
            1 if used_vl else 0,
            1 if add_noise else 0,
        )

    @classmethod
    def mcts_get_result(cls, handle: int, temperature: float) -> Tuple[np.ndarray, float, int, int, int, int, int]:
        lib = cls.require()
        pol = np.zeros(ACTION_SIZE, dtype=np.float32)
        q = ctypes.c_double(0.0)
        visits = ctypes.c_int32(0)
        pv_depth = ctypes.c_int32(0)
        proven = ctypes.c_int32(0)
        t1 = ctypes.c_int32(0)
        t2 = ctypes.c_int32(0)
        lib.mcts_get_result(
            ctypes.c_void_p(handle),
            float(temperature),
            pol.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(q),
            ctypes.byref(visits),
            ctypes.byref(pv_depth),
            ctypes.byref(proven),
            ctypes.byref(t1),
            ctypes.byref(t2),
        )
        return pol, float(q.value), int(visits.value), int(pv_depth.value), int(proven.value), int(t1.value), int(t2.value)

    # ---- Batched Self-Play ----

    @classmethod
    def sp_create(
        cls,
        tb_handle: Optional[int],
        num_games: int,
        full_simulations: int,
        fast_simulations: int,
        full_search_prob: float,
        temp_threshold_ply: int,
        c_puct: float,
        fpu_reduction: float,
        root_fpu_reduction: float,
        dirichlet_alpha: float,
        dirichlet_eps: float,
        seed: int,
    ) -> int:
        lib = cls.require()
        h = lib.sp_create(
            ctypes.c_void_p(tb_handle) if tb_handle else None,
            int(num_games),
            int(full_simulations),
            int(fast_simulations),
            float(full_search_prob),
            int(temp_threshold_ply),
            float(c_puct),
            float(fpu_reduction),
            float(root_fpu_reduction),
            float(dirichlet_alpha),
            float(dirichlet_eps),
            int(seed),
        )
        return int(h)

    @classmethod
    def sp_destroy(cls, handle: int) -> None:
        lib = cls.require()
        lib.sp_destroy(ctypes.c_void_p(handle))

    @classmethod
    def sp_clear_cache(cls, handle: int) -> None:
        lib = cls.require()
        lib.sp_clear_cache(ctypes.c_void_p(handle))

    @classmethod
    def sp_collect_batch(cls, handle: int, obs_buf: np.ndarray, mask_buf: np.ndarray, max_batch: int) -> int:
        lib = cls.require()
        return int(
            lib.sp_collect_batch(
                ctypes.c_void_p(handle),
                obs_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                mask_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                int(max_batch),
            )
        )

    @classmethod
    def sp_apply_evals(cls, handle: int, probs: np.ndarray, vals: np.ndarray, batch_count: int) -> None:
        lib = cls.require()
        p_arr = np.ascontiguousarray(probs, dtype=np.float32)
        v_arr = np.ascontiguousarray(vals, dtype=np.float32)
        lib.sp_apply_evals(
            ctypes.c_void_p(handle),
            p_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(batch_count),
        )

    @classmethod
    def sp_finished_sample_count(cls, handle: int) -> int:
        lib = cls.require()
        return int(lib.sp_finished_sample_count(ctypes.c_void_p(handle)))

    @classmethod
    def sp_pop_finished_samples(cls, handle: int, max_samples: int) -> Dict[str, np.ndarray]:
        lib = cls.require()
        avail = cls.sp_finished_sample_count(handle)
        n = min(max_samples, avail)
        if n <= 0:
            return {}
        c_grids = np.empty((n, NUM_SQUARES), dtype=np.int8)
        quiets = np.empty((n,), dtype=np.uint8)
        pol_acts = np.empty((n, 80), dtype=np.uint16)
        pol_probs = np.empty((n, 80), dtype=np.float32)
        pol_lens = np.empty((n,), dtype=np.uint8)
        wdl = np.empty((n,), dtype=np.uint8)
        train_pol = np.empty((n,), dtype=np.uint8)

        popped = lib.sp_pop_finished_samples(
            ctypes.c_void_p(handle),
            c_grids.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            quiets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            pol_acts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            pol_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pol_lens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            wdl.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            train_pol.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(n),
        )
        return {
            "c_grids": c_grids[:popped],
            "quiets": quiets[:popped],
            "pol_acts": pol_acts[:popped],
            "pol_probs": pol_probs[:popped],
            "pol_lens": pol_lens[:popped],
            "wdl": wdl[:popped],
            "train_pol": train_pol[:popped],
        }

    @classmethod
    def sp_get_stats(cls, handle: int) -> Dict[str, int]:
        lib = cls.require()
        arr = np.zeros(11, dtype=np.uint64)
        lib.sp_get_stats(ctypes.c_void_p(handle), arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
        return {
            "games_finished": int(arr[0]),
            "total_plies": int(arr[1]),
            "blue_wins": int(arr[2]),
            "red_wins": int(arr[3]),
            "draws": int(arr[4]),
            "tb_terminations": int(arr[5]),
            "tb_node_hits": int(arr[6]),
            "prover_hits": int(arr[7]),
            "cache_hits": int(arr[8]),
            "nn_evals": int(arr[9]),
            "avg_pv_depth": int(arr[10]),
        }
