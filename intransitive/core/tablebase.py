"""Python interface to the C++17 Syzygy-style 2-4 piece Endgame Tablebase."""

from pathlib import Path
from typing import Optional, Tuple

from intransitive.core.board import Board
from intransitive.core.move import action_to_uci
from intransitive.native.native_bridge import NativeEngine


class TablebaseManager:
    """Loads and probes memory-mapped .wdl / .dtz endgame tablebases via the C++ core."""

    def __init__(self, tb_dir: Optional[str | Path] = None) -> None:
        self.tb_dir: Optional[Path] = None
        self.handle: Optional[int] = None
        self.max_pieces: int = 0
        if tb_dir is not None:
            self.load(tb_dir)

    def load(self, tb_dir: str | Path) -> bool:
        path = Path(tb_dir)
        if not path.exists() or not NativeEngine.is_available():
            return False
        self.close()
        handle, max_p = NativeEngine.tb_load(str(path))
        if handle and max_p >= 2:
            self.handle = handle
            self.max_pieces = max_p
            self.tb_dir = path
            return True
        return False

    def is_loaded(self) -> bool:
        return self.handle is not None and self.max_pieces >= 2

    def close(self) -> None:
        if self.handle is not None and NativeEngine.is_available():
            NativeEngine.tb_destroy(self.handle)
        self.handle = None
        self.max_pieces = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def probe(self, board: Board) -> Tuple[bool, int, int, int]:
        """
        Probe WDL and DTZ from `board.side`'s perspective.
        Returns: (available, wdl in {-1, 0, +1}, dtz in 0..202, raw_wdl in 0..3).
        """
        if not self.is_loaded() or self.handle is None:
            return False, 0, 0, 1
        return NativeEngine.tb_probe(self.handle, board.grid, board.side, board.quiet, board.ply)

    def probe_best_move(self, board: Board) -> Optional[str]:
        """Return exact optimal UCI move from tablebase, or None if unavailable/terminal."""
        if not self.is_loaded() or self.handle is None:
            return None
        act = NativeEngine.tb_probe_best_action(self.handle, board.grid, board.side, board.quiet, board.ply)
        if act < 0:
            return None
        return action_to_uci(act)

    def format_status(self, board: Board) -> Optional[str]:
        """Human-readable status string for GUI / CLI display."""
        avail, wdl, dtz, raw_wdl = self.probe(board)
        if not avail:
            return None
        if wdl > 0:
            return f"TB: Win in {dtz}" if dtz > 0 else "TB: Win"
        if wdl < 0:
            return f"TB: Loss in {dtz}" if dtz > 0 else "TB: Loss"
        if raw_wdl == 3:
            return "TB: Draw (200-ply)"
        return "TB: Draw"
