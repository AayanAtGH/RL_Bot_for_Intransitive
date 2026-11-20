"""Board state representation, FEN serialization, and game state management."""

from typing import List, Optional, Tuple
import numpy as np

from intransitive.core.constants import (
    BLUE,
    BOARD_SIZE,
    CHAR_TO_PIECE,
    EMPTY,
    NUM_SQUARES,
    PIECE_TO_CHAR,
    SIDE_CHAR_TO_INT,
    SIDE_INT_TO_CHAR,
    START_FEN,
)
from intransitive.core.move import action_to_squares, action_to_uci, to_notation, to_uci, uci_to_squares
from intransitive.core.rules import (
    GameResult,
    action_mask,
    game_result,
    legal_actions,
    legal_moves,
)


def parse_fen(fen: str) -> Tuple[np.ndarray, int]:
    """
    Parse record FEN dialect (rank 1 first, side 'b' or 'r').
    Example: '9/3RP4/2RPS4/1RPS5/1PS3sp1/5spr1/4spr2/4pr3/9 b'
    """
    fields = fen.strip().split()
    if len(fields) < 2 or fields[1] not in SIDE_CHAR_TO_INT:
        raise ValueError(f"Expected V6 pieces and a b/r side to move, got: {fen!r}")

    rows = fields[0].split("/")
    if len(rows) != BOARD_SIZE:
        raise ValueError(f"V6 requires {BOARD_SIZE} ranks, got {len(rows)}")

    cells: List[int] = []
    for row in rows:
        row_cells: List[int] = []
        for ch in row:
            if "1" <= ch <= "9":
                row_cells.extend([EMPTY] * int(ch))
            elif ch in CHAR_TO_PIECE and ch != ".":
                row_cells.append(CHAR_TO_PIECE[ch])
            else:
                raise ValueError(f"Invalid board character in FEN: {ch!r}")
        if len(row_cells) != BOARD_SIZE:
            raise ValueError(f"V6 requires {BOARD_SIZE} files per rank, got {len(row_cells)} in {row!r}")
        cells.extend(row_cells)

    board = np.array(cells, dtype=np.int8)
    side = SIDE_CHAR_TO_INT[fields[1]]
    return board, side


def board_to_fen_rows(board: np.ndarray) -> List[str]:
    """Convert flat 81-element board array into 9 rank FEN strings (rank 1 to rank 9)."""
    rows: List[str] = []
    for rank in range(BOARD_SIZE):
        row_str = ""
        empty_run = 0
        for file_idx in range(BOARD_SIZE):
            piece = int(board[rank * BOARD_SIZE + file_idx])
            if piece == EMPTY:
                empty_run += 1
            else:
                if empty_run > 0:
                    row_str += str(empty_run)
                    empty_run = 0
                row_str += PIECE_TO_CHAR[piece]
        if empty_run > 0:
            row_str += str(empty_run)
        rows.append(row_str)
    return rows


def to_fen(board: np.ndarray, side: int) -> str:
    """Serialize board and side to record FEN dialect (rank 1 first, 'b'/'r')."""
    rows = board_to_fen_rows(board)
    return f"{'/'.join(rows)} {SIDE_INT_TO_CHAR[side]}"


def to_engine_fen(board: np.ndarray, side: int, quiet: int = 0, fullmove: int = 1) -> str:
    """
    Serialize to Fairy-Stockfish / Hushhush engine dialect:
    Rank 9 first, Blue as 'w', Red as 'b', quiet counter in halfmove clock.
    """
    rows = list(reversed(board_to_fen_rows(board)))
    side_char = "w" if side == BLUE else "b"
    return f"{'/'.join(rows)} {side_char} - - {quiet} {fullmove}"


class Board:
    """Mutable or copyable game state for Intransitive (RPS V6)."""

    def __init__(
        self,
        fen: str = START_FEN,
        quiet: int = 0,
        ply: int = 0,
        _board_array: Optional[np.ndarray] = None,
        _side: Optional[int] = None,
    ) -> None:
        if _board_array is not None and _side is not None:
            self.grid: np.ndarray = _board_array
            self.side: int = _side
        else:
            self.grid, self.side = parse_fen(fen)
        self.quiet: int = quiet
        self.ply: int = ply

    def copy(self) -> "Board":
        """Return a fast shallow-cloned Board with a copied grid array."""
        return Board(
            quiet=self.quiet,
            ply=self.ply,
            _board_array=self.grid.copy(),
            _side=self.side,
        )

    def to_fen(self) -> str:
        """Return record dialect FEN."""
        return to_fen(self.grid, self.side)

    def to_engine_fen(self) -> str:
        """Return engine dialect FEN."""
        fullmove = (self.ply // 2) + 1
        return to_engine_fen(self.grid, self.side, self.quiet, fullmove)

    def legal_moves(self) -> List[str]:
        """Return list of legal UCI moves in the current position."""
        return legal_moves(self.grid, self.side, self.quiet)

    def legal_actions(self) -> List[int]:
        """Return list of legal flat action indices (0..647)."""
        return legal_actions(self.grid, self.side, self.quiet)

    def action_mask(self) -> np.ndarray:
        """Return boolean action mask of shape (648,)."""
        return action_mask(self.grid, self.side, self.quiet)

    def result(self) -> Optional[GameResult]:
        """Return GameResult if game is over, else None."""
        return game_result(self.grid, self.side, self.quiet)

    def is_terminal(self) -> bool:
        """Return True if game has ended."""
        return self.result() is not None

    def push_uci(self, move: str, validate: bool = True) -> bool:
        """Apply a UCI or record-notation move in-place. Returns True if capture occurred."""
        uci = to_uci(move)
        if validate and uci not in self.legal_moves():
            raise ValueError(f"Illegal V6 move: {uci} in position {self.to_fen()}")
        from_sq, to_sq = uci_to_squares(uci)
        was_capture = int(self.grid[to_sq]) != EMPTY
        self.grid[to_sq] = self.grid[from_sq]
        self.grid[from_sq] = EMPTY
        self.side = -self.side
        self.quiet = 0 if was_capture else self.quiet + 1
        self.ply += 1
        return was_capture

    def push_action(self, action: int, validate: bool = True) -> bool:
        """Apply a flat action index (0..647) in-place. Returns True if capture occurred."""
        if validate:
            uci = action_to_uci(action)
            return self.push_uci(uci, validate=True)
        from_sq, to_sq = action_to_squares(action)
        was_capture = int(self.grid[to_sq]) != EMPTY
        self.grid[to_sq] = self.grid[from_sq]
        self.grid[from_sq] = EMPTY
        self.side = -self.side
        self.quiet = 0 if was_capture else self.quiet + 1
        self.ply += 1
        return was_capture

    def state_key(self) -> Tuple[bytes, int, int]:
        """Return hashable key for transposition tables."""
        return (self.grid.tobytes(), self.side, self.quiet)

    def render_ascii(self) -> str:
        """Return human-readable ASCII board with rank 9 at top and rank 1 at bottom."""
        lines = [f"Side to move: {'Blue (R/P/S -> i9)' if self.side == BLUE else 'Red (r/p/s -> a1)'} | Quiet: {self.quiet}/{200} | Ply: {self.ply}"]
        lines.append("   +---------------------------+")
        for rank in range(BOARD_SIZE - 1, -1, -1):
            row_cells = []
            for file_idx in range(BOARD_SIZE):
                sq = rank * BOARD_SIZE + file_idx
                piece = int(self.grid[sq])
                ch = PIECE_TO_CHAR[piece]
                if sq == 0 and piece == EMPTY:
                    ch = "o"  # Red target corner a1
                elif sq == 80 and piece == EMPTY:
                    ch = "*"  # Blue target corner i9
                row_cells.append(f" {ch} ")
            lines.append(f" {rank + 1} |{''.join(row_cells)}|")
        lines.append("   +---------------------------+")
        lines.append("     a  b  c  d  e  f  g  h  i")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Board(fen={self.to_fen()!r}, quiet={self.quiet}, ply={self.ply})"


def replay(fen: str = START_FEN, moves: Optional[List[str]] = None, start_quiet: int = 0) -> List[dict]:
    """
    Replay a sequence of moves from `fen` and return a list of position dicts
    matching Hushhush `replay(fen, moves)` semantics.
    """
    board = Board(fen=fen, quiet=start_quiet)
    history = [
        {
            "board": board.grid.copy(),
            "side": SIDE_INT_TO_CHAR[board.side],
            "quiet": board.quiet,
            "move": None,
            "notation": None,
        }
    ]
    if not moves:
        return history

    for mv in moves:
        notation = to_notation(board.grid, mv)
        uci = to_uci(mv)
        capture = board.push_uci(uci, validate=True)
        history.append(
            {
                "board": board.grid.copy(),
                "side": SIDE_INT_TO_CHAR[board.side],
                "quiet": board.quiet,
                "move": uci,
                "notation": notation,
                "capture": capture,
            }
        )
    return history
