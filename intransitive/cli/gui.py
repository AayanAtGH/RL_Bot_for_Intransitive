"""Tkinter graphical interface matching the visual appearance and styling of Intransitive (meaf.us)."""

import argparse
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image, ImageTk
import torch

from intransitive.baselines.base import Player
from intransitive.baselines.greedy_player import GreedyPlayer
from intransitive.baselines.minimax_player import MinimaxPlayer
from intransitive.baselines.random_player import RandomPlayer
from intransitive.core.board import Board
from intransitive.core.constants import (
    BLUE,
    BLUE_GOAL,
    BOARD_SIZE,
    EMPTY,
    NUM_SQUARES,
    PAPER,
    QUIET_LIMIT,
    RED,
    RED_GOAL,
    ROCK,
    SCISSORS,
    START_FEN,
)
from intransitive.core.move import (
    action_to_uci,
    square_label,
    to_notation,
    uci_to_squares,
)
from intransitive.core.tablebase import TablebaseManager
from intransitive.env.canonical import from_canonical_action
from intransitive.rl.device import resolve_device
from intransitive.rl.mcts import AlphaZeroPlayer
from intransitive.rl.network import IntransitiveNet, load_checkpoint

# Visual theme matching meaf.us/rps2
THEMES: Dict[str, Dict[str, str]] = {
    "light": {
        "page_bg": "#f4f4f4",
        "surface": "#ffffff",
        "surface_muted": "#f7f7f7",
        "card_border": "#222222",
        "grid_border": "#333333",
        "cell_bg": "#ffffff",
        "base_blue": "#6f8fe8",       # a1 Blue home base in meaf.us
        "base_red": "#ea8080",        # i9 Red home base in meaf.us
        "last_move": "#fff7ce",       # Pale yellow last move highlight
        "selected_border": "#f4c542", # Amber/gold selected outline
        "capturable_border": "#d94b4b",
        "move_dot": "#555555",
        "ink": "#1a1a1a",
        "ink_muted": "#666666",
        "blue_color": "#4b6fd9",
        "red_color": "#d94b4b",
        "eval_bg": "#e2e8f0",
        "eval_blue": "#4b6fd9",
        "eval_red": "#d94b4b",
    },
    "dark": {
        "page_bg": "#242a2e",
        "surface": "#323842",
        "surface_muted": "#3b424e",
        "card_border": "#171a1f",
        "grid_border": "#171a1f",
        "cell_bg": "#3b4049",
        "base_blue": "#354f85",
        "base_red": "#6d3838",
        "last_move": "#6a5e2b",
        "selected_border": "#e8b839",
        "capturable_border": "#ff5d5d",
        "move_dot": "#cbd5e0",
        "ink": "#f0f4f8",
        "ink_muted": "#a0aec0",
        "blue_color": "#5983ff",
        "red_color": "#ff5d5d",
        "eval_bg": "#1f242c",
        "eval_blue": "#5983ff",
        "eval_red": "#ff5d5d",
    },
}

PLAYER_CHOICES: List[str] = [
    "Human",
    "AlphaZero (Best)",
    "AlphaZero (Latest)",
    "Minimax (d=2)",
    "GreedyBot",
    "RandomBot",
]


class IntransitiveGUI:
    """Tkinter GUI mirroring the original meaf.us/rps2 board aesthetics."""

    CELL_SIZE = 60
    BOARD_PIXELS = CELL_SIZE * BOARD_SIZE
    EVAL_BAR_WIDTH = 20

    def __init__(self, root: tk.Tk, default_red: str = "AlphaZero (Best)") -> None:
        self.root = root
        self.root.title("Intransitive (RPS V6) — meaf.us Style")
        self.root.resizable(False, False)

        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.device = resolve_device("auto")
        self.current_theme_name = "light"
        self.theme = THEMES[self.current_theme_name]

        # Game state
        self.board = Board(START_FEN)
        self.board_history: List[Board] = [self.board.copy()]
        self.move_log: List[Tuple[str, str, str]] = []  # (ply, side, notation)
        self.selected_sq: Optional[int] = None
        self.legal_dest_map: Dict[int, str] = {}
        self.last_move_squares: Set[int] = set()
        self.flipped: bool = False
        self.ai_thinking: bool = False
        self.game_generation: int = 0
        self.current_blue_win_pct: float = 50.0

        # Cached neural networks, tablebase, and agents
        self._loaded_nets: Dict[str, torch.nn.Module] = {}
        self.tablebase: Optional[TablebaseManager] = None
        tb_dir = self.root_dir / "data/tablebase"
        if tb_dir.exists():
            tb = TablebaseManager(tb_dir)
            if tb.is_loaded():
                self.tablebase = tb
        self.blue_agent: Optional[Player] = None
        self.red_agent: Optional[Player] = None

        # Load piece graphics
        self.piece_images: Dict[Tuple[str, int], ImageTk.PhotoImage] = {}
        self._load_piece_graphics()

        self._build_ui(default_red)
        self.new_game()

    def _find_pieces_dir(self) -> Optional[Path]:
        """Locate the 2D piece graphics directory across common installation layouts."""
        candidates = [
            Path(__file__).resolve().parent.parent / "assets" / "pieces",
            Path(__file__).resolve().parent.parent.parent / "assets" / "pieces",
            self.root_dir / "assets" / "pieces",
            self.root_dir / "intransitive" / "assets" / "pieces",
            Path("assets/pieces").resolve(),
            Path("intransitive/assets/pieces").resolve(),
        ]
        for c in candidates:
            if c.exists() and (c / "blue_rock.png").exists():
                return c
        return None

    def _load_piece_graphics(self) -> None:
        """Load and resize transparent 2D piece artwork (Rock, Paper, Scissors)."""
        pieces_dir = self._find_pieces_dir()
        if pieces_dir is None:
            return

        img_size = int(self.CELL_SIZE * 0.82)
        mapping = {
            ("blue", ROCK): "blue_rock.png",
            ("blue", PAPER): "blue_paper.png",
            ("blue", SCISSORS): "blue_scissors.png",
            ("red", ROCK): "red_rock.png",
            ("red", PAPER): "red_paper.png",
            ("red", SCISSORS): "red_scissors.png",
        }

        for (side, p_type), fname in mapping.items():
            fpath = pieces_dir / fname
            if fpath.exists():
                try:
                    pil_im = Image.open(fpath).convert("RGBA")
                    pil_im = pil_im.resize((img_size, img_size), Image.Resampling.LANCZOS)
                    self.piece_images[(side, p_type)] = ImageTk.PhotoImage(pil_im)
                except Exception as exc:
                    print(f"Warning: Failed to load piece image {fpath}: {exc}")

    def _load_alphazero(self, ckpt_rel_paths: List[str], sims: int) -> AlphaZeroPlayer:
        ckpt_path = self.root_dir / ckpt_rel_paths[0]
        for rel in ckpt_rel_paths:
            cand = self.root_dir / rel
            if cand.exists():
                ckpt_path = cand
                break

        key = str(ckpt_path)
        if key not in self._loaded_nets:
            if ckpt_path.exists():
                net, _ = load_checkpoint(ckpt_path, device=self.device, prefer_ema=True)
            else:
                net = IntransitiveNet().to(self.device)
                net.eval()
            self._loaded_nets[key] = net

        return AlphaZeroPlayer(
            network=self._loaded_nets[key],  # type: ignore[arg-type]
            num_simulations=sims,
            temperature=0.0,
            device=self.device,
            tablebase=self.tablebase,
            min_pv_depth=6,
            name=f"AlphaZero ({ckpt_path.stem})",
        )

    def _create_player(self, choice: str) -> Optional[Player]:
        if choice == "Human":
            return None
        sims = int(self.sims_var.get())
        if choice == "AlphaZero (Best)":
            return self._load_alphazero(
                ["checkpoints/alphazero_v2_best.pt", "checkpoints/alphazero_best.pt", "checkpoints/alphazero_latest.pt"],
                sims,
            )
        if choice == "AlphaZero (Latest)":
            return self._load_alphazero(
                ["checkpoints/alphazero_v2_latest.pt", "checkpoints/alphazero_latest.pt"],
                sims,
            )
        if choice == "Minimax (d=2)":
            return MinimaxPlayer(depth=2)
        if choice == "GreedyBot":
            return GreedyPlayer()
        if choice == "RandomBot":
            return RandomPlayer()
        return None

    def _build_ui(self, default_red: str) -> None:
        th = self.theme
        self.root.configure(bg=th["page_bg"])

        self.main_container = tk.Frame(self.root, bg=th["page_bg"], padx=18, pady=16)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # -------------------------------------------------------------
        # Left Area: Material counter (top), Board + Eval Bar, Material counter (bottom)
        # -------------------------------------------------------------
        left_box = tk.Frame(self.main_container, bg=th["page_bg"])
        left_box.pack(side=tk.LEFT, padx=(0, 20), anchor="n")

        # Top player header (Red by default, or Blue if flipped)
        self.top_material_bar = tk.Frame(left_box, bg=th["surface"], highlightthickness=2, highlightbackground=th["card_border"], padx=10, pady=5)
        self.top_material_bar.pack(fill=tk.X, pady=(0, 8))
        self.top_player_title = tk.Label(self.top_material_bar, text="Red", font=("Segoe UI", 11, "bold"), fg=th["red_color"], bg=th["surface"])
        self.top_player_title.pack(side=tk.LEFT)
        self.top_material_label = tk.Label(self.top_material_bar, text="🪨 3  📄 4  ✂️ 3", font=("Segoe UI", 10), fg=th["ink"], bg=th["surface"])
        self.top_material_label.pack(side=tk.RIGHT)

        # Board + Eval bar row
        board_row = tk.Frame(left_box, bg=th["page_bg"])
        board_row.pack()

        # Ranks labels (left of board)
        self.ranks_canvas = tk.Canvas(board_row, width=20, height=self.BOARD_PIXELS, bg=th["page_bg"], highlightthickness=0)
        self.ranks_canvas.pack(side=tk.LEFT, padx=(0, 4))

        # Main 9x9 Board Canvas
        self.board_canvas = tk.Canvas(
            board_row,
            width=self.BOARD_PIXELS,
            height=self.BOARD_PIXELS,
            bg=th["grid_border"],
            highlightthickness=2,
            highlightbackground=th["grid_border"],
            cursor="hand2",
        )
        self.board_canvas.pack(side=tk.LEFT)
        self.board_canvas.bind("<Button-1>", self._on_canvas_click)

        # Vertical Engine Eval Bar (next to board, exactly like meaf.us / Hushhush)
        self.eval_canvas = tk.Canvas(
            board_row,
            width=self.EVAL_BAR_WIDTH,
            height=self.BOARD_PIXELS,
            bg=th["eval_bg"],
            highlightthickness=2,
            highlightbackground=th["grid_border"],
        )
        self.eval_canvas.pack(side=tk.LEFT, padx=(6, 0))

        # Files labels (bottom of board)
        files_row = tk.Frame(left_box, bg=th["page_bg"])
        files_row.pack(fill=tk.X, pady=(4, 0))
        tk.Frame(files_row, width=24, bg=th["page_bg"]).pack(side=tk.LEFT)  # align with rank offset
        self.files_canvas = tk.Canvas(files_row, width=self.BOARD_PIXELS, height=20, bg=th["page_bg"], highlightthickness=0)
        self.files_canvas.pack(side=tk.LEFT)

        # Bottom player header (Blue by default, or Red if flipped)
        self.bottom_material_bar = tk.Frame(left_box, bg=th["surface"], highlightthickness=2, highlightbackground=th["card_border"], padx=10, pady=5)
        self.bottom_material_bar.pack(fill=tk.X, pady=(8, 0))
        self.bottom_player_title = tk.Label(self.bottom_material_bar, text="Blue", font=("Segoe UI", 11, "bold"), fg=th["blue_color"], bg=th["surface"])
        self.bottom_player_title.pack(side=tk.LEFT)
        self.bottom_material_label = tk.Label(self.bottom_material_bar, text="🪨 3  📄 4  ✂️ 3", font=("Segoe UI", 10), fg=th["ink"], bg=th["surface"])
        self.bottom_material_label.pack(side=tk.RIGHT)

        # -------------------------------------------------------------
        # Right Area: Neo-brutalist control panel & move history
        # -------------------------------------------------------------
        right_box = tk.Frame(self.main_container, bg=th["surface"], highlightthickness=2, highlightbackground=th["card_border"], width=310, padx=16, pady=14)
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_box.pack_propagate(False)

        # Game title & Theme toggle
        hdr_frame = tk.Frame(right_box, bg=th["surface"])
        hdr_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(hdr_frame, text="INTRANSITIVE", font=("Segoe UI", 14, "bold"), fg=th["ink"], bg=th["surface"]).pack(side=tk.LEFT)
        self.theme_btn = tk.Button(
            hdr_frame,
            text="🌙 Dark" if self.current_theme_name == "light" else "☀️ Light",
            command=self.toggle_theme,
            font=("Segoe UI", 8, "bold"),
            bg=th["surface_muted"],
            fg=th["ink"],
            relief=tk.FLAT,
            padx=6,
            pady=2,
            highlightthickness=1,
            highlightbackground=th["card_border"],
            cursor="hand2",
        )
        self.theme_btn.pack(side=tk.RIGHT)

        # Status badge banner
        self.status_banner = tk.Label(
            right_box,
            text="Blue to move",
            font=("Segoe UI", 11, "bold"),
            fg=th["surface"],
            bg=th["blue_color"],
            padx=10,
            pady=6,
            relief=tk.FLAT,
        )
        self.status_banner.pack(fill=tk.X, pady=(0, 12))

        # Match setup controls
        setup_frame = tk.Frame(right_box, bg=th["surface"])
        setup_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(setup_frame, text="Blue Player (Goal: i9)", fg=th["blue_color"], bg=th["surface"], font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.blue_var = tk.StringVar(value="Human")
        blue_cb = ttk.Combobox(setup_frame, textvariable=self.blue_var, values=PLAYER_CHOICES, state="readonly")
        blue_cb.pack(fill=tk.X, pady=(2, 6))

        tk.Label(setup_frame, text="Red Player (Goal: a1)", fg=th["red_color"], bg=th["surface"], font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.red_var = tk.StringVar(value=default_red)
        red_cb = ttk.Combobox(setup_frame, textvariable=self.red_var, values=PLAYER_CHOICES, state="readonly")
        red_cb.pack(fill=tk.X, pady=(2, 8))

        # Simulations setting
        sim_frame = tk.Frame(setup_frame, bg=th["surface"])
        sim_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(sim_frame, text="MCTS Rollouts:", fg=th["ink_muted"], bg=th["surface"], font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.sims_var = tk.IntVar(value=400)
        sim_spin = tk.Spinbox(sim_frame, from_=10, to=2000, increment=20, textvariable=self.sims_var, width=5)
        sim_spin.pack(side=tk.RIGHT)

        # Action buttons
        btn_frame = tk.Frame(right_box, bg=th["surface"])
        btn_frame.pack(fill=tk.X, pady=(0, 12))

        self.btn_new = tk.Button(
            btn_frame,
            text="New Game",
            command=self.new_game,
            bg="#222222",
            fg="#ffffff",
            activebackground="#444444",
            activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=5,
            cursor="hand2",
        )
        self.btn_new.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self.btn_undo = tk.Button(
            btn_frame,
            text="Undo",
            command=self.undo_move,
            bg=th["surface_muted"],
            fg=th["ink"],
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=5,
            highlightthickness=1,
            highlightbackground=th["card_border"],
            cursor="hand2",
        )
        self.btn_undo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_flip = tk.Button(
            btn_frame,
            text="Flip",
            command=self.flip_board,
            bg=th["surface_muted"],
            fg=th["ink"],
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=5,
            highlightthickness=1,
            highlightbackground=th["card_border"],
            cursor="hand2",
        )
        self.btn_flip.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        # Bot evaluation text box
        self.eval_box = tk.Label(
            right_box,
            text="Engine Eval: Ready",
            font=("Consolas", 9),
            fg=th["ink"],
            bg=th["surface_muted"],
            padx=8,
            pady=6,
            justify=tk.LEFT,
            anchor="w",
            highlightthickness=1,
            highlightbackground=th["card_border"],
        )
        self.eval_box.pack(fill=tk.X, pady=(0, 10))

        # Move History Section
        tk.Label(right_box, text="MOVE HISTORY", font=("Segoe UI", 9, "bold"), fg=th["ink_muted"], bg=th["surface"]).pack(anchor="w", pady=(0, 4))

        hist_container = tk.Frame(right_box, bg=th["surface"], highlightthickness=1, highlightbackground=th["card_border"])
        hist_container.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(hist_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_list = tk.Listbox(
            hist_container,
            bg=th["surface"],
            fg=th["ink"],
            font=("Consolas", 9),
            selectbackground=th["blue_color"],
            selectforeground="#ffffff",
            borderwidth=0,
            highlightthickness=0,
            yscrollcommand=scrollbar.set,
        )
        self.history_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        scrollbar.config(command=self.history_list.yview)

    def toggle_theme(self) -> None:
        self.current_theme_name = "dark" if self.current_theme_name == "light" else "light"
        self.theme = THEMES[self.current_theme_name]
        th = self.theme

        self.root.configure(bg=th["page_bg"])
        self.main_container.configure(bg=th["page_bg"])
        self.ranks_canvas.configure(bg=th["page_bg"])
        self.files_canvas.configure(bg=th["page_bg"])
        self.board_canvas.configure(bg=th["grid_border"], highlightbackground=th["grid_border"])
        self.eval_canvas.configure(bg=th["eval_bg"], highlightbackground=th["grid_border"])

        self.top_material_bar.configure(bg=th["surface"], highlightbackground=th["card_border"])
        self.top_material_label.configure(bg=th["surface"], fg=th["ink"])
        self.bottom_material_bar.configure(bg=th["surface"], highlightbackground=th["card_border"])
        self.bottom_material_label.configure(bg=th["surface"], fg=th["ink"])

        self.theme_btn.configure(
            text="🌙 Dark" if self.current_theme_name == "light" else "☀️ Light",
            bg=th["surface_muted"],
            fg=th["ink"],
            highlightbackground=th["card_border"],
        )
        self.btn_undo.configure(bg=th["surface_muted"], fg=th["ink"], highlightbackground=th["card_border"])
        self.btn_flip.configure(bg=th["surface_muted"], fg=th["ink"], highlightbackground=th["card_border"])
        self.eval_box.configure(bg=th["surface_muted"], fg=th["ink"], highlightbackground=th["card_border"])
        self.history_list.configure(bg=th["surface"], fg=th["ink"])

        self._draw_board()
        self._draw_eval_bar()

    def _board_to_canvas_coords(self, rank: int, file_idx: int) -> Tuple[int, int]:
        if not self.flipped:
            row = (BOARD_SIZE - 1) - rank
            col = file_idx
        else:
            row = rank
            col = (BOARD_SIZE - 1) - file_idx
        return row, col

    def _canvas_xy_to_sq(self, x: int, y: int) -> Optional[int]:
        col = x // self.CELL_SIZE
        row = y // self.CELL_SIZE
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return None
        if not self.flipped:
            rank = (BOARD_SIZE - 1) - row
            file_idx = col
        else:
            rank = row
            file_idx = (BOARD_SIZE - 1) - col
        return rank * BOARD_SIZE + file_idx

    def _draw_eval_bar(self) -> None:
        """Render the vertical engine win-percentage bar matching meaf.us."""
        self.eval_canvas.delete("all")
        th = self.theme
        h = self.BOARD_PIXELS
        w = self.EVAL_BAR_WIDTH

        blue_pct = max(0.0, min(100.0, self.current_blue_win_pct))
        blue_h = int((blue_pct / 100.0) * h)

        if not self.flipped:
            # Blue is on bottom, Red on top
            # Red portion (top)
            self.eval_canvas.create_rectangle(0, 0, w, h - blue_h, fill=th["eval_red"], outline="")
            # Blue portion (bottom)
            self.eval_canvas.create_rectangle(0, h - blue_h, w, h, fill=th["eval_blue"], outline="")
        else:
            # Red is on bottom, Blue on top
            self.eval_canvas.create_rectangle(0, 0, w, blue_h, fill=th["eval_blue"], outline="")
            self.eval_canvas.create_rectangle(0, blue_h, w, h, fill=th["eval_red"], outline="")

    def _draw_board(self) -> None:
        self.board_canvas.delete("all")
        self.ranks_canvas.delete("all")
        self.files_canvas.delete("all")
        th = self.theme

        # Draw coordinate labels around board
        for i in range(BOARD_SIZE):
            file_idx = i if not self.flipped else (BOARD_SIZE - 1) - i
            rank_idx = (BOARD_SIZE - 1) - i if not self.flipped else i
            file_char = chr(ord("a") + file_idx)
            rank_char = str(rank_idx + 1)

            cx = i * self.CELL_SIZE + self.CELL_SIZE // 2
            cy = i * self.CELL_SIZE + self.CELL_SIZE // 2

            self.ranks_canvas.create_text(10, cy, text=rank_char, fill=th["ink_muted"], font=("Segoe UI", 10, "bold"))
            self.files_canvas.create_text(cx, 10, text=file_char, fill=th["ink_muted"], font=("Segoe UI", 10, "bold"))

        # Draw 9x9 grid cells
        # In meaf.us, cells sit inside a grid with 2px borders
        for sq in range(NUM_SQUARES):
            rank, file_idx = sq // BOARD_SIZE, sq % BOARD_SIZE
            row, col = self._board_to_canvas_coords(rank, file_idx)
            x1 = col * self.CELL_SIZE
            y1 = row * self.CELL_SIZE
            x2 = x1 + self.CELL_SIZE
            y2 = y1 + self.CELL_SIZE
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Cell background color
            bg = th["cell_bg"]
            if sq == BLUE_GOAL:
                bg = th["base_red"]  # i9 is Red's starting base / Blue's target corner
            elif sq == RED_GOAL:
                bg = th["base_blue"] # a1 is Blue's starting base / Red's target corner

            if sq in self.last_move_squares:
                bg = th["last_move"]

            # Draw cell rectangle with 2px gap (grid line)
            self.board_canvas.create_rectangle(
                x1 + 1, y1 + 1, x2 - 1, y2 - 1, fill=bg, outline=""
            )

            # Draw selected square highlight (inset 3px gold ring)
            if sq == self.selected_sq:
                self.board_canvas.create_rectangle(
                    x1 + 2, y1 + 2, x2 - 2, y2 - 2, outline=th["selected_border"], width=3
                )

            # Draw piece if present
            piece = int(self.board.grid[sq])
            if piece != EMPTY:
                side_name = "blue" if piece > 0 else "red"
                img = self.piece_images.get((side_name, abs(piece)))
                if img is not None:
                    self.board_canvas.create_image(cx, cy, image=img)
                else:
                    self._draw_vector_piece_fallback(cx, cy, piece)

            # Draw legal move hints
            if sq in self.legal_dest_map:
                if piece == EMPTY:
                    # Original meaf.us translucent move dot
                    r = int(self.CELL_SIZE * 0.14)
                    self.board_canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r, fill=th["move_dot"], outline=""
                    )
                else:
                    # Original meaf.us capturable target outline
                    self.board_canvas.create_rectangle(
                        x1 + 2, y1 + 2, x2 - 2, y2 - 2, outline=th["capturable_border"], width=3
                    )

        # Update material counts in header bars
        blue_r = int((self.board.grid == ROCK).sum())
        blue_p = int((self.board.grid == PAPER).sum())
        blue_s = int((self.board.grid == SCISSORS).sum())
        red_r = int((self.board.grid == -ROCK).sum())
        red_p = int((self.board.grid == -PAPER).sum())
        red_s = int((self.board.grid == -SCISSORS).sum())

        top_is_red = not self.flipped
        if top_is_red:
            self.top_player_title.configure(text=f"Red  ({self.red_var.get()})", fg=th["red_color"])
            self.top_material_label.configure(text=f"🪨 {red_r}   📄 {red_p}   ✂️ {red_s}")
            self.bottom_player_title.configure(text=f"Blue ({self.blue_var.get()})", fg=th["blue_color"])
            self.bottom_material_label.configure(text=f"🪨 {blue_r}   📄 {blue_p}   ✂️ {blue_s}")
        else:
            self.top_player_title.configure(text=f"Blue ({self.blue_var.get()})", fg=th["blue_color"])
            self.top_material_label.configure(text=f"🪨 {blue_r}   📄 {blue_p}   ✂️ {blue_s}")
            self.bottom_player_title.configure(text=f"Red  ({self.red_var.get()})", fg=th["red_color"])
            self.bottom_material_label.configure(text=f"🪨 {red_r}   📄 {red_p}   ✂️ {red_s}")

        # Update status banner
        res = self.board.result()
        if res is not None:
            if res.winner == BLUE:
                self.status_banner.configure(text=f"Blue Wins ({res.reason})", bg=th["blue_color"])
            elif res.winner == RED:
                self.status_banner.configure(text=f"Red Wins ({res.reason})", bg=th["red_color"])
            else:
                self.status_banner.configure(text=f"Draw ({res.reason})", bg="#d97706")
        else:
            side_txt = "Blue to move" if self.board.side == BLUE else "Red to move"
            think_txt = " (thinking...)" if self.ai_thinking else ""
            banner_bg = th["blue_color"] if self.board.side == BLUE else th["red_color"]
            self.status_banner.configure(text=f"{side_txt}{think_txt} | Clock: {self.board.quiet}/{QUIET_LIMIT}", bg=banner_bg)

        self._draw_eval_bar()

    def _draw_vector_piece_fallback(self, cx: int, cy: int, piece: int) -> None:
        """Render high-quality 2D vector piece fallback directly onto the canvas."""
        p_type = abs(piece)
        if piece > 0:
            col_prim = "#4b6fd9"
            col_light = "#93b4ff"
            col_dark = "#233c96"
            col_out = "#0f1946"
        else:
            col_prim = "#d94b4b"
            col_light = "#ff9696"
            col_dark = "#9b2323"
            col_out = "#460c0c"

        if p_type == ROCK:
            pts = [
                (cx - 16, cy + 10), (cx - 19, cy - 2), (cx - 13, cy - 14), (cx - 2, cy - 20),
                (cx + 12, cy - 18), (cx + 19, cy - 7), (cx + 18, cy + 9), (cx + 11, cy + 18),
                (cx - 3, cy + 19), (cx - 13, cy + 16)
            ]
            self.board_canvas.create_polygon(pts, fill=col_prim, outline=col_out, width=2)
            self.board_canvas.create_polygon([(cx - 2, cy - 20), (cx + 12, cy - 18), (cx + 6, cy - 8), (cx - 5, cy - 9)], fill=col_light, outline=col_out)
            self.board_canvas.create_polygon([(cx - 13, cy - 14), (cx - 2, cy - 20), (cx - 5, cy - 9), (cx - 14, cy - 2)], fill=col_light, outline=col_out)
            self.board_canvas.create_polygon([(cx - 5, cy - 9), (cx + 6, cy - 8), (cx + 10, cy + 3), (cx - 2, cy + 7), (cx - 10, cy + 2)], fill=col_prim, outline=col_out)
            self.board_canvas.create_polygon([(cx - 3, cy + 19), (cx + 11, cy + 18), (cx + 4, cy + 14), (cx - 3, cy + 11)], fill=col_dark, outline=col_out)
        elif p_type == PAPER:
            body_pts = [
                (cx - 14, cy - 16), (cx + 8, cy - 16), (cx + 8, cy - 7), (cx + 15, cy),
                (cx + 15, cy + 16), (cx + 11, cy + 19), (cx + 5, cy + 16), (cx - 2, cy + 19),
                (cx - 9, cy + 16), (cx - 16, cy + 19), (cx - 16, cy - 13)
            ]
            self.board_canvas.create_polygon(body_pts, fill="#ffffff", outline=col_out, width=2)
            fold_pts = [(cx + 8, cy - 16), (cx + 15, cy - 7), (cx + 8, cy - 7)]
            self.board_canvas.create_polygon(fold_pts, fill=col_light, outline=col_out, width=1)
            self.board_canvas.create_rectangle(cx - 14, cy - 13, cx + 5, cy - 7, fill=col_prim, outline="")
            for dy in [-1, 4, 9, 14]:
                w = 18 if dy < 14 else 12
                self.board_canvas.create_line(cx - 12, cy + dy, cx - 12 + w, cy + dy, fill=col_prim, width=2)
            self.board_canvas.create_oval(cx + 6, cy + 7, cx + 14, cy + 15, fill="#f5c542", outline=col_out, width=1)
        elif p_type == SCISSORS:
            self.board_canvas.create_polygon([(cx - 1, cy - 2), (cx + 14, cy - 19), (cx + 17, cy - 16), (cx + 2, cy)], fill="#e8edf5", outline=col_out, width=1)
            self.board_canvas.create_polygon([(cx + 1, cy - 2), (cx - 14, cy - 19), (cx - 17, cy - 16), (cx - 2, cy)], fill="#d0d8e4", outline=col_out, width=1)
            self.board_canvas.create_line(cx - 1, cy, cx - 8, cy + 8, fill=col_prim, width=3)
            self.board_canvas.create_oval(cx - 18, cy + 6, cx - 2, cy + 22, fill=col_prim, outline=col_out, width=2)
            self.board_canvas.create_oval(cx - 14, cy + 10, cx - 6, cy + 18, fill=self.theme["cell_bg"], outline=col_out, width=1)
            self.board_canvas.create_line(cx + 1, cy, cx + 8, cy + 8, fill=col_prim, width=3)
            self.board_canvas.create_oval(cx + 2, cy + 6, cx + 18, cy + 22, fill=col_dark, outline=col_out, width=2)
            self.board_canvas.create_oval(cx + 6, cy + 10, cx + 14, cy + 18, fill=self.theme["cell_bg"], outline=col_out, width=1)
            self.board_canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#f5c542", outline=col_out, width=1)

    def new_game(self) -> None:
        self.game_generation += 1
        self.ai_thinking = False
        self.board = Board(START_FEN)
        self.board_history = [self.board.copy()]
        self.move_log.clear()
        self.history_list.delete(0, tk.END)
        self.selected_sq = None
        self.legal_dest_map.clear()
        self.last_move_squares.clear()
        self.current_blue_win_pct = 50.0
        self.eval_box.configure(text="Engine Eval: Ready")

        self.blue_agent = self._create_player(self.blue_var.get())
        self.red_agent = self._create_player(self.red_var.get())

        if self.blue_agent is not None and self.red_agent is None:
            self.flipped = True
        elif self.blue_agent is None and self.red_agent is not None:
            self.flipped = False

        self._draw_board()
        self._maybe_trigger_ai()

    def flip_board(self) -> None:
        self.flipped = not self.flipped
        self._draw_board()

    def undo_move(self) -> None:
        if self.ai_thinking or len(self.board_history) <= 1:
            return
        steps_back = 2 if (self.blue_agent is not None or self.red_agent is not None) and len(self.board_history) >= 3 else 1
        for _ in range(steps_back):
            self.board_history.pop()
            if self.move_log:
                self.move_log.pop()
                self.history_list.delete(tk.END)

        self.board = self.board_history[-1].copy()
        self.selected_sq = None
        self.legal_dest_map.clear()
        self.last_move_squares.clear()
        self._draw_board()

    def _apply_move_uci(self, uci: str) -> None:
        from_sq, to_sq = uci_to_squares(uci)
        notation = to_notation(self.board.grid, uci)
        side_tag = "B" if self.board.side == BLUE else "R"
        ply_num = self.board.ply + 1

        self.board.push_uci(uci, validate=True)
        self.board_history.append(self.board.copy())
        self.last_move_squares = {from_sq, to_sq}
        self.selected_sq = None
        self.legal_dest_map.clear()

        entry = f"{ply_num:3d}. [{side_tag}] {notation:8s}"
        self.move_log.append((str(ply_num), side_tag, notation))
        self.history_list.insert(tk.END, entry)
        self.history_list.yview_moveto(1.0)

        self._draw_board()
        self._maybe_trigger_ai()

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self.ai_thinking or self.board.is_terminal():
            return
        active_agent = self.blue_agent if self.board.side == BLUE else self.red_agent
        if active_agent is not None:
            return

        sq = self._canvas_xy_to_sq(event.x, event.y)
        if sq is None:
            return

        if sq in self.legal_dest_map:
            self._apply_move_uci(self.legal_dest_map[sq])
            return

        piece = int(self.board.grid[sq])
        if piece != EMPTY and ((piece > 0 and self.board.side == BLUE) or (piece < 0 and self.board.side == RED)):
            self.selected_sq = sq
            self.legal_dest_map.clear()
            prefix = square_label(sq)
            for mv in self.board.legal_moves():
                if mv.startswith(prefix):
                    _, dst_sq = uci_to_squares(mv)
                    self.legal_dest_map[dst_sq] = mv
        else:
            self.selected_sq = None
            self.legal_dest_map.clear()

        self._draw_board()

    def _maybe_trigger_ai(self) -> None:
        if self.board.is_terminal():
            return
        active_agent = self.blue_agent if self.board.side == BLUE else self.red_agent
        if active_agent is None:
            return

        self.ai_thinking = True
        self._draw_board()
        gen = self.game_generation
        board_snapshot = self.board.copy()

        def worker() -> None:
            if isinstance(active_agent, AlphaZeroPlayer):
                active_agent.mcts.num_simulations = int(self.sims_var.get())
            chosen_uci = active_agent.select_move(board_snapshot)

            eval_str = f"Bot: {active_agent.name}"
            blue_win_pct = 50.0
            if isinstance(active_agent, AlphaZeroPlayer) and active_agent.last_root is not None:
                root_node = active_agent.last_root
                q = root_node.q_value
                win_pct_for_active = (q + 1.0) * 50.0
                blue_win_pct = win_pct_for_active if board_snapshot.side == BLUE else (100.0 - win_pct_for_active)

                pv_d = getattr(root_node, "pv_depth", 0)
                tb_st = getattr(root_node, "tb_status", None)
                children = getattr(root_node, "children", None) or {}
                if children:
                    top_children = sorted(
                        children.items(),
                        key=lambda kv: kv[1].visit_count,
                        reverse=True,
                    )[:3]
                    top_str = ", ".join(
                        f"{action_to_uci(from_canonical_action(ca, board_snapshot.side))}({ch.visit_count})"
                        for ca, ch in top_children
                    )
                    detail_line = f"Top: {top_str}"
                elif tb_st:
                    detail_line = tb_st
                else:
                    detail_line = f"PV depth: {pv_d} | Visits: {root_node.visit_count}"
                eval_str = f"Blue Win: {blue_win_pct:4.1f}% (Q={q:+.2f})\n{detail_line}"

            self.root.after(0, lambda: self._on_ai_finished(gen, chosen_uci, eval_str, blue_win_pct))

        threading.Thread(target=worker, daemon=True).start()

    def _on_ai_finished(self, gen: int, chosen_uci: str, eval_str: str, blue_win_pct: float) -> None:
        if gen != self.game_generation:
            return
        self.ai_thinking = False
        self.current_blue_win_pct = blue_win_pct
        self.eval_box.configure(text=eval_str)
        self._apply_move_uci(chosen_uci)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch meaf.us-style Intransitive GUI")
    parser.add_argument("--opponent", type=str, default="AlphaZero (Best)", choices=PLAYER_CHOICES)
    args = parser.parse_args()

    root = tk.Tk()
    IntransitiveGUI(root, default_red=args.opponent)
    root.mainloop()


if __name__ == "__main__":
    main()
