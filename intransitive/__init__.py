"""Intransitive (RPS V6) Game Engine and Reinforcement Learning Package."""

from intransitive.core.board import Board
from intransitive.core.constants import BLUE, RED, START_FEN, QUIET_LIMIT
from intransitive.core.rules import game_result, legal_moves, play_move

__all__ = [
    "Board",
    "BLUE",
    "RED",
    "START_FEN",
    "QUIET_LIMIT",
    "game_result",
    "legal_moves",
    "play_move",
]
