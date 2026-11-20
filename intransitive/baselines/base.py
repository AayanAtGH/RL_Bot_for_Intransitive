"""Base player interface for Intransitive agents."""

from abc import ABC, abstractmethod
from intransitive.core.board import Board


class Player(ABC):
    """Abstract base class for all Intransitive game agents."""

    def __init__(self, name: str = "Player") -> None:
        self.name = name

    @abstractmethod
    def select_move(self, board: Board) -> str:
        """Return a legal UCI move string (e.g. 'd3d4') for the current board position."""
        raise NotImplementedError

    def reset(self) -> None:
        """Optional hook called before a new game starts."""
        pass
