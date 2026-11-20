"""Gymnasium environment for Intransitive with action masking and canonical observations."""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from intransitive.baselines.base import Player
from intransitive.core.board import Board
from intransitive.core.constants import ACTION_SIZE, BOARD_SIZE, START_FEN
from intransitive.core.move import uci_to_action
from intransitive.env.canonical import (
    NUM_OBS_CHANNELS,
    canonical_action_mask,
    encode_board,
    from_canonical_action,
)


class IntransitiveEnv(gym.Env):
    """
    Gymnasium environment for Intransitive (RPS V6).

    Supports two modes:
    1. Self-play / two-player alternating mode (`opponent=None`):
       Each `step(canonical_action)` advances one ply and returns the observation
       and action mask from the next active player's canonical perspective.
    2. Single-agent vs Opponent mode (`opponent=Player(...)`):
       Each `step(canonical_action)` executes the agent's move and, if the game
       is not over, immediately executes the opponent's response.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        opponent: Optional[Player] = None,
        render_mode: Optional[str] = None,
        start_fen: str = START_FEN,
    ) -> None:
        super().__init__()
        self.opponent = opponent
        self.render_mode = render_mode
        self.start_fen = start_fen

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.board = Board(fen=self.start_fen)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "action_mask": canonical_action_mask(self.board),
            "side": self.board.side,
            "quiet": self.board.quiet,
            "ply": self.board.ply,
            "fen": self.board.to_fen(),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        fen = (options or {}).get("fen", self.start_fen)
        self.board = Board(fen=fen)
        if self.opponent is not None:
            self.opponent.reset()
        return encode_board(self.board), self._get_info()

    def step(
        self, canonical_action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        acting_side = self.board.side
        real_action = from_canonical_action(int(canonical_action), acting_side)
        self.board.push_action(real_action, validate=True)

        res = self.board.result()
        if res is not None:
            reward = 0.0 if res.winner == 0 else (1.0 if res.winner == acting_side else -1.0)
            info = self._get_info()
            info["result"] = res
            return encode_board(self.board), reward, True, False, info

        if self.opponent is not None:
            opp_move = self.opponent.select_move(self.board)
            opp_action = uci_to_action(opp_move)
            self.board.push_action(opp_action, validate=True)
            res = self.board.result()
            if res is not None:
                reward = 0.0 if res.winner == 0 else (1.0 if res.winner == acting_side else -1.0)
                info = self._get_info()
                info["result"] = res
                return encode_board(self.board), reward, True, False, info

        return encode_board(self.board), 0.0, False, False, self._get_info()

    def render(self) -> Optional[str]:
        ascii_str = self.board.render_ascii()
        if self.render_mode == "human":
            print(ascii_str)
        return ascii_str
