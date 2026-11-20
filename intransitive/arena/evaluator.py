"""Arena match runner and Elo rating calculation."""

from dataclasses import dataclass
from typing import Tuple

from intransitive.baselines.base import Player
from intransitive.core.board import Board
from intransitive.core.constants import BLUE, RED, START_FEN
from intransitive.core.rules import GameResult


@dataclass
class MatchStats:
    """Summary of a head-to-head series between Player A and Player B."""

    player_a: str
    player_b: str
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    total_plies: int = 0

    @property
    def games(self) -> int:
        return self.a_wins + self.b_wins + self.draws

    @property
    def a_win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.a_wins + 0.5 * self.draws) / self.games


def play_game(
    blue_player: Player,
    red_player: Player,
    start_fen: str = START_FEN,
    max_plies: int = 250,
) -> Tuple[GameResult, int]:
    """Play a single game between `blue_player` and `red_player`."""
    board = Board(fen=start_fen)
    blue_player.reset()
    red_player.reset()

    while True:
        res = board.result()
        if res is not None:
            return res, board.ply
        if board.ply >= max_plies:
            return GameResult(winner=0, reason="max-plies"), board.ply

        active = blue_player if board.side == BLUE else red_player
        move = active.select_move(board)
        board.push_uci(move, validate=True)


def play_series(
    player_a: Player,
    player_b: Player,
    num_games: int = 10,
    max_plies: int = 250,
) -> MatchStats:
    """
    Play `num_games` alternating colors between `player_a` and `player_b`.
    Even game indices: player_a = Blue, player_b = Red.
    Odd game indices: player_b = Blue, player_a = Red.
    """
    stats = MatchStats(player_a=player_a.name, player_b=player_b.name)
    for i in range(num_games):
        if i % 2 == 0:
            res, plies = play_game(player_a, player_b, max_plies=max_plies)
            if res.winner == BLUE:
                stats.a_wins += 1
            elif res.winner == RED:
                stats.b_wins += 1
            else:
                stats.draws += 1
        else:
            res, plies = play_game(player_b, player_a, max_plies=max_plies)
            if res.winner == RED:
                stats.a_wins += 1
            elif res.winner == BLUE:
                stats.b_wins += 1
            else:
                stats.draws += 1
        stats.total_plies += plies
    return stats


def update_elo(
    elo_a: float, elo_b: float, score_a: float, k_factor: float = 32.0
) -> Tuple[float, float]:
    """Update Elo ratings given `score_a` in {1.0, 0.5, 0.0}."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a
    new_a = elo_a + k_factor * (score_a - expected_a)
    new_b = elo_b + k_factor * ((1.0 - score_a) - expected_b)
    return new_a, new_b
