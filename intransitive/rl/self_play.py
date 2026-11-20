"""Self-play and curriculum sparring episode generation for AlphaZero training."""

import math
from typing import List, Optional, Tuple
import numpy as np

from intransitive.baselines.base import Player
from intransitive.baselines.minimax_player import evaluate_board
from intransitive.core.board import Board
from intransitive.core.constants import ACTION_SIZE, BLUE, START_FEN
from intransitive.core.move import uci_to_action
from intransitive.core.rules import GameResult
from intransitive.env.canonical import (
    encode_board,
    from_canonical_action,
    reflect_obs_and_policy,
    to_canonical_action,
)
from intransitive.rl.mcts import MCTS
from intransitive.rl.replay_buffer import TrainingSample


def generate_self_play_episode(
    mcts: MCTS,
    start_fen: str = START_FEN,
    temp_threshold_ply: int = 24,
    max_plies: Optional[int] = 140,
    augment_symmetry: bool = True,
    adjudicate_draws: bool = True,
    sparring_opponent: Optional[Player] = None,
    az_side: int = BLUE,
) -> Tuple[List[TrainingSample], GameResult, int]:
    """
    Play one complete game using `mcts` (either pure self-play or against `sparring_opponent`).
    Returns:
      - samples: list of TrainingSample(obs, policy, value) (including diagonal reflections if enabled)
      - result: terminal GameResult
      - total_plies: number of plies played
    """
    board = Board(fen=start_fen)
    if sparring_opponent is not None:
        sparring_opponent.reset()

    trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []

    while True:
        res = board.result()
        if res is not None:
            break
        if max_plies is not None and board.ply >= max_plies:
            res = GameResult(winner=0, reason="max-plies")
            break

        acting_side = board.side
        obs = encode_board(board)

        if sparring_opponent is not None and acting_side != az_side:
            # Sparring opponent selects a move
            opp_uci = sparring_opponent.select_move(board)
            real_action = uci_to_action(opp_uci)
            c_action = to_canonical_action(real_action, acting_side)
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[c_action] = 1.0
            trajectory.append((obs, policy, acting_side))
            board.push_action(real_action, validate=False)
        else:
            temp = 1.0 if board.ply < temp_threshold_ply else 0.15
            policy, _ = mcts.search(board, temperature=temp, add_noise=True)
            trajectory.append((obs, policy, acting_side))

            prob_sum = float(np.sum(policy))
            if prob_sum > 0:
                norm_p = policy / prob_sum
                c_action = int(mcts.rng.choice(ACTION_SIZE, p=norm_p))
            else:
                c_action = int(np.argmax(policy))

            real_action = from_canonical_action(c_action, acting_side)
            board.push_action(real_action, validate=False)

    assert res is not None

    # Compute outcome value in [-1, 1] from Blue's perspective
    if res.winner != 0:
        blue_outcome = 1.0 if res.winner == BLUE else -1.0
    elif adjudicate_draws:
        # Evaluate board from current side's perspective, convert to Blue's perspective
        eval_cp = evaluate_board(board)
        eval_blue = eval_cp if board.side == BLUE else -eval_cp
        blue_outcome = math.tanh(eval_blue / 350.0) * 0.75
    else:
        blue_outcome = 0.0

    samples: List[TrainingSample] = []
    for obs, policy, side in trajectory:
        val = blue_outcome if side == BLUE else -blue_outcome
        samples.append(TrainingSample(obs=obs, policy=policy, value=float(val)))
        if augment_symmetry:
            ref_obs, ref_pol = reflect_obs_and_policy(obs, policy)
            samples.append(TrainingSample(obs=ref_obs, policy=ref_pol, value=float(val)))

    return samples, res, board.ply
