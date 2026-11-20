"""Tests for Policy-Value ResNet, MCTS tactical search, Gymnasium env, and Baselines."""

import numpy as np
import torch

from intransitive.baselines.greedy_player import GreedyPlayer
from intransitive.baselines.minimax_player import MinimaxPlayer
from intransitive.core.board import Board
from intransitive.env.gym_env import IntransitiveEnv
from intransitive.rl.device import get_device_summary, optimize_with_ipex, resolve_device
from intransitive.rl.mcts import AlphaZeroPlayer
from intransitive.rl.network import IntransitiveNet


def test_device_and_batch_prediction() -> None:
    dev = resolve_device("auto")
    summary = get_device_summary(dev)
    assert "active_device" in summary
    assert "ipex_installed" in summary

    net = IntransitiveNet(hidden_channels=32, num_res_blocks=2).to(dev)
    boards = [Board(), Board("9/9/2r6/9/9/9/9/7R1/9 b")]
    probs_list, vals_list = net.predict_batch(boards, device=dev)
    assert len(probs_list) == 2
    assert len(vals_list) == 2
    for probs, val in zip(probs_list, vals_list):
        assert probs.shape == (648,)
        assert abs(float(np.sum(probs)) - 1.0) < 1e-5
        assert -1.0 <= val <= 1.0

    opt_net, _ = optimize_with_ipex(net)
    assert opt_net is not None


def test_network_forward_and_predict() -> None:
    torch.manual_seed(0)
    net = IntransitiveNet(hidden_channels=32, num_res_blocks=2)
    board = Board()
    probs, val = net.predict(board)
    assert probs.shape == (648,)
    assert abs(float(np.sum(probs)) - 1.0) < 1e-5
    assert -1.0 <= val <= 1.0


def test_mcts_finds_1ply_blue_corner_win() -> None:
    torch.manual_seed(0)
    net = IntransitiveNet(hidden_channels=32, num_res_blocks=2)
    player = AlphaZeroPlayer(network=net, num_simulations=20, temperature=0.0)

    # Blue R at h8 can step to i9 for immediate victory
    board = Board("9/9/2r6/9/9/9/9/7R1/9 b")
    move = player.select_move(board)
    assert move == "h8i9"


def test_mcts_finds_1ply_red_corner_win() -> None:
    torch.manual_seed(0)
    net = IntransitiveNet(hidden_channels=32, num_res_blocks=2)
    player = AlphaZeroPlayer(network=net, num_simulations=20, temperature=0.0)

    # Red s at b2 can step to a1 for immediate victory
    board = Board("9/1s7/9/9/9/9/6R2/9/9 r")
    move = player.select_move(board)
    assert move == "b2a1"


def test_mcts_defends_1ply_opponent_corner_threat() -> None:
    torch.manual_seed(0)
    net = IntransitiveNet(hidden_channels=32, num_res_blocks=2)
    # Give MCTS enough simulations to expand all root moves and discover Red's 1-ply win
    player = AlphaZeroPlayer(network=net, num_simulations=40, temperature=0.0)

    # Red has s at b2 threatening b2a1 next turn!
    # Blue has R at c2 which can capture s at b2 (c2b2) to save the game!
    board = Board("9/1sR6/9/9/9/9/9/9/9 b")
    move = player.select_move(board)
    assert move == "c2b2"


def test_minimax_and_greedy_find_tactics() -> None:
    board_win = Board("9/9/2r6/9/9/9/9/7R1/9 b")
    assert GreedyPlayer().select_move(board_win) == "h8i9"
    assert MinimaxPlayer(depth=2).select_move(board_win) == "h8i9"

    board_defend = Board("9/1sR6/9/9/9/9/9/9/9 b")
    assert MinimaxPlayer(depth=2).select_move(board_defend) == "c2b2"


def test_gym_environment_step_and_mask() -> None:
    env = IntransitiveEnv(opponent=GreedyPlayer(seed=42))
    obs, info = env.reset()
    assert obs.shape == (7, 9, 9)
    mask = info["action_mask"]
    assert mask.shape == (648,)
    legal_indices = np.flatnonzero(mask)
    assert len(legal_indices) > 0

    next_obs, reward, terminated, truncated, next_info = env.step(int(legal_indices[0]))
    assert next_obs.shape == (7, 9, 9)
    assert isinstance(reward, float)
    assert not truncated
