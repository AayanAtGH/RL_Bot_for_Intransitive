"""Tests for v2 SE-ResNet, 3-ply tactical prover, deep C++ MCTS, replay buffer, and batched self-play."""

from pathlib import Path
import numpy as np
import torch

from intransitive.core.board import Board
from intransitive.core.constants import ACTION_SIZE, START_FEN
from intransitive.core.move import action_to_uci
from intransitive.native.native_bridge import NativeEngine
from intransitive.rl.mcts import AlphaZeroPlayer, MCTS
from intransitive.rl.network import IntransitiveNetV1, IntransitiveNetV2, load_checkpoint
from intransitive.rl.replay_buffer import MAX_POLICY_ENTRIES, ReplayBuffer
from intransitive.rl.self_play import BatchedSelfPlayRunner


def test_3ply_corner_sprint_prover() -> None:
    # Blue R at g7 (dist 2 from i9), Red p at b4, Red r at d3 -> Blue g7h8 wins on ply 3!
    fen = "9/9/4r4/1p7/9/9/6R2/9/9 b"
    b = Board(fen)
    act = NativeEngine.prove_corner_win_3ply(b.grid, b.side, b.quiet)
    assert act >= 0
    assert action_to_uci(act) == "g7h8"

    # Untrained random v2 network finds g7h8 with just 1 simulation via pre-NN prover!
    torch.manual_seed(0)
    net = IntransitiveNetV2(hidden_channels=32, num_res_blocks=2)
    player = AlphaZeroPlayer(network=net, num_simulations=1, min_pv_depth=0, temperature=0.0)
    assert player.select_move(b) == "g7h8"

    # Red r at c3 (dist 2 from a1), Blue S at f6, Blue P at g5 -> Red c3b2 wins on ply 3!
    fen_r = "9/9/2r6/9/6P2/5S3/9/9/9 r"
    br = Board(fen_r)
    act_r = NativeEngine.prove_corner_win_3ply(br.grid, br.side, br.quiet)
    assert act_r >= 0
    assert action_to_uci(act_r) == "c3b2"


def test_mcts_defends_2ply_corner_threat() -> None:
    # Red s at b3 threatens b3->b2->a1; Blue R at d3 can block/capture by moving d3c2 (capturing or threatening b3/b2)!
    # Put Red s at b2? That's 1-ply. Put Red s at b3 (dist 2 to a1) and Blue R at c4 (can step to c3 or b3? c4 to b3 is 1 step!).
    # Wait: if Blue R is at c4, c4b3 captures Red's only runner! Every non-capturing/non-blocking move lets Red play b3b2/b3a2!
    fen = "9/9/1s7/2R6/9/9/9/9/9 b"
    b = Board(fen)
    torch.manual_seed(1)
    net = IntransitiveNetV2(hidden_channels=32, num_res_blocks=2)
    player = AlphaZeroPlayer(network=net, num_simulations=120, min_pv_depth=6, temperature=0.0)
    mv = player.select_move(b)
    # Blue must either capture b3 (c4b3) or step to a2/b2 (impossible from c4 in 1 ply),
    # or check: can c4c3/c4b4/c4c2? Wait: from c3 can R capture b2? YES! R beats s!
    # So c4b3 (captures s) or c4c3 (attacks b2, b3 can't win next turn via b2 since R captures b2? Wait: s at b3 -> a2 still wins!).
    # Indeed ONLY c4b3 stops b3->a2->a1!
    assert mv == "c4b3"


class _PeakedConsistentNet(torch.nn.Module):
    """Mock network putting 0.85 prior on an advancing non-repeating move and 0.0 value."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def predict_batch_raw(self, obs_np, masks_np, device="cpu", use_amp=True):
        bsz = obs_np.shape[0]
        probs = np.zeros((bsz, ACTION_SIZE), dtype=np.float32)
        for i in range(bsz):
            legal = np.flatnonzero(masks_np[i])
            if len(legal) >= 2:
                adv = [int(a) for a in legal if (int(a) % 8) in (1, 2, 0)]
                top = adv[-1] if adv else int(legal[-1])
                rem_mass = 0.15 / float(len(legal) - 1)
                for a in legal:
                    probs[i, int(a)] = 0.85 if int(a) == top else rem_mass
            elif len(legal) == 1:
                probs[i, int(legal[0])] = 1.0
        vals = np.zeros((bsz,), dtype=np.float32)
        return probs, vals


def test_fpu_subtree_reuse_and_pv_depth() -> None:
    torch.manual_seed(0)
    net = IntransitiveNetV2(hidden_channels=32, num_res_blocks=2)
    # 1. Verify adaptive PV-depth extension fires on START_FEN when pv_depth < min_pv_depth
    player = AlphaZeroPlayer(
        network=net,
        num_simulations=100,
        min_pv_depth=6,
        max_simulations=300,
        reuse_subtree=True,
        temperature=0.0,
    )
    b_start = Board(START_FEN)
    mv0 = player.select_move(b_start)
    assert player.last_root is not None
    assert player.last_root.visit_count > 100
    assert mv0 in b_start.legal_moves()

    # 2. Focused policy + Dynamic FPU + Subtree Reuse reaches deep PV (>= 6 plies)
    peaked_net = _PeakedConsistentNet()
    player_deep = AlphaZeroPlayer(
        network=peaked_net,  # type: ignore[arg-type]
        num_simulations=300,
        min_pv_depth=6,
        max_simulations=600,
        reuse_subtree=True,
        temperature=0.0,
    )
    b = Board(START_FEN)
    mv1 = player_deep.select_move(b)
    assert player_deep.last_root is not None
    assert player_deep.last_root.pv_depth >= 6

    # Opponent follows top advancing move along the explored PV subtree -> subtree reuse preserves PV depth >= 6
    b.push_uci(mv1)
    from intransitive.env.canonical import canonical_legal_actions, from_canonical_action
    opp_c_acts = canonical_legal_actions(b)
    opp_adv = [a for a in opp_c_acts if (a % 8) in (1, 2, 0)]
    opp_c = opp_adv[-1] if opp_adv else opp_c_acts[-1]
    opp_mv = action_to_uci(from_canonical_action(opp_c, b.side))
    b.push_uci(opp_mv)
    mv2 = player_deep.select_move(b)
    assert mv2 in b.legal_moves()
    assert player_deep.last_root is not None
    assert player_deep.last_root.pv_depth >= 6


def test_mcts_repetition_avoidance() -> None:
    torch.manual_seed(0)
    net = IntransitiveNetV2(hidden_channels=32, num_res_blocks=2)
    mcts = MCTS(network=net, num_simulations=40, batch_size=4, use_tactical_prover=False)
    try:
        # 5-piece symmetrical board so 4-pc TB does not short-circuit; verifies repetition checks + search complete cleanly.
        b = Board("9/9/2r6/9/3Rp4/4S1s2/9/9/9 b")
        pol, root = mcts.search(b, temperature=0.0, add_noise=False)
        assert float(np.sum(pol)) > 0.99
        assert root.visit_count >= 40
    finally:
        mcts.close()


def test_network_v2_conv_policy_and_legacy_v1_loader(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net_v2 = IntransitiveNetV2(hidden_channels=128, num_res_blocks=6, se_every=2)
    param_count = sum(p.numel() for p in net_v2.parameters())
    assert 1_400_000 < param_count < 2_500_000

    x = torch.randn(4, 11, 9, 9)
    p_log, wdl_log = net_v2(x)
    assert p_log.shape == (4, ACTION_SIZE)
    assert wdl_log.shape == (4, 3)

    # Save and reload v2 checkpoint
    v2_path = tmp_path / "v2.pt"
    torch.save(
        {
            "arch_version": 2,
            "model_state_dict": net_v2.state_dict(),
            "ema_state_dict": net_v2.state_dict(),
            "config": {"in_channels": 11, "hidden_channels": 128, "num_res_blocks": 6, "se_every": 2},
        },
        v2_path,
    )
    loaded_v2, meta2 = load_checkpoint(v2_path)
    assert isinstance(loaded_v2, IntransitiveNetV2)
    assert meta2["arch_version"] == 2

    # Save and reload synthetic v1 checkpoint
    net_v1 = IntransitiveNetV1(in_channels=7, hidden_channels=64, num_res_blocks=4)
    v1_path = tmp_path / "v1.pt"
    torch.save({"model_state_dict": net_v1.state_dict()}, v1_path)
    loaded_v1, _ = load_checkpoint(v1_path)
    assert isinstance(loaded_v1, IntransitiveNetV1)

    # Also test existing repository v1 checkpoint if present
    repo_v1 = Path(__file__).resolve().parent.parent / "checkpoints" / "alphazero_best.pt"
    if repo_v1.exists():
        existing_net, _ = load_checkpoint(repo_v1)
        assert isinstance(existing_net, IntransitiveNetV1)
        probs, val = existing_net.predict(Board())
        assert probs.shape == (ACTION_SIZE,)
        assert -1.0 <= val <= 1.0


def test_replay_buffer_compact_6x_and_npz(tmp_path: Path) -> None:
    buf = ReplayBuffer(capacity=100, seed=7)
    n = 40
    c_grids = np.zeros((n, 81), dtype=np.int8)
    c_grids[:, 10] = 1
    c_grids[:, 50] = -2
    quiets = np.arange(n, dtype=np.uint8)
    pol_acts = np.zeros((n, MAX_POLICY_ENTRIES), dtype=np.uint16)
    pol_acts[:, 0] = 80
    pol_acts[:, 1] = 81
    pol_probs = np.zeros((n, MAX_POLICY_ENTRIES), dtype=np.float32)
    pol_probs[:, 0] = 0.7
    pol_probs[:, 1] = 0.3
    pol_lens = np.full((n,), 2, dtype=np.uint8)
    wdl = np.array([i % 3 for i in range(n)], dtype=np.uint8)
    train_pol = np.array([1 if i % 2 == 0 else 0 for i in range(n)], dtype=np.uint8)

    buf.add_batch(c_grids, quiets, pol_acts, pol_probs, pol_lens, wdl, train_pol)
    assert len(buf) == 40

    obs_t, pol_t, wdl_t, mask_t = buf.sample_batch(32, augment_6x=True)
    assert obs_t.shape == (32, 11, 9, 9)
    assert pol_t.shape == (32, ACTION_SIZE)
    assert wdl_t.shape == (32,)
    assert mask_t.shape == (32,)
    assert torch.allclose(pol_t.sum(dim=-1), torch.ones(32), atol=1e-3)

    npz_path = tmp_path / "buf.npz"
    buf.save_npz(npz_path)
    buf2 = ReplayBuffer(capacity=100, seed=7)
    assert buf2.load_npz(npz_path) == 40
    assert len(buf2) == 40


def test_batched_cpp_self_play_smoke() -> None:
    net = IntransitiveNetV2(hidden_channels=32, num_res_blocks=2)
    buf = ReplayBuffer(capacity=10_000, seed=9)
    runner = BatchedSelfPlayRunner(
        num_games=16,
        full_simulations=24,
        fast_simulations=8,
        full_search_prob=0.25,
        seed=9,
    )
    try:
        stats = runner.collect_samples(net, buf, min_new_samples=200, batch_size=16, use_amp=False)
        assert stats["games_delta"] >= 1
        assert len(buf) >= 200
        assert stats["nn_evals_per_sec"] > 0.0
    finally:
        runner.close()
