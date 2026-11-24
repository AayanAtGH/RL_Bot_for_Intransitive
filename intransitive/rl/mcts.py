"""Deep Monte Carlo Tree Search (C++17 native core + Python fallback) with FPU, Subtree Reuse, Tactical Prover & Tablebase."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch

from intransitive.baselines.base import Player
from intransitive.core.board import Board
from intransitive.core.constants import ACTION_SIZE, NUM_SQUARES
from intransitive.core.move import action_to_uci
from intransitive.core.rules import has_immediate_corner_win
from intransitive.core.tablebase import TablebaseManager
from intransitive.env.canonical import (
    canonical_legal_actions,
    from_canonical_action,
)
from intransitive.native.native_bridge import NativeEngine
from intransitive.rl.device import resolve_device
from intransitive.rl.network import IntransitiveNetV1, IntransitiveNetV2


@dataclass
class RootSummary:
    """Lightweight summary matching `MCTSNode` interface for GUI & diagnostics."""

    q_value: float = 0.0
    visit_count: int = 0
    pv_depth: int = 0
    proven: int = 0
    tb_status: Optional[str] = None
    children: Dict[int, Any] = field(default_factory=dict)


class MCTSNode:
    """Single node in the Python fallback Monte Carlo Search Tree."""

    __slots__ = (
        "board",
        "parent",
        "prior",
        "children",
        "visit_count",
        "value_sum",
        "is_expanded",
        "proven",
        "pv_depth",
        "tb_status",
    )

    def __init__(
        self,
        board: Board,
        parent: Optional["MCTSNode"] = None,
        prior: float = 0.0,
    ) -> None:
        self.board: Board = board
        self.parent: Optional["MCTSNode"] = parent
        self.prior: float = prior
        self.children: Dict[int, "MCTSNode"] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.is_expanded: bool = False
        self.proven: int = 0
        self.pv_depth: int = 0
        self.tb_status: Optional[str] = None

    @property
    def q_value(self) -> float:
        """Mean value from `self.board.side`'s perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """
    Deep AlphaZero MCTS with PUCT, Dynamic FPU, Subtree Reuse, MCTS-Solver,
    3-ply Corner Prover, Zobrist NN Cache, and 4-piece Tablebase integration.
    Automatically delegates search to the compiled C++17 engine when available.
    """

    def __init__(
        self,
        network: Union[IntransitiveNetV1, IntransitiveNetV2],
        num_simulations: int = 100,
        c_puct: float = 1.5,
        fpu_reduction: float = 0.25,
        root_fpu_reduction: float = 0.05,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: Union[str, torch.device] = "auto",
        batch_size: Optional[int] = None,
        tablebase: Optional[TablebaseManager] = None,
        use_tactical_prover: bool = True,
        use_amp: bool = True,
        seed: Optional[int] = None,
        use_native: bool = True,
    ) -> None:
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.root_fpu_reduction = root_fpu_reduction
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        if isinstance(device, str) and device.strip().lower() == "auto":
            try:
                self.device = next(network.parameters()).device
            except StopIteration:
                self.device = resolve_device(device)
        else:
            self.device = resolve_device(device)
        if batch_size is None:
            self.batch_size = 4
        else:
            self.batch_size = max(1, batch_size)
        self.tablebase = tablebase
        self.use_tactical_prover = use_tactical_prover
        self.use_amp = use_amp
        self.seed = 42 if seed is None else int(seed)
        self.rng = np.random.default_rng(seed)

        self._native_handle: Optional[int] = None
        if use_native and NativeEngine.is_available():
            tb_h = self.tablebase.handle if (self.tablebase and self.tablebase.is_loaded()) else None
            self._native_handle = NativeEngine.mcts_create(
                tb_handle=tb_h,
                c_puct=self.c_puct,
                fpu_reduction=self.fpu_reduction,
                root_fpu_reduction=self.root_fpu_reduction,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_eps=self.dirichlet_eps,
                use_tactical_prover=self.use_tactical_prover,
                seed=self.seed,
            )
            self._obs_buf = np.empty((self.batch_size, 11, 9, 9), dtype=np.float32)
            self._mask_buf = np.empty((self.batch_size, ACTION_SIZE), dtype=np.uint8)

    def set_tablebase(self, tablebase: Optional[TablebaseManager]) -> None:
        self.tablebase = tablebase
        if self._native_handle is not None:
            tb_h = tablebase.handle if (tablebase and tablebase.is_loaded()) else None
            NativeEngine.mcts_set_tablebase(self._native_handle, tb_h)

    def close(self) -> None:
        if self._native_handle is not None and NativeEngine.is_available():
            NativeEngine.mcts_destroy(self._native_handle)
        self._native_handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def advance_root(self, canon_action: int) -> None:
        if self._native_handle is not None:
            NativeEngine.mcts_advance_root(self._native_handle, canon_action)

    def _run_native_sims(
        self,
        sims_target: int,
        add_noise: bool,
    ) -> int:
        assert self._native_handle is not None
        sims_done = 0
        while sims_done < sims_target:
            rem = sims_target - sims_done
            b_cnt, resolved_imm = NativeEngine.mcts_collect_batch(
                self._native_handle,
                self.batch_size,
                rem,
                add_noise,
                self._obs_buf,
                self._mask_buf,
            )
            sims_done += resolved_imm
            if b_cnt > 0:
                probs_np, vals_np = self.network.predict_batch_raw(
                    self._obs_buf[:b_cnt],
                    self._mask_buf[:b_cnt],
                    device=self.device,
                    use_amp=self.use_amp,
                )
                NativeEngine.mcts_apply_batch(
                    self._native_handle,
                    probs_np,
                    vals_np,
                    b_cnt,
                    self.batch_size > 1,
                    add_noise,
                )
                sims_done += b_cnt
            elif resolved_imm == 0:
                break
        return sims_done

    def search(
        self,
        board: Board,
        temperature: float = 1.0,
        add_noise: bool = False,
        min_pv_depth: int = 0,
        max_simulations: Optional[int] = None,
        try_reuse: bool = False,
        history_hashes: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Union[RootSummary, MCTSNode]]:
        """
        Run MCTS from `board` (reusing subtree if `try_reuse=True` and matching).
        When `min_pv_depth > 0`, adaptively extends search up to `max_simulations`
        if principal variation depth is below `min_pv_depth` or top-2 moves are close.
        """
        if self._native_handle is not None:
            hist_arr = (
                np.array(history_hashes, dtype=np.uint64)
                if history_hashes
                else None
            )
            NativeEngine.mcts_start_search(
                self._native_handle,
                board.grid,
                board.side,
                board.quiet,
                board.ply,
                try_reuse=try_reuse,
                add_noise=add_noise,
                history_hashes=hist_arr,
            )

            base_sims = max(1, self.num_simulations)
            hard_cap = max(base_sims, max_simulations) if max_simulations is not None else base_sims

            sims_done = self._run_native_sims(base_sims, add_noise=add_noise)
            pol, root_q, visits, pv_depth, proven, t1, t2 = NativeEngine.mcts_get_result(
                self._native_handle, temperature
            )

            # Adaptive depth extension in play mode
            while (
                min_pv_depth > 0
                and sims_done < hard_cap
                and proven == 0
                and (pv_depth < min_pv_depth or (t1 > 0 and t2 > 0.6 * t1))
            ):
                chunk = min(max(self.batch_size, 32), hard_cap - sims_done)
                added = self._run_native_sims(chunk, add_noise=False)
                if added == 0:
                    break
                sims_done += added
                pol, root_q, visits, pv_depth, proven, t1, t2 = NativeEngine.mcts_get_result(
                    self._native_handle, temperature
                )

            tb_str = self.tablebase.format_status(board) if (self.tablebase and self.tablebase.is_loaded()) else None
            summary = RootSummary(
                q_value=root_q,
                visit_count=visits,
                pv_depth=pv_depth,
                proven=proven,
                tb_status=tb_str,
            )
            return pol, summary

        return self._search_python(board, temperature=temperature, add_noise=add_noise)

    # ---- Upgraded Python Fallback (if native C++ binary has not been built yet) ----

    def _populate_children_from_probs(
        self, node: MCTSNode, probs: np.ndarray, net_value: float
    ) -> float:
        c_actions = canonical_legal_actions(node.board)
        winning_c_actions = []
        losing_c_actions = []
        child_boards: Dict[int, Board] = {}

        for c_act in c_actions:
            real_act = from_canonical_action(c_act, node.board.side)
            child_board = node.board.copy()
            child_board.push_action(real_act, validate=False)
            child_boards[c_act] = child_board
            c_res = child_board.result()
            if c_res is not None and c_res.winner == node.board.side:
                winning_c_actions.append(c_act)
            elif c_res is None and has_immediate_corner_win(child_board.grid, child_board.side):
                losing_c_actions.append(c_act)

        if winning_c_actions:
            win_prob = 1.0 / len(winning_c_actions)
            for c_act in c_actions:
                pr = win_prob if c_act in winning_c_actions else 1e-5
                cn = MCTSNode(board=child_boards[c_act], parent=node, prior=pr)
                if c_act in winning_c_actions:
                    cn.proven = 2  # child loses
                    cn.visit_count = 1
                    cn.value_sum = -1.0
                node.children[c_act] = cn
            node.is_expanded = True
            node.proven = 1
            return 1.0

        safe_actions = [a for a in c_actions if a not in losing_c_actions]
        if safe_actions and losing_c_actions:
            safe_probs = np.array([float(probs[a]) for a in safe_actions], dtype=np.float64)
            s_sum = float(np.sum(safe_probs))
            safe_probs = safe_probs / s_sum if s_sum > 0 else np.ones_like(safe_probs) / len(safe_probs)
            safe_map = {a: float(p) for a, p in zip(safe_actions, safe_probs)}

            for c_act in c_actions:
                if c_act in safe_map:
                    node.children[c_act] = MCTSNode(
                        board=child_boards[c_act], parent=node, prior=safe_map[c_act]
                    )
                else:
                    suicidal_node = MCTSNode(board=child_boards[c_act], parent=node, prior=1e-5)
                    suicidal_node.visit_count = 1
                    suicidal_node.value_sum = 1.0
                    suicidal_node.proven = 1
                    node.children[c_act] = suicidal_node
            node.is_expanded = True
            return net_value

        for c_act in c_actions:
            node.children[c_act] = MCTSNode(
                board=child_boards[c_act],
                parent=node,
                prior=float(probs[c_act]),
            )
        node.is_expanded = True
        return net_value

    def _expand(self, node: MCTSNode) -> float:
        res = node.board.result()
        if res is not None:
            node.is_expanded = True
            if res.winner == 0:
                node.proven = 3
                return 0.0
            win = res.winner == node.board.side
            node.proven = 1 if win else 2
            return 1.0 if win else -1.0

        if self.tablebase and self.tablebase.is_loaded():
            avail, wdl, _, _ = self.tablebase.probe(node.board)
            if avail:
                node.is_expanded = True
                node.proven = 1 if wdl > 0 else (2 if wdl < 0 else 3)
                return float(wdl)

        probs, leaf_value = self.network.predict(node.board, device=self.device)
        return self._populate_children_from_probs(node, probs, leaf_value)

    def _expand_batch(self, leaves: List[MCTSNode]) -> List[float]:
        values: List[float] = [0.0] * len(leaves)
        non_terminal_indices: List[int] = []
        non_terminal_boards: List[Board] = []

        for idx, node in enumerate(leaves):
            res = node.board.result()
            if res is not None:
                node.is_expanded = True
                if res.winner == 0:
                    node.proven = 3
                    values[idx] = 0.0
                else:
                    win = res.winner == node.board.side
                    node.proven = 1 if win else 2
                    values[idx] = 1.0 if win else -1.0
                continue
            if self.tablebase and self.tablebase.is_loaded():
                avail, wdl, _, _ = self.tablebase.probe(node.board)
                if avail:
                    node.is_expanded = True
                    node.proven = 1 if wdl > 0 else (2 if wdl < 0 else 3)
                    values[idx] = float(wdl)
                    continue
            non_terminal_indices.append(idx)
            non_terminal_boards.append(node.board)

        if non_terminal_boards:
            probs_list, net_vals = self.network.predict_batch(
                non_terminal_boards, device=self.device
            )
            for idx, probs, net_val in zip(non_terminal_indices, probs_list, net_vals):
                values[idx] = self._populate_children_from_probs(leaves[idx], probs, net_val)

        return values

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        if not root.children:
            return
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(root.children))
        eps = self.dirichlet_eps
        for child, n_val in zip(root.children.values(), noise):
            child.prior = (1.0 - eps) * child.prior + eps * float(n_val)

    def _select_child(self, node: MCTSNode, is_root: bool = False) -> MCTSNode:
        sqrt_total = math.sqrt(max(1, node.visit_count))
        visited_prior = sum(c.prior for c in node.children.values() if c.visit_count > 0)
        red = self.root_fpu_reduction if is_root else self.fpu_reduction
        fpu_val = max(-1.0, min(1.0, node.q_value - red * math.sqrt(max(0.0, visited_prior))))

        best_score = -1e12
        best_child: Optional[MCTSNode] = None
        for child in node.children.values():
            if child.proven == 2:
                return child
            q_val = (-child.q_value) if child.visit_count > 0 else fpu_val
            u_val = self.c_puct * child.prior * (sqrt_total / (1 + child.visit_count))
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None
        return best_child

    def _apply_virtual_loss(self, node: MCTSNode) -> None:
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visit_count += 1
            if curr.parent is not None:
                curr.value_sum += 1.0
            curr = curr.parent

    def _revert_virtual_loss(self, node: MCTSNode) -> None:
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visit_count -= 1
            if curr.parent is not None:
                curr.value_sum -= 1.0
            curr = curr.parent

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        curr: Optional[MCTSNode] = node
        val = value
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += val
            val = -val
            curr = curr.parent

    def _search_python(
        self,
        board: Board,
        temperature: float = 1.0,
        add_noise: bool = False,
    ) -> Tuple[np.ndarray, MCTSNode]:
        root = MCTSNode(board=board.copy())
        root_val = self._expand(root)
        self._backpropagate(root, root_val)

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        sims_done = 0
        while sims_done < self.num_simulations and root.proven == 0:
            current_batch = min(self.batch_size, self.num_simulations - sims_done)
            leaves: List[MCTSNode] = []
            seen_ids: Set[int] = set()

            for _ in range(current_batch):
                node = root
                while node.is_expanded and node.children:
                    node = self._select_child(node, is_root=(node is root))
                if id(node) in seen_ids:
                    break
                seen_ids.add(id(node))
                self._apply_virtual_loss(node)
                leaves.append(node)

            for leaf in leaves:
                self._revert_virtual_loss(leaf)

            leaf_values = self._expand_batch(leaves)
            for leaf, val in zip(leaves, leaf_values):
                self._backpropagate(leaf, val)

            sims_done += len(leaves)

        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        if not root.children:
            return policy, root

        c_actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in c_actions], dtype=np.float64)

        if temperature <= 1e-3:
            best_idx = int(np.argmax(visits))
            policy[c_actions[best_idx]] = 1.0
        else:
            scaled = np.power(visits, 1.0 / temperature)
            total = float(np.sum(scaled))
            probs = scaled / total if total > 0 else np.ones_like(scaled) / len(scaled)
            for a, p in zip(c_actions, probs):
                policy[a] = float(p)

        return policy, root


class AlphaZeroPlayer(Player):
    """Player wrapper using IntransitiveNet + Deep C++ MCTS + Subtree Reuse + 4-piece Tablebase."""

    def __init__(
        self,
        network: Union[IntransitiveNetV1, IntransitiveNetV2],
        num_simulations: int = 200,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        device: Union[str, torch.device] = "auto",
        name: str = "AlphaZeroBot",
        tablebase: Optional[TablebaseManager] = None,
        reuse_subtree: bool = True,
        min_pv_depth: int = 6,
        max_simulations: Optional[int] = None,
        use_tactical_prover: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name)
        self.device = resolve_device(device)
        self.network = network.to(self.device)
        self.tablebase = tablebase
        self.reuse_subtree = reuse_subtree
        self.min_pv_depth = min_pv_depth
        self.max_simulations = max_simulations
        self.mcts = MCTS(
            network=self.network,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=self.device,
            tablebase=tablebase,
            use_tactical_prover=use_tactical_prover,
            seed=seed,
        )
        self.temperature = temperature
        self.last_root: Optional[Union[RootSummary, MCTSNode]] = None
        self._last_ply: int = -1
        self._history_hashes: List[int] = []

    def reset(self) -> None:
        self.last_root = None
        self._last_ply = -1
        self._history_hashes.clear()

    def select_move(self, board: Board) -> str:
        # 1. Exact Endgame Tablebase lookup at root when <= max_pieces
        if self.tablebase is not None and self.tablebase.is_loaded():
            if int(np.sum(board.grid != 0)) <= self.tablebase.max_pieces:
                tb_move = self.tablebase.probe_best_move(board)
                if tb_move is not None:
                    _, wdl, dtz, _ = self.tablebase.probe(board)
                    self.last_root = RootSummary(
                        q_value=float(wdl),
                        visit_count=1,
                        pv_depth=dtz,
                        proven=(1 if wdl > 0 else (2 if wdl < 0 else 3)),
                        tb_status=self.tablebase.format_status(board),
                    )
                    self._last_ply = board.ply
                    return tb_move

        # Reset game history if a new game or non-forward board jump occurred
        if board.ply <= self._last_ply:
            self._history_hashes.clear()
        self._last_ply = board.ply

        max_sims = (
            self.max_simulations
            if self.max_simulations is not None
            else (max(self.mcts.num_simulations, 2 * self.mcts.num_simulations) if self.min_pv_depth > 0 else self.mcts.num_simulations)
        )

        policy, root = self.mcts.search(
            board,
            temperature=self.temperature,
            add_noise=False,
            min_pv_depth=self.min_pv_depth,
            max_simulations=max_sims,
            try_reuse=self.reuse_subtree,
            history_hashes=self._history_hashes,
        )
        self.last_root = root
        if self.temperature <= 1e-3:
            c_action = int(np.argmax(policy))
        else:
            p_sum = float(np.sum(policy))
            norm_p = policy / p_sum if p_sum > 0 else None
            c_action = int(self.mcts.rng.choice(ACTION_SIZE, p=norm_p)) if norm_p is not None else int(np.argmax(policy))

        if self.reuse_subtree:
            self.mcts.advance_root(c_action)

        real_action = from_canonical_action(c_action, board.side)
        return action_to_uci(real_action)
