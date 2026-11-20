"""Monte Carlo Tree Search (MCTS) with PUCT, Virtual-Loss Batching, and Action Masking."""

import math
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch

from intransitive.baselines.base import Player
from intransitive.core.board import Board
from intransitive.core.constants import ACTION_SIZE
from intransitive.core.move import action_to_uci
from intransitive.core.rules import has_immediate_corner_win
from intransitive.env.canonical import (
    canonical_legal_actions,
    from_canonical_action,
)
from intransitive.rl.device import resolve_device
from intransitive.rl.network import IntransitiveNet


class MCTSNode:
    """Single node in the Monte Carlo Search Tree."""

    __slots__ = ("board", "parent", "prior", "children", "visit_count", "value_sum", "is_expanded")

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

    @property
    def q_value(self) -> float:
        """Mean value from `self.board.side`'s perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """
    AlphaZero-style Monte Carlo Tree Search using PUCT.
    Supports both single-leaf evaluation and virtual-loss batched evaluation
    for hardware accelerators (Intel XPU / CUDA).
    """

    def __init__(
        self,
        network: IntransitiveNet,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: Union[str, torch.device] = "auto",
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = resolve_device(device)
        # Default to batched evaluation (8) on XPU/CUDA, or 4 on CPU
        if batch_size is None:
            self.batch_size = 8 if self.device.type in ("xpu", "cuda") else 4
        else:
            self.batch_size = max(1, batch_size)
        self.rng = np.random.default_rng(seed)

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
                node.children[c_act] = MCTSNode(
                    board=child_boards[c_act], parent=node, prior=pr
                )
            node.is_expanded = True
            return 1.0

        # If some moves allow an immediate opponent corner win, suppress them if safe alternatives exist
        safe_actions = [a for a in c_actions if a not in losing_c_actions]
        if safe_actions and losing_c_actions:
            safe_probs = np.array([float(probs[a]) for a in safe_actions], dtype=np.float64)
            s_sum = float(np.sum(safe_probs))
            if s_sum > 0:
                safe_probs = safe_probs / s_sum
            else:
                safe_probs = np.ones_like(safe_probs) / len(safe_probs)
            safe_map = {a: float(p) for a, p in zip(safe_actions, safe_probs)}

            for c_act in c_actions:
                if c_act in safe_map:
                    node.children[c_act] = MCTSNode(
                        board=child_boards[c_act], parent=node, prior=safe_map[c_act]
                    )
                else:
                    suicidal_node = MCTSNode(board=child_boards[c_act], parent=node, prior=1e-5)
                    suicidal_node.visit_count = 1
                    suicidal_node.value_sum = 1.0  # +1.0 for opponent = -1.0 for current player
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
        """Expand a single leaf node."""
        res = node.board.result()
        if res is not None:
            node.is_expanded = True
            if res.winner == 0:
                return 0.0
            return 1.0 if res.winner == node.board.side else -1.0

        probs, leaf_value = self.network.predict(node.board, device=self.device)
        return self._populate_children_from_probs(node, probs, leaf_value)

    def _expand_batch(self, leaves: List[MCTSNode]) -> List[float]:
        """Expand a batch of leaf nodes with a single batched network forward pass."""
        values: List[float] = [0.0] * len(leaves)
        non_terminal_indices: List[int] = []
        non_terminal_boards: List[Board] = []

        for idx, node in enumerate(leaves):
            res = node.board.result()
            if res is not None:
                node.is_expanded = True
                if res.winner == 0:
                    values[idx] = 0.0
                else:
                    values[idx] = 1.0 if res.winner == node.board.side else -1.0
            else:
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

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        sqrt_total = math.sqrt(max(1, node.visit_count))
        best_score = -1e12
        best_child: Optional[MCTSNode] = None

        for child in node.children.values():
            q_val = -child.q_value
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
                # Increase curr.value_sum so parent (-curr.q_value) sees a penalty
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

    def search(
        self,
        board: Board,
        temperature: float = 1.0,
        add_noise: bool = False,
    ) -> Tuple[np.ndarray, MCTSNode]:
        """
        Run `num_simulations` MCTS rollouts from `board`.
        Returns:
          - policy_target: (648,) float32 probability distribution in canonical coordinates
          - root: the expanded root MCTSNode
        """
        root = MCTSNode(board=board.copy())
        root_val = self._expand(root)
        self._backpropagate(root, root_val)

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        sims_done = 0
        while sims_done < self.num_simulations:
            current_batch = min(self.batch_size, self.num_simulations - sims_done)
            leaves: List[MCTSNode] = []
            seen_ids: Set[int] = set()

            for _ in range(current_batch):
                node = root
                while node.is_expanded and node.children:
                    node = self._select_child(node)
                if id(node) in seen_ids:
                    # Collision on an unexpanded leaf: evaluate gathered batch first
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
            if total > 0:
                probs = scaled / total
            else:
                probs = np.ones_like(scaled) / len(scaled)
            for a, p in zip(c_actions, probs):
                policy[a] = float(p)

        return policy, root


class AlphaZeroPlayer(Player):
    """Player wrapper that uses IntransitiveNet + MCTS to select moves."""

    def __init__(
        self,
        network: IntransitiveNet,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        device: Union[str, torch.device] = "auto",
        name: str = "AlphaZeroBot",
    ) -> None:
        super().__init__(name=name)
        self.device = resolve_device(device)
        self.network = network.to(self.device)
        self.mcts = MCTS(
            network=self.network,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=self.device,
        )
        self.temperature = temperature
        self.last_root: Optional[MCTSNode] = None

    def select_move(self, board: Board) -> str:
        policy, root = self.mcts.search(
            board, temperature=self.temperature, add_noise=False
        )
        self.last_root = root
        if self.temperature <= 1e-3:
            c_action = int(np.argmax(policy))
        else:
            c_action = int(self.mcts.rng.choice(ACTION_SIZE, p=policy))
        real_action = from_canonical_action(c_action, board.side)
        return action_to_uci(real_action)
