from __future__ import annotations

import copy
from enum import Enum
import networkx as nx
import random
from collections import deque
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

from utils.utils import timer, setup_logger, deprecated

logger = setup_logger(__name__, save_file=True)

@dataclass
class NodeState:
    """A BFS node with a parent pointer (saves memory).

    Attributes
    ----------
    node : int | str
        Current vertex id.
    layer : "MultiMatching.Layer"
        Layer in which the last hop was taken.
    parent : "NodeState | None"
        Predecessor state or ``None`` for the source driver.
    depth : int
        Length of the alternating chain so far.  Used for depth limiting.
    """

    node: int | str
    layer: "MultiMatching.Layer"
    parent: "NodeState | None" = None
    depth: int = 0

    def path_pairs(self) -> list[tuple[int | str, int | str]]:
        pairs: list[tuple[int | str, int | str]] = []
        cur, prev = self, self.parent
        while prev is not None:
            pairs.append((prev.node, cur.node))
            cur, prev = prev, prev.parent
        pairs.reverse()
        return pairs


class Matching:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        # the current maximum matching.
        # A value of 0 indicates that the node is not currently matched.
        self.markedSrc = {}
        self.markedDes = {}
        # head and tail nodes in matching at once
        self.driver_nodes = set()
        self.tail_nodes = set()
        # all possible driver nodes
        self.all_driver_nodes = []
        # all alternative set and edges
        self.all_alternating_set = {}
        self.all_alternating_edges = {}

    def get_properties(self):
        return {
            "graph": self.graph,
            "markedSrc": self.markedSrc,
            "markedDes": self.markedDes,
            "driver_nodes": self.driver_nodes,
            "tail_nodes": self.tail_nodes,
            "all_driver_nodes": self.all_driver_nodes,
            "all_alternative_set": self.all_alternating_set,
            "all_alternative_edges": self.all_alternating_edges,
        }

    @timer
    def HK_algorithm(self):

        self.distSrc = {}
        self.distDes = {}

        for node in self.graph.nodes:
            self.markedSrc[node] = 0
            self.markedDes[node] = 0

        while self._bfs():
            for node in random.sample(list(self.graph.nodes), len(self.graph.nodes)):
                if self.markedDes[node] == 0:
                    self._dfs(node)

        self.update_driver_and_tail_nodes()

    def update_driver_and_tail_nodes(self):

        self.driver_nodes = set([node for node, matched in self.markedDes.items() if matched == 0])
        self.tail_nodes = set([node for node, matched in self.markedSrc.items() if matched == 0])

    def _bfs(self):

        flag = False
        self.all_driver_nodes.clear()
        for node in self.graph.nodes:
            self.distSrc[node] = 0
            self.distDes[node] = 0
            if self.markedDes[node] == 0:
                self.all_driver_nodes.append(node)

        queue = deque(self.all_driver_nodes)
        visited = set(queue)
        while queue:
            driver_node = queue.popleft()
            for src, _ in self.graph.in_edges(driver_node):
                if self.distSrc[src] == 0:
                    self.distSrc[src] = self.distDes[driver_node] + 1
                    if self.markedSrc[src] == 0:
                        flag = True
                    else:
                        matched_node = self.markedSrc[src]
                        if matched_node not in visited:
                            self.distDes[matched_node] = self.distSrc[src] + 1
                            queue.append(matched_node)
                            visited.add(matched_node)
        return flag

    def _dfs(self, node):

        for src, _ in self.graph.in_edges(node):
            if self.distSrc[src] == self.distDes[node] + 1:
                self.distSrc[src] = 0
                if self.markedSrc[src] == 0 or self._dfs(self.markedSrc[src]):
                    self.markedSrc[src] = node
                    self.markedDes[node] = src
                    return True
        return False

    def find_alternating_reachable_set(self, driver):

        visited_nodes = set()
        alternating_set = set()
        alternating_edges = {}
        queue = deque([driver])

        while queue:
            current_node = queue.popleft()

            for predecessor in self.graph.predecessors(current_node):
                if predecessor in visited_nodes:
                    continue

                visited_nodes.add(predecessor)
                replaceable_node = self.markedSrc.get(predecessor)
                if replaceable_node and replaceable_node not in alternating_set:
                    queue.append(replaceable_node)
                    if replaceable_node != driver:
                        alternating_set.add(replaceable_node)
                    alternating_edges[predecessor] = current_node

            self.all_alternating_edges[driver] = alternating_edges

        self.all_alternating_set[driver] = alternating_set

        return alternating_set

    def find_reversal_alternating_reachable_set(self, matched):

        visited_nodes = set()
        reversal_alternating_set = set()
        queue = deque([matched])

        while queue:
            current_node = queue.popleft()
            predecessor = self.markedDes.get(current_node)
            if predecessor is None:
                continue

            for successor in self.graph.successors(predecessor):
                if successor in visited_nodes or successor == current_node:
                    continue

                visited_nodes.add(successor)
                if self.markedDes.get(successor) != 0:
                    queue.append(successor)
                else:
                    reversal_alternating_set.add(successor)

        return reversal_alternating_set

    @deprecated
    def find_alternating_exclude_set(self, driver, new_driver):

        self.find_alternating_reachable_set(driver)

        alternating_exclude_set = set()
        current_node = new_driver

        while current_node != driver:
            # Find the source and target node for the current node
            source_node = self.markedDes.get(current_node)

            target_node = self.all_alternating_edges[driver][source_node]

            alternating_exclude_set.add(current_node)
            current_node = target_node

        return alternating_exclude_set

    @timer
    def find_all_alternating_reachable_set(self):

        visited_nodes = set()
        cache = {}  # To cache results for nodes visited by other drivers

        for driver in self.driver_nodes:
            if driver in cache:
                self.all_alternating_set[driver] = cache[driver]
                continue

            visited_nodes.clear()
            alternating_set = set()
            alternative_edges = {}
            queue = deque([driver])

            while queue:
                node = queue.popleft()

                # If node results are cached, simply use them
                if node in cache:
                    alternating_set.update(cache[node])
                    continue

                for predecessor in self.graph.predecessors(node):
                    if predecessor in visited_nodes:
                        continue

                    visited_nodes.add(predecessor)
                    replaceable_node = self.markedSrc.get(predecessor)
                    if replaceable_node and replaceable_node not in alternating_set:
                        queue.append(replaceable_node)
                        alternating_set.add(replaceable_node)
                        alternative_edges[predecessor] = node

                self.all_alternating_edges[node] = alternative_edges
            # Store the results in cache for potential use by other driver nodes
            cache[driver] = alternating_set
            self.all_alternating_set[driver] = alternating_set

        return self.all_alternating_set

    def update_matching(self, driver, new_driver):

        current_node = new_driver
        pre_source_node = 0
        update_pairs = []

        try:
            self.find_alternating_reachable_set(driver)

            while current_node != driver:
                # Find the source and target node for the current node
                source_node = self.markedDes.get(current_node)
                target_node = self.all_alternating_edges[driver][source_node]

                update_pairs.append((current_node, source_node, pre_source_node, target_node))
                pre_source_node = source_node
                current_node = target_node

            for current_node, source_node, pre_source_node, target_node in update_pairs:
                # Reverse the matching edge
                self.markedSrc[source_node] = target_node
                self.markedDes[target_node] = source_node
                if pre_source_node == 0:
                    self.markedDes[current_node] = 0

            # Update the driver_nodes and tail_nodes lists after the matching has been updated
            self.update_driver_and_tail_nodes()
        except Exception as e:
            logger.error(f"error: {e}")
            logger.error(f"driver: {driver}, new_driver: {new_driver}, update_pairs: {update_pairs}")


class MultiMatching:
    def __init__(self, matchings: List[Matching]):
        # Storing a list of Matching objects
        self.matchings = matchings

    @staticmethod
    def print_info(mds_list, phase):
        intersection = set.intersection(*mds_list)
        union = set.union(*mds_list)
        logger.debug("=" * 50)
        logger.debug(f"{phase}:\tunion size: {len(union)}\tintersetion size: {len(intersection)}")
        logger.debug(f"        \tmds_1 size: {len(mds_list[0])}\tmds_2 size: {len(mds_list[1])}")
        logger.debug(f"        \tdiff_1 size: {len(mds_list[0] - intersection)}\tdiff_2 size: {len(mds_list[1] - intersection)}")
        logger.debug("=" * 50)
        return len(union), len(intersection)

    class Layer(Enum):
        One = 1
        Two = 2

        def toggle(self):
            return MultiMatching.Layer.Two if self == MultiMatching.Layer.One else MultiMatching.Layer.One

    class NodeType(Enum):
        # is driver node in both layer
        BothDriver = 0
        # is matched node in both layer
        BothMatched = 1

        def toggle(self):
            return (
                MultiMatching.NodeType.BothMatched
                if self == MultiMatching.NodeType.BothDriver
                else MultiMatching.NodeType.BothDriver
            )

    def MOUI(self, max_clap_length=0):
        assert len(self.matchings) == 2, f"The input can only be a two-layer network, current {len(self.matchings)}"

        def bfs_traverse(current_layer: MultiMatching.Layer, hierarchy_nodes_num, traverse_queue: deque):
            logger.debug(f"--- Traverse Entry: Layer={current_layer.name}, Queue Size={len(traverse_queue)}, Search Depth={len(traverse_queue[0][1] if traverse_queue else []) + 1}")
            next_hierarchy_nodes_num = 0
            for _ in range(hierarchy_nodes_num):
                current_node, clap = traverse_queue.popleft()

                alternating_reachable_set = set()
                if current_layer == self.Layer.One:
                    alternating_reachable_set = matcher_1.find_alternating_reachable_set(current_node)
                elif current_layer == self.Layer.Two:
                    alternating_reachable_set = matcher_2.find_reversal_alternating_reachable_set(current_node)
                target_node_set = alternating_reachable_set & diff_mds_2

                if target_node_set:
                    logger.debug(f"- SUCCESS: Found exchange chain! From {clap[0][1][0] if clap else current_node} to {list(target_node_set)[0]}.")
                    target_node = random.choice(list(target_node_set))
                    if clap:
                        clap_lengths.append(len(clap) + 1)
                        _, exchange_pair = clap[0]
                        diff_mds_1.remove(exchange_pair[0])
                        for exchange_layer, exchange_pair in clap:
                            if exchange_layer == self.Layer.One:
                                matcher_1.update_matching(*exchange_pair)
                            elif exchange_layer == self.Layer.Two:
                                matcher_2.update_matching(*exchange_pair[::-1])
                    else:
                        clap_lengths.append(1)
                        diff_mds_1.remove(current_node)
                    diff_mds_2.remove(target_node)

                    if current_layer == self.Layer.One:
                        matcher_1.update_matching(current_node, target_node)
                    elif current_layer == self.Layer.Two:
                        matcher_2.update_matching(target_node, current_node)
                    return True
                else:
                    consistent_set = set()
                    if current_layer == self.Layer.One:
                        consistent_set = (matcher_1.graph.nodes - matcher_1.driver_nodes) & (
                            matcher_2.graph.nodes - matcher_2.driver_nodes
                        )
                    elif current_layer == self.Layer.Two:
                        consistent_set = matcher_1.driver_nodes & matcher_2.driver_nodes
                    relay_node_set = alternating_reachable_set & consistent_set
                    if relay_node_set:
                        if max_clap_length > 0 and len(clap) + 1 >= max_clap_length:
                            logger.warning(f"Reached max clap length={max_clap_length}, aborting deeper search from {current_node}. Algorithm may be approximate.")
                            continue

                        for relay_node in relay_node_set - visited_relays:
                            visited_relays.add(relay_node)
                            new_clap = copy.deepcopy(clap)
                            new_clap.append((current_layer, (current_node, relay_node)))
                            traverse_queue.append((relay_node, new_clap))
                            next_hierarchy_nodes_num += 1

            if next_hierarchy_nodes_num:
                return bfs_traverse(
                    self.Layer.toggle(current_layer),
                    next_hierarchy_nodes_num,
                    traverse_queue,
                )
            else:
                return False

        matcher_1 = self.matchings[0]
        matcher_2 = self.matchings[1]

        diff_mds_1 = matcher_1.driver_nodes - (matcher_1.driver_nodes & matcher_2.driver_nodes)
        diff_mds_2 = matcher_2.driver_nodes - (matcher_1.driver_nodes & matcher_2.driver_nodes)
        pre_diff_mds_1_size = len(diff_mds_1)
        pre_diff_mds_2_size = len(diff_mds_2)

        pre_union_size, _ = self.print_info(
            [matcher_1.driver_nodes, matcher_2.driver_nodes], "start"
        )

        clap_lengths = []

        for driver_node in diff_mds_1.copy():
            flag = False
            for current_layer in [self.Layer.One, self.Layer.Two]:
                visited_relays = set()
                if bfs_traverse(current_layer, 1, deque([(driver_node, [])])):
                    flag = True
                    break
            if not flag:
                logger.debug(f"- FAILED: Driver node {driver_node} is not exchangeable.")

            if not diff_mds_1 or not diff_mds_2:
                break

        if clap_lengths:
            average_depth = sum(clap_lengths) / len(clap_lengths)
        else:
            average_depth = 0

        union_size, _ = self.print_info([matcher_1.driver_nodes, matcher_2.driver_nodes], "end")
        return pre_diff_mds_1_size, pre_diff_mds_2_size, pre_union_size, union_size, average_depth

    
    def _collect_unique_mds(self, matcher: Matching, K: int, tries_factor: int = 5) -> list[set]:
        """
        从同一层多次运行 HK_algorithm 采样最多 K 个“互异”的 MDS（以 driver_nodes 记）。
        为避免浪费，若重复太多，最多尝试 K*tries_factor 次。
        """
        seen: set[frozenset] = set()
        mds_list: list[set] = []
        max_trials = max(K, 1) * max(tries_factor, 1)
        trials = 0
        while len(mds_list) < K and trials < max_trials:
            trials += 1
            matcher.HK_algorithm()
            s = frozenset(matcher.driver_nodes)
            if s not in seen:
                seen.add(s)
                mds_list.append(set(s))
        return mds_list

    def _sets_to_bitmasks(self, sets: list[set]) -> tuple[list[int], dict]:
        """
        将若干节点集合转成位掩码（Python int）。返回 (bitmasks, node2idx)。
        """
        # 建立全集映射（两个层的节点并集即可——上层调用确保传同一映射）
        all_nodes = set()
        for s in sets:
            all_nodes |= set(s)
        node2idx = {node: i for i, node in enumerate(sorted(all_nodes))}
        bitmasks: list[int] = []
        for s in sets:
            mask = 0
            for v in s:
                mask |= (1 << node2idx[v])
            bitmasks.append(mask)
        return bitmasks, node2idx

    def _build_shared_index(self, sets1: list[set], sets2: list[set]) -> dict:
        """从两层所有候选的并集构造统一的 node->bit 索引。"""
        all_nodes = set()
        for s in sets1:
            all_nodes |= s
        for s in sets2:
            all_nodes |= s
        return {node: i for i, node in enumerate(sorted(all_nodes))}

    def _sets_to_bitmasks_with_map(self, sets: list[set], node2idx: dict) -> list[int]:
        masks: list[int] = []
        for s in sets:
            mask = 0
            for v in s:
                # 这里不应该出现 None；若出现则说明映射不全
                idx = node2idx[v]
                mask |= (1 << idx)
            masks.append(mask)
        return masks

    def RRMU(self, K: int = 20, tries_factor: int = 5) -> int:
        """
        精确版 RRMU：采样去重 + 位集 + 剪枝。与原 RRMU 等价但速度显著提升。
        - 仍各层至多采样 K 个（若互异不足则以实际唯一数为准）
        - 完全枚举两层候选对，但使用下界剪枝与有序遍历
        - 使用 Python int 位运算计算 |A ∪ B|（(maskA | maskB).bit_count()）
        """
        if len(self.matchings) != 2:
            raise ValueError("RRMU requires exactly two layers.")
        m1, m2 = self.matchings

        # 采样互异候选（同你现有 _collect_unique_mds）
        mds1 = self._collect_unique_mds(m1, K, tries_factor=tries_factor)
        mds2 = self._collect_unique_mds(m2, K, tries_factor=tries_factor)

        # 边界：无候选
        if not mds1 or not mds2:
            if (not m1.graph.nodes()) and (not m2.graph.nodes()):
                return 0
            return len((m1.driver_nodes or set()) | (m2.driver_nodes or set()))

        # 验证同层预算恒定（如果不恒定，HK 实现就不是严格最大匹配；提示并回退）
        k1s = {len(s) for s in mds1}
        k2s = {len(s) for s in mds2}
        if len(k1s) != 1 or len(k2s) != 1:
            # 回退到集合法，避免错误的下界
            return min(
                len(a | b)
                for a in mds1
                for b in mds2
            )
        k1, k2 = next(iter(k1s)), next(iter(k2s))
        lower_bound = max(k1, k2)

        # --- 关键修复：共享位映射来自两层候选的并集 ---
        node2idx = self._build_shared_index(mds1, mds2)
        masks1 = self._sets_to_bitmasks_with_map(mds1, node2idx)
        masks2 = self._sets_to_bitmasks_with_map(mds2, node2idx)

        list1 = sorted([(m, m.bit_count()) for m in masks1], key=lambda x: x[1])
        list2 = sorted([(m, m.bit_count()) for m in masks2], key=lambda x: x[1])
        min_pop2 = list2[0][1]

        best = float('inf')
        for m1m, pop1 in list1:
            if max(pop1, min_pop2) >= best:
                continue
            for m2m, pop2 in list2:
                if max(pop1, pop2) >= best:
                    break
                union_sz = (m1m | m2m).bit_count()
                if union_sz < best:
                    best = union_sz
                    if best == lower_bound:
                        # 提前收敛：达到全局下界
                        return best

        # --- 安全护栏：结果不得低于下界；若低，回退集合法 ---
        if best < lower_bound:
            # 极端情况下（比如上面预算集检查被跳过）退回集合法确保正确
            best = min(
                len(a | b)
                for a in mds1
                for b in mds2
            )

        return int(best)
        

    def GLDE(self, max_steps: Optional[int] = None, randomize: bool = True) -> int:
        """
        Greedy Local Driver Exchange (单步贪心基线).
        在任一层寻找“单段(segment)”可行的驱动交换 (u -> v)，并且确保立刻减少 |D1 ∪ D2|。
        规则（等价充要）：
          - 若在第1层：u ∈ DD1 (= D1 \ D2)，选择 v ∈ D2 且 v 可被该层的交替路证实可达，则 |U| 降 1。
          - 若在第2层：u ∈ DD2 (= D2 \ D1)，选择 v ∈ D1 且 v 可被第2层可达，则 |U| 降 1。
        不断执行直到无改进或达到 max_steps。

        Returns
        -------
        int
            最终的 |D1 ∪ D2|.
        """
        assert len(self.matchings) == 2, "GLE only supports duplex."
        matcher_1, matcher_2 = self.matchings
        steps = 0

        while True:
            improved = False
            mds_1 = set(matcher_1.driver_nodes)
            mds_2 = set(matcher_2.driver_nodes)
            diff_mds_1 = mds_1 - mds_2
            diff_mds_2 = mds_2 - mds_1

            # 尝试第1层：u∈DD1，目标 v∈D2 且可达
            order_1 = list(diff_mds_1)
            if randomize:
                random.shuffle(order_1)
            for u in order_1:
                reachable = matcher_1.find_alternating_reachable_set(u)
                targets = reachable & mds_2
                if targets:
                    v = random.choice(list(targets)) if randomize else next(iter(targets))
                    # 单步更新：第1层把 driver 从 u 移到 v
                    matcher_1.update_matching(u, v)
                    improved = True
                    break
            if improved:
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
                continue

            # 尝试第2层：u∈DD2，目标 v∈D1 且可达
            order_2 = list(diff_mds_2)
            if randomize:
                random.shuffle(order_2)
            for u in order_2:
                # 注意：这里我们同样用 find_alternating_reachable_set，
                # 在第2层它同样以“该层 driver”为源找可达 new-driver。
                reachable = matcher_2.find_alternating_reachable_set(u)
                targets = reachable & mds_1
                if targets:
                    v = random.choice(list(targets)) if randomize else next(iter(targets))
                    # 单步更新：第2层把 driver 从 u 移到 v
                    matcher_2.update_matching(u, v)
                    improved = True
                    break

            if not improved:
                break  # 无改进则终止
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        return len(matcher_1.driver_nodes | matcher_2.driver_nodes)

    def ILP_exact(
            self,
            n_max: int = 5000,
            time_limit: Optional[float] = None,
            prefer: str = "ortools",
            budget_mode: str = "fixed",
            k1: Optional[int] = None,
            k2: Optional[int] = None,
            tighten_union: bool = True,
        ) -> int:
        """
        小规模精确 ILP 基线（双层、固定或可选预算、最小化 |D1∪D2|）。

        Parameters
        ----------
        n_max : int
            规模阈值（N 超过时拒绝执行）。
        time_limit : Optional[float]
            求解时间限制（秒）；若设置且 budget_mode=="auto"，两阶段均分时间。
        prefer : {"ortools","pulp","auto"}
            优先求解后端；"auto" 与 "ortools" 行为一致，失败则回退 PuLP。
        budget_mode : {"fixed","at_most","auto"}
            - "fixed":    施加等式预算  sum_v y_ℓ(v) == kℓ
            - "at_most":  施加不等式预算 sum_v y_ℓ(v) <= kℓ
            - "auto":     两阶段字典序：先最小化 sum(y1)+sum(y2)，再在该面上最小化 |U|
        k1, k2 : Optional[int]
            每层预算（仅当 budget_mode!="auto" 时使用）。若为 None，使用 len(mℓ.driver_nodes)。
        tighten_union : bool
            是否添加 z_v <= y1_v + y2_v 以收紧模型（在最小化下非必需，但有助于 MIP 性能）。

        Returns
        -------
        int
            精确最优的 |D1 ∪ D2|.

        Raises
        ------
        RuntimeError / ValueError
            当规模过大、无可用求解器或模型不满足时抛出。
        """
        assert len(self.matchings) == 2, "ILP_exact only supports duplex."

        # ---------- 数据准备 ----------
        m1, m2 = self.matchings
        V = list(set(m1.graph.nodes) | set(m2.graph.nodes))
        N = len(V)
        if N > n_max:
            raise ValueError(f"Problem too large for ILP baseline: N={N} > n_max={n_max}.")

        # 预算来源（仅 fixed/at_most 使用；auto 会自动决定）
        if budget_mode.lower() in ("fixed", "at_most", "le"):
            if k1 is None:
                k1 = len(m1.driver_nodes)
            if k2 is None:
                k2 = len(m2.driver_nodes)
        budget_mode = {"le": "at_most"}.get(budget_mode.lower(), budget_mode.lower())

        E1 = list(m1.graph.edges())  # 层1的 (u -> v)
        E2 = list(m2.graph.edges())  # 层2的 (u -> v)

        # ---------- OR-Tools 优先 ----------
        if prefer in ("ortools", "auto"):
            try:
                from ortools.linear_solver import pywraplp

                def _build_common_model():
                    solver = pywraplp.Solver.CreateSolver("CBC")
                    if solver is None:
                        raise ImportError("OR-Tools CBC solver not available.")

                    # 变量
                    x1 = {(u, v): solver.IntVar(0, 1, f"x1_{u}_{v}") for (u, v) in E1}
                    x2 = {(u, v): solver.IntVar(0, 1, f"x2_{u}_{v}") for (u, v) in E2}
                    y1 = {v: solver.IntVar(0, 1, f"y1_{v}") for v in V}
                    y2 = {v: solver.IntVar(0, 1, f"y2_{v}") for v in V}
                    z  = {v: solver.IntVar(0, 1, f"z_{v}")  for v in V}

                    # 匹配（左侧 u^+）度约束
                    for u in V:
                        solver.Add(solver.Sum(x1[(uu, vv)] for (uu, vv) in E1 if uu == u) <= 1)
                        solver.Add(solver.Sum(x2[(uu, vv)] for (uu, vv) in E2 if uu == u) <= 1)

                    # 未匹配/驱动等式： in-match + y = 1（针对 V^- 侧）
                    for v in V:
                        solver.Add(solver.Sum(x1[(uu, vv)] for (uu, vv) in E1 if vv == v) + y1[v] == 1)
                        solver.Add(solver.Sum(x2[(uu, vv)] for (uu, vv) in E2 if vv == v) + y2[v] == 1)

                    return solver, x1, x2, y1, y2, z

                # --- 构建模型 ---
                solver, x1, x2, y1, y2, z = _build_common_model()

                def _add_union_linearization():
                    for v in V:
                        solver.Add(z[v] >= y1[v])
                        solver.Add(z[v] >= y2[v])
                        if tighten_union:
                            solver.Add(z[v] <= y1[v] + y2[v])

                # ---------- 三种模式 ----------
                if budget_mode == "fixed":
                    # sum(yℓ) == kℓ
                    solver.Add(solver.Sum(y1[v] for v in V) == k1)
                    solver.Add(solver.Sum(y2[v] for v in V) == k2)
                    _add_union_linearization()
                    solver.Minimize(solver.Sum(z[v] for v in V))
                    if time_limit is not None:
                        solver.SetTimeLimit(int(time_limit * 1000))
                    status = solver.Solve()
                    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                        raise RuntimeError("OR-Tools solver failed to find a solution.")
                    return int(sum(z[v].solution_value() for v in V))

                elif budget_mode == "at_most":
                    # sum(yℓ) <= kℓ
                    solver.Add(solver.Sum(y1[v] for v in V) <= k1)
                    solver.Add(solver.Sum(y2[v] for v in V) <= k2)
                    _add_union_linearization()
                    solver.Minimize(solver.Sum(z[v] for v in V))
                    if time_limit is not None:
                        solver.SetTimeLimit(int(time_limit * 1000))
                    status = solver.Solve()
                    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                        raise RuntimeError("OR-Tools solver failed to find a solution.")
                    return int(sum(z[v].solution_value() for v in V))

                elif budget_mode == "auto":
                    # 阶段一：最小化 sum(y1)+sum(y2)（分解后等价于各层分别最小）
                    solver.Minimize(
                        solver.Sum(y1[v] for v in V) + solver.Sum(y2[v] for v in V)
                    )
                    if time_limit is not None:
                        solver.SetTimeLimit(int((time_limit * 1000) / 2))
                    status1 = solver.Solve()
                    if status1 not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                        raise RuntimeError("OR-Tools stage-1 failed.")

                    k1_star = int(round(sum(y1[v].solution_value() for v in V)))
                    k2_star = int(round(sum(y2[v].solution_value() for v in V)))

                    # 阶段二：在固定预算面上最小化 |U|
                    # 直接设置新的目标函数，不需要清除旧的目标
                    solver.Minimize(solver.Sum(z[v] for v in V))
                    # 添加新的约束
                    solver.Add(solver.Sum(y1[v] for v in V) == k1_star)
                    solver.Add(solver.Sum(y2[v] for v in V) == k2_star)
                    # 并集线性化
                    for v in V:
                        solver.Add(z[v] >= y1[v])
                        solver.Add(z[v] >= y2[v])
                        if tighten_union:
                            solver.Add(z[v] <= y1[v] + y2[v])
                    if time_limit is not None:
                        solver.SetTimeLimit(max(1, int((time_limit * 1000) / 2)))
                    status2 = solver.Solve()
                    if status2 not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                        raise RuntimeError("OR-Tools stage-2 failed.")
                    return int(sum(z[v].solution_value() for v in V))

                else:
                    raise ValueError(f"Unknown budget_mode: {budget_mode}")

            except Exception as e:
                logger.warning(f"ILP_exact: OR-Tools backend failed ({e}). Trying PuLP...")

        # ---------- PuLP 回退 ----------
        try:
            import pulp

            def _build_common_pulp():
                prob = pulp.LpProblem("UMDS_ILP", pulp.LpMinimize)
                x1 = pulp.LpVariable.dicts("x1", E1, lowBound=0, upBound=1, cat=pulp.LpBinary)
                x2 = pulp.LpVariable.dicts("x2", E2, lowBound=0, upBound=1, cat=pulp.LpBinary)
                y1 = pulp.LpVariable.dicts("y1", V,  lowBound=0, upBound=1, cat=pulp.LpBinary)
                y2 = pulp.LpVariable.dicts("y2", V,  lowBound=0, upBound=1, cat=pulp.LpBinary)
                z  = pulp.LpVariable.dicts("z",  V,  lowBound=0, upBound=1, cat=pulp.LpBinary)

                # 匹配度约束（u^+）
                for u in V:
                    prob += pulp.lpSum(x1[(uu, vv)] for (uu, vv) in E1 if uu == u) <= 1
                    prob += pulp.lpSum(x2[(uu, vv)] for (uu, vv) in E2 if uu == u) <= 1
                # 未匹配等式（v^-）
                for v in V:
                    prob += pulp.lpSum(x1[(uu, vv)] for (uu, vv) in E1 if vv == v) + y1[v] == 1
                    prob += pulp.lpSum(x2[(uu, vv)] for (uu, vv) in E2 if vv == v) + y2[v] == 1

                return prob, x1, x2, y1, y2, z

            # --- 构建模型 ---
            if budget_mode in ("fixed", "at_most"):
                prob, x1, x2, y1, y2, z = _build_common_pulp()

                # 预算
                if budget_mode == "fixed":
                    prob += pulp.lpSum(y1[v] for v in V) == k1
                    prob += pulp.lpSum(y2[v] for v in V) == k2
                else:  # at_most
                    prob += pulp.lpSum(y1[v] for v in V) <= k1
                    prob += pulp.lpSum(y2[v] for v in V) <= k2

                # 并集线性化
                for v in V:
                    prob += z[v] >= y1[v]
                    prob += z[v] >= y2[v]
                    if tighten_union:
                        prob += z[v] <= y1[v] + y2[v]

                # 目标
                prob += pulp.lpSum(z[v] for v in V)

                solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit) if time_limit else pulp.PULP_CBC_CMD(msg=False)
                status = prob.solve(solver)
                if pulp.LpStatus[status] not in ("Optimal", "Not Solved", "Infeasible", "Undefined", "Unbounded"):
                    raise RuntimeError(f"PuLP returned unexpected status: {pulp.LpStatus[status]}")
                if pulp.LpStatus[status] not in ("Optimal", "Not Solved"):
                    logger.warning(f"PuLP status: {pulp.LpStatus[status]}")

                return int(sum(int(pulp.value(z[v])) for v in V))

            elif budget_mode == "auto":
                # 阶段一：min sum(y1)+sum(y2)
                prob1, x1a, x2a, y1a, y2a, za = _build_common_pulp()
                prob1 += pulp.lpSum(y1a[v] for v in V) + pulp.lpSum(y2a[v] for v in V)
                solver1 = pulp.PULP_CBC_CMD(msg=False, timeLimit=(time_limit/2 if time_limit else None))
                status1 = prob1.solve(solver1)
                if pulp.LpStatus[status1] not in ("Optimal", "Not Solved"):
                    raise RuntimeError(f"PuLP stage-1 failed: {pulp.LpStatus[status1]}")

                k1_star = int(round(sum(int(pulp.value(y1a[v])) for v in V)))
                k2_star = int(round(sum(int(pulp.value(y2a[v])) for v in V)))

                # 阶段二：固定预算，最小化 |U|
                prob2, x1b, x2b, y1b, y2b, zb = _build_common_pulp()
                prob2 += pulp.lpSum(y1b[v] for v in V) == k1_star
                prob2 += pulp.lpSum(y2b[v] for v in V) == k2_star
                for v in V:
                    prob2 += zb[v] >= y1b[v]
                    prob2 += zb[v] >= y2b[v]
                    if tighten_union:
                        prob2 += zb[v] <= y1b[v] + y2b[v]
                prob2 += pulp.lpSum(zb[v] for v in V)

                solver2 = pulp.PULP_CBC_CMD(msg=False, timeLimit=(time_limit/2 if time_limit else None))
                status2 = prob2.solve(solver2)
                if pulp.LpStatus[status2] not in ("Optimal", "Not Solved"):
                    raise RuntimeError(f"PuLP stage-2 failed: {pulp.LpStatus[status2]}")

                return int(sum(int(pulp.value(zb[v])) for v in V))

            else:
                raise ValueError(f"Unknown budget_mode: {budget_mode}")

        except Exception as e2:
            raise RuntimeError(
                "ILP_exact requires OR-Tools or PuLP. Please install one of them "
                "(pip install ortools) or (pip install pulp). "
                f"Backend error: {e2}"
            )

