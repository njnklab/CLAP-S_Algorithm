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
    """
    Represents a matching in a directed graph and provides methods to find and manipulate it.

    This class uses a variant of the Hopcroft-Karp algorithm to find a maximum matching
    in a directed graph, which is treated as a bipartite graph where edges go from a set of
    source nodes (U) to a set of destination nodes (V), and U and V are the same set of nodes.

    Attributes
    ----------
    graph : nx.DiGraph
        The directed graph on which the matching is performed.
    markedSrc : dict
        A mapping from a source node `u` to the destination node `v` it is matched with.
        If `markedSrc[u] == 0`, `u` is unmatched from the source side (a tail node).
    markedDes : dict
        A mapping from a destination node `v` to the source node `u` it is matched with.
        If `markedDes[v] == 0`, `v` is unmatched from the destination side (a driver node).
    driver_nodes : set
        The set of unmatched destination nodes (MDS - Minimum Dominating Set equivalent).
    tail_nodes : set
        The set of unmatched source nodes.
    all_driver_nodes : list
        A temporary list of all possible driver nodes, used during the BFS phase of HK algorithm.
    all_alternating_set : dict
        A cache mapping a driver node to its alternating reachable set.
    all_alternating_edges : dict
        A cache mapping a driver node to the edges forming the alternating paths from it.
    """
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
        """Returns a dictionary containing the current state of the matching."""
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
        """
        Executes the Hopcroft-Karp algorithm to find a maximum matching in the graph.

        This method iteratively finds augmenting paths using BFS and DFS phases
        and updates the matching until no more augmenting paths can be found.
        After finding the matching, it identifies the driver and tail nodes.
        """

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
        """
        Updates the sets of driver and tail nodes based on the current matching.

        A driver node is a node that is not matched as a destination.
        A tail node is a node that is not matched as a source.
        """

        self.driver_nodes = set([node for node, matched in self.markedDes.items() if matched == 0])
        self.tail_nodes = set([node for node, matched in self.markedSrc.items() if matched == 0])

    def _bfs(self):
        """
        The BFS phase of the Hopcroft-Karp algorithm.

        It builds layers of augmenting paths starting from all unmatched destination
        nodes (driver nodes). It computes distances from these drivers to other nodes
        along alternating paths.

        Returns
        -------
        bool
            True if at least one augmenting path is found, False otherwise.
        """

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
        """
        The DFS phase of the Hopcroft-Karp algorithm.

        It searches for an augmenting path from a given node `node` using the
        distance information computed by the BFS phase. If a path is found,
        it updates the matching along this path.

        Parameters
        ----------
        node : int | str
            The node to start the DFS from.

        Returns
        -------
        bool
            True if an augmenting path was found and the matching was updated, False otherwise.
        """

        for src, _ in self.graph.in_edges(node):
            if self.distSrc[src] == self.distDes[node] + 1:
                self.distSrc[src] = 0
                if self.markedSrc[src] == 0 or self._dfs(self.markedSrc[src]):
                    self.markedSrc[src] = node
                    self.markedDes[node] = src
                    return True
        return False

    def find_alternating_reachable_set(self, driver):
        """
        Finds the set of nodes reachable from a given driver node via an alternating path.

        An alternating path is a path that alternates between edges not in the matching
        and edges in the matching. This set represents potential new driver nodes if the
        matching is reconfigured.

        Parameters
        ----------
        driver : int | str
            The driver node to start the search from.

        Returns
        -------
        set
            The set of nodes reachable from the driver via alternating paths.
        """

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
        """
        Finds a set of potential driver nodes by exploring alternating paths in reverse.

        Starting from a matched node, this method follows the matching edge backward
        and then explores forward along non-matching edges to find other nodes that
        could become drivers. This is used in the context of two-layer networks
        where a matched node in one layer might be a driver in another.

        Parameters
        ----------
        matched : int | str
            A node that is currently matched as a destination.

        Returns
        -------
        set
            A set of nodes that can become drivers through a reversal alternating path.
        """

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
        """
        (Deprecated) Finds the set of nodes in the alternating path between a driver and a new driver.

        This method was intended to identify the specific nodes involved in an
        augmenting path that would make `new_driver` the new driver, excluding
        other reachable nodes.

        Parameters
        ----------
        driver : int | str
            The original driver node.
        new_driver : int | str
            The target new driver node from the alternating reachable set.

        Returns
        -------
        set
            The set of nodes on the specific alternating path.
        """

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
        """
        Finds and caches the alternating reachable sets for all current driver nodes.

        This method iterates through all driver nodes and computes their respective
        alternating reachable sets, using a cache to avoid redundant computations
        for nodes that are reachable from multiple drivers.

        Returns
        -------
        dict
            A dictionary mapping each driver node to its alternating reachable set.
        """

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
        """
        Updates the matching by augmenting along the alternating path from `driver` to `new_driver`.

        This operation effectively makes `new_driver` a driver node and `driver` a matched node
        by flipping the status of edges along the unique alternating path between them.

        Parameters
        ----------
        driver : int | str
            The original driver node.
        new_driver : int | str
            The node that will become the new driver. It must be in the alternating
            reachable set of `driver`.
        """

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
    """
    Manages and optimizes matchings across multiple network layers.

    This class holds a list of `Matching` objects, one for each layer of a
    multilayer network. It provides algorithms to analyze and minimize the
    discrepancy between the driver node sets (MDS) of different layers.

    Attributes
    ----------
    matchings : List[Matching]
        A list of `Matching` objects, where each object corresponds to a layer.
    """
    def __init__(self, matchings: List[Matching]):
        # Storing a list of Matching objects
        self.matchings = matchings

    @staticmethod
    def print_info(mds_list, phase):
        """
        Logs statistics about the driver node sets (MDS) of two layers.

        Calculates and prints the size of the union, intersection, and differences
        of the two MDS sets for diagnostic purposes.

        Parameters
        ----------
        mds_list : list[set]
            A list containing two sets of driver nodes.
        phase : str
            A label for the current stage of the algorithm (e.g., "start", "end").

        Returns
        -------
        tuple[int, int]
            A tuple containing the size of the union and intersection.
        """
        intersection = set.intersection(*mds_list)
        union = set.union(*mds_list)
        logger.debug("=" * 50)
        logger.debug(f"{phase}:\tunion size: {len(union)}\tintersetion size: {len(intersection)}")
        logger.debug(f"        \tmds_1 size: {len(mds_list[0])}\tmds_2 size: {len(mds_list[1])}")
        logger.debug(f"        \tdiff_1 size: {len(mds_list[0] - intersection)}\tdiff_2 size: {len(mds_list[1] - intersection)}")
        logger.debug("=" * 50)
        return len(union), len(intersection)

    class Layer(Enum):
        """Enumeration for the two layers of the network."""
        One = 1
        Two = 2

        def toggle(self):
            """Switches between Layer.One and Layer.Two."""
            return MultiMatching.Layer.Two if self == MultiMatching.Layer.One else MultiMatching.Layer.One

    class NodeType(Enum):
        """(Not actively used) Enumeration for node types based on driver status."""
        # is driver node in both layer
        BothDriver = 0
        # is matched node in both layer
        BothMatched = 1

        def toggle(self):
            """Switches between NodeType.BothDriver and NodeType.BothMatched."""
            return (
                MultiMatching.NodeType.BothMatched
                if self == MultiMatching.NodeType.BothDriver
                else MultiMatching.NodeType.BothDriver
            )

    def CLAPS(self, max_clap_length=0):
        """
        Controllability-based Layer-Aware Path Searching (CLAPS) algorithm.

        This algorithm attempts to reduce the symmetric difference between the driver
        node sets of two layers by finding and executing "exchange chains" (CLAPs).
        A CLAP is a sequence of alternating path augmentations across both layers
        that swaps a driver node from the symmetric difference of one layer with a
        driver node from the symmetric difference of the other.

        Parameters
        ----------
        max_clap_length : int, optional
            The maximum allowed length of an exchange chain. If `> 0`, the search
            for chains is depth-limited, which may result in an approximate solution.
            Defaults to 0 (unlimited depth).

        Returns
        -------
        tuple
            A tuple containing:
            - pre_diff_mds_1_size (int): Initial size of `D1 \\ D2`.
            - pre_diff_mds_2_size (int): Initial size of `D2 \\ D1`.
            - pre_union_size (int): Initial size of `D1 U D2`.
            - union_size (int): Final size of `D1 U D2`.
            - average_depth (float): Average length of the CLAPs found.
        """
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
        Samples up to K unique Maximum Driver Sets (MDS) from a single layer.

        It repeatedly runs the HK_algorithm and collects distinct sets of driver nodes.
        To avoid excessive computation, it stops after `K * tries_factor` attempts
        if K unique sets are not found.

        Parameters
        ----------
        matcher : Matching
            The `Matching` object for the layer.
        K : int
            The target number of unique MDS to collect.
        tries_factor : int, optional
            A multiplier for the maximum number of trials. Defaults to 5.

        Returns
        -------
        list[set]
            A list of unique driver node sets.
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
        Converts a list of node sets into bitmasks for efficient set operations.

        It first creates a unified index mapping for all nodes present in the sets.

        Parameters
        ----------
        sets : list[set]
            A list of node sets.

        Returns
        -------
        tuple[list[int], dict]
            A tuple containing the list of bitmasks and the node-to-index mapping.
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
        """
        Creates a unified node-to-index mapping from the union of all nodes in two collections of sets.

        Parameters
        ----------
        sets1 : list[set]
            First collection of node sets.
        sets2 : list[set]
            Second collection of node sets.

        Returns
        -------
        dict
            A dictionary mapping each unique node to an integer index.
        """
        all_nodes = set()
        for s in sets1:
            all_nodes |= s
        for s in sets2:
            all_nodes |= s
        return {node: i for i, node in enumerate(sorted(all_nodes))}

    def _sets_to_bitmasks_with_map(self, sets: list[set], node2idx: dict) -> list[int]:
        """
        Converts a list of node sets to bitmasks using a predefined node-to-index map.

        Parameters
        ----------
        sets : list[set]
            The list of node sets to convert.
        node2idx : dict
            The predefined mapping from nodes to integer indices.

        Returns
        -------
        list[int]
            A list of integer bitmasks.
        """
        masks: list[int] = []
        for s in sets:
            mask = 0
            for v in s:
                # 这里不应该出现 None；若出现则说明映射不全
                idx = node2idx[v]
                mask |= (1 << idx)
            masks.append(mask)
        return masks

    def RSU(self, K: int = 20, tries_factor: int = 5) -> int:
        """
        Finds the minimum union size of driver sets using a Randomized Sampling and Union approach.

        This method provides a fast and exact solution for the minimum union problem.
        It works by:
        1. Sampling up to `K` unique Maximum Driver Sets (MDS) from each of the two layers.
        2. Converting these sets to bitmasks for highly efficient computation.
        3. Performing an exhaustive search over all pairs of sampled MDS from the two layers.
        4. Using bitwise operations to calculate the union size and find the minimum.
        The search is optimized with pruning based on a running lower bound.

        Parameters
        ----------
        K : int, optional
            The number of unique MDS to sample from each layer. Defaults to 20.
        tries_factor : int, optional
            A multiplier for the sampling trials, to avoid excessive runs if unique
            MDS are rare. Defaults to 5.

        Returns
        -------
        int
            The minimum possible size of the union of driver sets `|D1 U D2|`.
        """
        if len(self.matchings) != 2:
            raise ValueError("RSU requires exactly two layers.")
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
        

    def CLAPG(self, max_steps: Optional[int] = None, randomize: bool = True) -> int:
        """
        A greedy algorithm for minimizing the driver set union via single-step exchanges.

        This method, "CLAP Greedy" (CLAPG), iteratively performs the most straightforward
        form of driver exchange. In each step, it looks for a driver node `u` in the
        symmetric difference of one layer (`D1 \\ D2` or `D2 \\ D1`) and an alternating
        path that leads to a node `v` that is a driver in the *other* layer.
        If such a pair `(u, v)` is found, the matching is updated, which is guaranteed
        to reduce the union size by one. The process repeats until no such single-step
        improvement can be found or `max_steps` is reached.

        Parameters
        ----------
        max_steps : Optional[int], optional
            The maximum number of greedy steps to perform. If None, runs until convergence.
            Defaults to None.
        randomize : bool, optional
            If True, shuffles the order of driver nodes to try, which can help escape
            local minima in some cases. Defaults to True.

        Returns
        -------
        int
            The final size of the union of driver sets `|D1 U D2|` after the greedy process.
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
            n_max: int = 10000,
            time_limit: Optional[float] = None,
            prefer: str = "ortools",
            budget_mode: str = "fixed",
            k1: Optional[int] = None,
            k2: Optional[int] = None,
            tighten_union: bool = True,
        ) -> int:
        """
        Finds the exact minimum union size using an Integer Linear Programming (ILP) formulation.

        This method provides a baseline for the exact optimal solution by modeling the
        two-layer maximum matching and driver set union problem as an ILP. It is suitable
        for small to medium-sized graphs.

        Parameters
        ----------
        n_max : int
            A safeguard threshold. The method will raise an error if the number of nodes
            exceeds this value. Defaults to 10000.
        time_limit : Optional[float]
            An optional time limit in seconds for the ILP solver. Defaults to None.
        prefer : {"ortools", "pulp", "auto"}
            The preferred ILP solver backend. "ortools" is generally faster.
            Defaults to "ortools".
        budget_mode : {"fixed", "at_most", "auto"}
            - "fixed": Enforces that the size of each driver set `|D_l|` must equal `k_l`.
            - "at_most": Enforces `|D_l| <= k_l`.
            - "auto": A two-stage optimization that first finds the minimum possible
              driver set sizes (`k1*`, `k2*`) and then minimizes the union size
              subject to these fixed budgets. This corresponds to finding the
              absolute minimum union over all possible maximum matchings.
        k1, k2 : Optional[int]
            The budget for each layer's driver set size. Used only when `budget_mode`
            is "fixed" or "at_most". If None, the size of the current driver set is used.
        tighten_union : bool
            Whether to add redundant constraints to tighten the linear relaxation,
            which can improve solver performance. Defaults to True.

        Returns
        -------
        int
            The exact minimum size of the union `|D1 U D2|`.

        Raises
        ------
        ValueError
            If the graph is too large or `budget_mode` is unknown.
        RuntimeError
            If the ILP solver fails or is not installed.
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
                prob = pulp.LpProblem("UDS_ILP", pulp.LpMinimize)
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


    def MI_exact(self):
        """
        Finds the exact minimum union size using a Matroid Intersection algorithm.

        This method implements a theoretically grounded, exact algorithm based on the
        intersection of two transversal matroids. The problem of minimizing `|D1 U D2|`
        is equivalent to maximizing `|X1 ∩ X2|`, where `X1` and `X2` are the sets of
        matched nodes (bases of the transversal matroids `M1` and `M2`). The algorithm finds
        the largest common independent set `S*` of `M1` and `M2`, then extends `S*` to
        bases `X1` and `X2` of `M1` and `M2` respectively. The final driver sets are
        `D1 = V \\ X1` and `D2 = V \\ X2`.

        This method is self-contained and does not require external solvers.

        Returns
        -------
        dict
            A dictionary containing detailed results of the intersection:
            - "V": The list of all nodes.
            - "S_common": The largest common independent set (node labels).
            - "X1", "X2": The bases of the two matroids (sets of matched nodes).
            - "D1", "D2": The resulting driver sets that achieve the minimum union.
            - "min_union_size": The size of the minimum union `|D1 U D2|`.
            - "union": The union set `D1 U D2`.
            - "sizes": A dict with cardinalities (`n`, `mu1`, `mu2`, `d1`, `d2`, `r_common`).
        """

        assert len(self.matchings) == 2, "MI_exact only supports duplex."

        # ---------- 内部依赖（保持方法内自包含） ----------
        from collections import deque
        from typing import List, Dict, Set, Tuple, Iterable, Optional

        class _HopcroftKarp:
            """最大匹配（左到右邻接），支持“右侧允许集”限制。"""
            __slots__ = ("n_left", "n_right", "adj")

            def __init__(self, n_left: int, n_right: int, adj: List[List[int]]):
                self.n_left = n_left
                self.n_right = n_right
                self.adj = adj

            def maximum_matching(
                self,
                allowed_right: Optional[Set[int]] = None,
                initial_match_L: Optional[List[int]] = None,
                initial_match_R: Optional[List[int]] = None,
            ) -> Tuple[int, List[int], List[int]]:
                nL, nR = self.n_left, self.n_right
                if allowed_right is None:
                    allowed_right = set(range(nR))

                if initial_match_L is None or initial_match_R is None:
                    matchL = [-1] * nL
                    matchR = [-1] * nR
                else:
                    matchL = initial_match_L[:]
                    matchR = initial_match_R[:]

                INF = 10**9
                dist = [INF] * nL

                def bfs() -> bool:
                    dq = deque()
                    for u in range(nL):
                        if matchL[u] == -1:
                            dist[u] = 0
                            dq.append(u)
                        else:
                            dist[u] = INF
                    reachable_free = False
                    while dq:
                        u = dq.popleft()
                        for v in self.adj[u]:
                            if v not in allowed_right:
                                continue
                            w = matchR[v]
                            if w == -1:
                                reachable_free = True
                            else:
                                if dist[w] == INF:
                                    dist[w] = dist[u] + 1
                                    dq.append(w)
                    return reachable_free

                def dfs(u: int) -> bool:
                    for v in self.adj[u]:
                        if v not in allowed_right:
                            continue
                        w = matchR[v]
                        if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                            matchL[u] = v
                            matchR[v] = u
                            return True
                    dist[u] = 10**9
                    return False

                matching_size = 0
                while bfs():
                    for u in range(nL):
                        if matchL[u] == -1 and dfs(u):
                            matching_size += 1

                return matching_size, matchL, matchR

        def _build_index(nodes: Iterable) -> Dict:
            nodes_list = list(dict.fromkeys(nodes))
            return {x: i for i, x in enumerate(nodes_list)}

        def _build_bipartite_from_directed(
            nodes: Iterable,
            edges: Iterable[Tuple],
        ) -> Tuple[_HopcroftKarp, Dict, Dict]:
            node_to_idx = _build_index(nodes)
            n = len(node_to_idx)
            adj = [[] for _ in range(n)]
            for (u, v) in edges:
                if u not in node_to_idx or v not in node_to_idx:
                    continue
                ui = node_to_idx[u]
                vi = node_to_idx[v]
                if vi not in adj[ui]:
                    adj[ui].append(vi)
            hk = _HopcroftKarp(n_left=n, n_right=n, adj=adj)
            idx_to_node = {i: x for x, i in node_to_idx.items()}
            return hk, idx_to_node, node_to_idx

        def _is_independent_transversal(hk: _HopcroftKarp, S_right: Set[int]) -> bool:
            """S 是否可被匹配饱和（横断拟阵独立）。"""
            if not S_right:
                return True
            msize, _, _ = hk.maximum_matching(allowed_right=S_right)
            return msize == len(S_right)

        def _matroid_intersection_max_common_independent(
            ground_right: List[int],
            indep1,
            indep2,
        ) -> Set[int]:
            """朴素 Edmonds 交换图 + 最短增广路，返回最大公共独立集 I。"""
            V = set(ground_right)
            I = set()

            # quick add
            improved = True
            while improved:
                improved = False
                for z in list(V - I):
                    if indep1(I | {z}) and indep2(I | {z}):
                        I.add(z)
                        improved = True

            while True:
                X1 = {z for z in (V - I) if indep1(I | {z})}
                X2 = {z for z in (V - I) if indep2(I | {z})}
                inter = X1 & X2
                if inter:
                    I.add(next(iter(inter)))
                    continue
                if not X1 or not X2:
                    break

                parents = {}
                visited_out, visited_in = set(), set()
                q = deque()
                for z in X1:
                    visited_out.add(z)
                    q.append(('out', z, None))

                found = False
                target_out = None
                while q and not found:
                    typ, node, par = q.popleft()
                    parents[(typ, node)] = par
                    if typ == 'out':
                        z = node
                        for y in I - visited_in:
                            if indep2((I - {y}) | {z}):
                                visited_in.add(y)
                                q.append(('in', y, ('out', z)))
                    else:
                        y = node
                        for zprime in (V - I) - visited_out:
                            if indep1((I - {y}) | {zprime}):
                                visited_out.add(zprime)
                                q.append(('out', zprime, ('in', y)))
                                if zprime in X2:
                                    target_out = zprime
                                    parents[('out', zprime)] = ('in', y)
                                    found = True
                                    break

                if not found:
                    break

                # 回溯增广（对称差）
                path = []
                cur = ('out', target_out)
                while cur is not None:
                    path.append(cur)
                    cur = parents.get(cur)
                path.reverse()
                for typ, node in path:
                    if typ == 'out':
                        I.add(node)
                    else:
                        if node in I:
                            I.remove(node)

                # 再 quick add 一次
                improved = True
                while improved:
                    improved = False
                    for z in list(V - I):
                        if indep1(I | {z}) and indep2(I | {z}):
                            I.add(z)
                            improved = True

            return I

        def _extend_to_base(hk: _HopcroftKarp, S: Set[int]) -> Set[int]:
            """把独立集 S 贪心扩展到某一基（匹配秩 μ）。"""
            n = hk.n_right
            allR = set(range(n))
            mu, _, _ = hk.maximum_matching(allowed_right=allR)
            X = set(S)
            if len(X) > mu:
                raise ValueError("S size > rank; not independent?")
            changed = True
            while len(X) < mu and changed:
                changed = False
                for z in list(allR - X):
                    if _is_independent_transversal(hk, X | {z}):
                        X.add(z)
                        changed = True
                        if len(X) == mu:
                            break
            if len(X) < mu:
                raise RuntimeError("Failed to extend to a base.")
            return X

        # ---------- 从现有两层 Matching 构造并求解 ----------
        m1, m2 = self.matchings
        V_labels = list(set(m1.graph.nodes) | set(m2.graph.nodes))
        E1 = list(m1.graph.edges())
        E2 = list(m2.graph.edges())

        hk1, idx_to_node, node_to_idx = _build_bipartite_from_directed(V_labels, E1)
        hk2, _, _ = _build_bipartite_from_directed(V_labels, E2)
        V_idx = list(range(len(node_to_idx)))

        indep1 = lambda S: _is_independent_transversal(hk1, set(S))
        indep2 = lambda S: _is_independent_transversal(hk2, set(S))

        # 最大公共独立集 S*
        S_star_idx = _matroid_intersection_max_common_independent(V_idx, indep1, indep2)

        # 将 S* 各自扩展到基 X^(1), X^(2)
        X1_idx = _extend_to_base(hk1, S_star_idx)
        X2_idx = _extend_to_base(hk2, S_star_idx)

        all_idx = set(V_idx)
        D1_idx = all_idx - X1_idx
        D2_idx = all_idx - X2_idx
        union_idx = D1_idx | D2_idx

        # 一些规模统计
        mu1, _, _ = hk1.maximum_matching(allowed_right=all_idx)
        mu2, _, _ = hk2.maximum_matching(allowed_right=all_idx)
        n = len(V_idx)
        d1 = n - mu1
        d2 = n - mu2
        r_common = len(S_star_idx)

        # 反解回原标签
        def _lab(s: Set[int]) -> Set:
            return {idx_to_node[i] for i in s}

        res = {
            "V": [idx_to_node[i] for i in V_idx],
            "S_common": _lab(S_star_idx),
            "X1": _lab(X1_idx),
            "X2": _lab(X2_idx),
            "D1": _lab(D1_idx),
            "D2": _lab(D2_idx),
            "min_union_size": len(union_idx),
            "union": _lab(union_idx),
            "sizes": {
                "n": n, "mu1": mu1, "mu2": mu2,
                "d1": d1, "d2": d2, "r_common": r_common
            }
        }
        return res
