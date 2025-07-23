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

    
    def RRMU(self, K: int = 20) -> int:
        """
        Repeated Random MDS Union (RRMU) baseline.
        Finds K MDSs for each layer and returns the size of the minimum union among all pairs.
        """
        if len(self.matchings) != 2:
            logger.error("RRMU baseline is implemented for exactly two layers.")
            raise ValueError("RRMU baseline requires exactly two Matching objects.")

        matcher_1 = self.matchings[0]
        matcher_2 = self.matchings[1]

        mds_list_layer1 = []
        mds_list_layer2 = []

        for _ in range(K):
            matcher_1.HK_algorithm() 
            mds_list_layer1.append(matcher_1.driver_nodes.copy())


        for _ in range(K):
            matcher_2.HK_algorithm()
            mds_list_layer2.append(matcher_2.driver_nodes.copy())


        min_union_size = float('inf')
        
        if not mds_list_layer1 or not mds_list_layer2:
            logger.warning("RRMU: One or both MDS lists are empty. Returning infinity or 0 if appropriate.")
            if not matcher_1.graph.nodes() and not matcher_2.graph.nodes():
                 min_union_size = 0

        for mds1 in mds_list_layer1:
            for mds2 in mds_list_layer2:
                current_union_size = len(mds1.union(mds2))
                if current_union_size < min_union_size:
                    min_union_size = current_union_size
        
        return min_union_size


    @deprecated
    def find_UMDS_deprecated(self):
        assert len(self.matchings) == 2, f"The input can only be a two-layer network, current {len(self.matchings)}"

        def traverse(current_layer, current_node_type, hierarchy_nodes_num, queue: deque):
            next_hierarchy_nodes_num = 0
            # 遍历当前层次的所有点
            for _ in range(hierarchy_nodes_num):
                current_node, exchange_history = queue.popleft()

                exclude_set = set()
                if len(exchange_history) >= 2:
                    for exchange_layer, exchange_pair in exchange_history:
                        if exchange_layer == current_layer:
                            exclude_set.update(set([exchange_pair[-1]]))

                if current_node_type == self.NodeType.BothMatched:
                    target_node_set = matcher_2.find_reversal_alternating_reachable_set(current_node) & diff_mds_2
                elif current_node_type == self.NodeType.BothDriver:
                    target_node_set = matcher_1.find_alternating_reachable_set(current_node) & diff_mds_2

                target_node_set -= exclude_set

                if target_node_set:
                    target_node = random.choice(list(target_node_set))
                    if exchange_history:
                        _, exchange_pair = exchange_history[0]
                        diff_mds_1.remove(exchange_pair[0])
                        for exchange_layer, exchange_pair in exchange_history:
                            if exchange_layer == self.Layer.One:
                                matcher_1.update_matching(*exchange_pair)
                            elif exchange_layer == self.Layer.Two:
                                matcher_2.update_matching(*exchange_pair[::-1])
                    else:
                        diff_mds_1.remove(current_node)
                    diff_mds_2.remove(target_node)

                    if current_layer == self.Layer.One:
                        matcher_1.update_matching(current_node, target_node)
                    elif current_layer == self.Layer.Two:
                        matcher_2.update_matching(target_node, current_node)
                    return True
                else:
                    inner_node_type = self.NodeType.toggle(current_node_type)
                    if inner_node_type == self.NodeType.BothMatched:
                        # consistent matched set
                        cms = (matcher_1.graph.nodes - matcher_1.driver_nodes) & (
                            matcher_2.graph.nodes - matcher_2.driver_nodes
                        )
                        inner_node_set = matcher_1.find_alternating_reachable_set(current_node) & cms
                    elif inner_node_type == self.NodeType.BothDriver:
                        # consistent driver set
                        cds = matcher_1.driver_nodes & matcher_2.driver_nodes
                        inner_node_set = matcher_2.find_reversal_alternating_reachable_set(current_node) & cds

                    inner_node_set -= exclude_set
                    if inner_node_set:
                        for inner_node in inner_node_set:
                            new_exchange_history = copy.deepcopy(exchange_history)
                            new_exchange_history.append((current_layer, (current_node, inner_node)))
                            queue.append((inner_node, new_exchange_history))
                            next_hierarchy_nodes_num += 1

            if next_hierarchy_nodes_num:
                return traverse(
                    self.Layer.toggle(current_layer),
                    self.NodeType.toggle(current_node_type),
                    next_hierarchy_nodes_num,
                    queue,
                )
            else:
                return False

        matcher_1 = self.matchings[0]
        matcher_2 = self.matchings[1]

        diff_mds_1 = matcher_1.driver_nodes - (matcher_1.driver_nodes & matcher_2.driver_nodes)
        diff_mds_2 = matcher_2.driver_nodes - (matcher_1.driver_nodes & matcher_2.driver_nodes)

        pre_union_size, pre_intersection_size = self.print_info(
            [matcher_1.driver_nodes, matcher_2.driver_nodes], "start"
        )

        while True:
            pre_same_size = len(matcher_1.driver_nodes & matcher_2.driver_nodes)

            for current_layer, current_node_type in [
                (self.Layer.One, self.NodeType.BothDriver),
                (self.Layer.Two, self.NodeType.BothMatched),
            ]:
                if traverse(
                    current_layer, current_node_type, len(diff_mds_1), deque([(node, []) for node in diff_mds_1])
                ):
                    break

            if (
                not diff_mds_1
                or not diff_mds_2
                or len(matcher_1.driver_nodes & matcher_2.driver_nodes) == pre_same_size
            ):
                break

        union_size, intersection_size = self.print_info([matcher_1.driver_nodes, matcher_2.driver_nodes], "end")
        return pre_intersection_size, intersection_size, pre_union_size, union_size