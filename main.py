import config
from utils.generator import SFGenerator, ERGenerator
from utils.utils import read_network, setup_logger
from utils.plot import (
    transform_graphs,
    visualize_networks_with_bipartite,
    visualize_networks_with_matching_and_bipartite,
)
from matching import Matching, MultiMatching
import copy
import os

logger = setup_logger(__name__)


# def test_visualize(n):
#     # generator = ERGenerator(10, 1.5)
#     # generator = BAGenerator(8, 1)
#     # graphs = generator.generate_networks(config.NETWORK_LAYER)

#     # graph1 = read_network(f"./assets/test_net/n={n}_k=2_1.txt", n)
#     # graph2 = read_network(f"./assets/test_net/n={n}_k=2_2.txt", n)

#     graph1 = read_network(f"assets/test_net/16_double_test_1", n)
#     graph2 = read_network(f"assets/test_net/16_double_test_2", n)
#     graphs = [graph1, graph2]

#     bi_graphs = transform_graphs(graphs)
#     visualize_networks_with_bipartite(graphs, bi_graphs)

#     matched_edges_list = []
#     mds_list = []
#     bi_matched_edges_list = []
#     bi_mds_list = []

#     def update_maximum_matching(matching: Matching):
#         matched_edges, mds = matching.find_maximum_matching_and_MDS()
#         bi_matched_edges, bi_mds = matching.transform_maximum_matching_and_MDS()
#         matched_edges_list.append(matched_edges)
#         mds_list.append(mds)
#         bi_matched_edges_list.append(bi_matched_edges)
#         bi_mds_list.append(bi_mds)
#         logger.debug(matching.get_properties())

#     graphs_show = []
#     bi_graphs_show = []
#     matchings = []

#     for i in range(len(graphs)):
#         matching = Matching(graphs[i])
#         matching.HK_algorithm()

#         update_maximum_matching(matching)
#         graphs_show.append(graphs[i])
#         bi_graphs_show.append(bi_graphs[i])

#         all_alternative_set = matching.find_all_alternating_reachable_set()
#         logger.debug(f"all alternative set: {all_alternative_set}")
#         logger.debug(f"all alternative edges: {matching.all_alternating_edges}")

#         matchings.append(matching)

#     matchings_2 = copy.deepcopy(matchings)

#     multi_matching = MultiMatching(matchings)
#     multi_matching.find_MSS()

#     for i in range(len(matchings)):
#         update_maximum_matching(matchings[i])
#         graphs_show.append(graphs[i])
#         bi_graphs_show.append(bi_graphs[i])

#     multi_matching = MultiMatching(matchings_2)
#     multi_matching.find_MSS_multi_new()

#     visualize_networks_with_matching_and_bipartite(
#         graphs_show, bi_graphs_show, matched_edges_list, bi_matched_edges_list, mds_list, bi_mds_list
#     )


def test(n=1000, k=2):
    # generator = BAGenerator(n, 2)
    # generator = ERGenerator(n, 2)
    # graphs = generator.generate_networks(config.NETWORK_LAYER)

    # network_type = "ER"
    # dir = f"{network_type}_n={n}_k={k}"
    # graphs = []
    # for file in os.listdir(os.path.join(config.SYNTHETIC_NET_PATH, network_type, dir)):
    #     if "base" in file or "0.1" in file:
    #         graph = read_network(os.path.join(config.SYNTHETIC_NET_PATH, network_type, dir, file), n)
    #         graphs.append(graph)

    net_type = "ER"
    graph1 = read_network(f"{config.SYNTHETIC_NET_PATH}/{net_type}/{net_type}_n={n}_k={k}/base.txt", n)
    graph2 = read_network(f"{config.SYNTHETIC_NET_PATH}/{net_type}/{net_type}_n={n}_k={k}/overlap=-1.txt", n)
    graphs = [graph1, graph2]

    # graph1 = read_network(f"{config.TEST_NET_PATH}/n={n}_k={k}_1.txt", n)
    # graph2 = read_network(f"{config.TEST_NET_PATH}/n={n}_k={k}_2.txt", n)
    # graphs = [graph1, graph2]

    # graph1 = read_network(f"{config.TEST_NET_PATH}/ap_confict_1.txt", n)
    # graph2 = read_network(f"{config.TEST_NET_PATH}/ap_confict_2.txt", n)
    # graphs = [graph1, graph2]

    matchings = []
    for i in range(len(graphs)):
        matching = Matching(graphs[i])
        matching.HK_algorithm()
        matching.find_all_alternating_reachable_set()
        matchings.append(matching)

    multi_matching = MultiMatching(matchings)
    pre_diff_mds_1, pre_diff_mds_2, pre_union, post_union, avg_h = multi_matching.MOUI()
    
    print(f"pre_diff_mds_1: {pre_diff_mds_1}, pre_diff_mds_2: {pre_diff_mds_2}")
    print(f"pre_union: {pre_union}, post_union: {post_union}")
    print(f"avg_h: {avg_h}")


# def test_compare(n):
#     graph1 = read_network(f"./assets/test_net/n={n}_k=2_1.txt", n)
#     graph2 = read_network(f"./assets/test_net/n={n}_k=2_2.txt", n)
#     graphs = [graph1, graph2]

#     matchings_1 = []
#     for i in range(len(graphs)):
#         matching = Matching(graphs[i])
#         matching.HK_algorithm()

#         all_alternative_set = matching.find_all_alternating_reachable_set()
#         logger.debug(f"all alternative set: {all_alternative_set}")
#         logger.debug(f"all alternative edges: {matching.all_alternating_edges}")

#         matchings_1.append(matching)

#     matchings_2 = copy.deepcopy(matchings_1)
#     matchings_3 = copy.deepcopy(matchings_1)

#     multi_matching = MultiMatching(matchings_1)
#     intersection_size_1, elapsed_time_1 = multi_matching.find_MSS_old()

#     multi_matching = MultiMatching(matchings_2)
#     intersection_size_2, elapsed_time_2 = multi_matching.find_MSS()

#     multi_matching = MultiMatching(matchings_3)
#     intersection_size_3, elapsed_time_3 = multi_matching.find_UMDS_old()

#     return [intersection_size_1, intersection_size_2, intersection_size_3], [
#         elapsed_time_1,
#         elapsed_time_2,
#         elapsed_time_3,
#     ]


if __name__ == "__main__":
    n = 1000
    k = 6.0
    for i in range(10):
        test(n, k)
        print("=" * 100)
    
