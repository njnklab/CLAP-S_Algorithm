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
import time

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
    generator = SFGenerator(n, k)
    generator = ERGenerator(n, k)
    graphs = generator.generate_networks(2)

    # net_type = "ER"
    # graph1 = read_network(f"{config.SYNTHETIC_NET_PATH}/{net_type}/{net_type}_n={n}_k={k}/base.txt", n)
    # graph2 = read_network(f"{config.SYNTHETIC_NET_PATH}/{net_type}/{net_type}_n={n}_k={k}/overlap=-1.txt", n)
    # graphs = [graph1, graph2]

    matchings = []
    for i in range(len(graphs)):
        matching = Matching(graphs[i])
        matching.HK_algorithm()
        matchings.append(matching)

    moui_matching = MultiMatching(matchings)
    rrmu_matching = copy.deepcopy(moui_matching)
    glde_matching = copy.deepcopy(moui_matching)
    ilp_matching = copy.deepcopy(moui_matching)

    print(f"================ MOUI ================")
    start_time = time.time()
    pre_diff_mds_1, pre_diff_mds_2, pre_union, post_union, avg_h = moui_matching.MOUI()
    end_time = time.time()
    time_1 = end_time - start_time
    print(f"pre_diff_mds_1: {pre_diff_mds_1}, pre_diff_mds_2: {pre_diff_mds_2}")
    print(f"pre_union: {pre_union}, post_union: {post_union}")
    print(f"avg_h: {avg_h}")
    
    print(f"================ RRMU ================")
    start_time = time.time()
    min_union_size = rrmu_matching.RRMU()
    end_time = time.time()
    time_2 = end_time - start_time
    print(f"min_union_size: {min_union_size}")

    print(f"================ GLDE ================")
    start_time = time.time()
    union_size = glde_matching.GLDE()
    end_time = time.time()
    time_3 = end_time - start_time
    print(f"union_size: {union_size}")

    print(f"================ ILP  ================")
    start_time = time.time()
    union_size = ilp_matching.ILP_exact(budget_mode="auto")
    end_time = time.time()
    time_4 = end_time - start_time
    print(f"union_size: {union_size}")

    print("================ Time ================")
    print(f"MOUI: {round(time_1, 3)}")
    print(f"RRMU: {round(time_2, 3)}")
    print(f"GLDE: {round(time_3, 3)}")
    print(f"ILP: {round(time_4, 3)}")

    if union_size < post_union:
        print("ILP union size is smaller than post_union!!!!")
        exit(1)
    


if __name__ == "__main__":
    n = 1000
    k = 5.0
    for i in range(20):
        test(n, k)
        print("=" * 100)
    
