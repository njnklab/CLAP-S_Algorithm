import config
from utils.generator import SFGenerator, ERGenerator
from utils.utils import read_network, setup_logger
from matching import Matching, MultiMatching
import copy
import time
import argparse
import pandas as pd

logger = setup_logger(__name__)


def get_graphs(args):
    """
    Generates or reads networks based on the provided arguments.

    :param args: The object containing arguments parsed from the command line.
    :return: A list of networkx.Graph objects.
    """
    if args.files:
        logger.info(f"Reading networks from files: {args.files}")
        graphs = [read_network(f, args.n) for f in args.files]
    else:
        if args.net_type.upper() == "SF":
            generator = SFGenerator(args.n, args.k)
        elif args.net_type.upper() == "ER":
            generator = ERGenerator(args.n, args.k)
        else:
            raise ValueError(f"Unknown network type: {args.net_type}")

        logger.info(
            f"Generating {args.layers}-layer {args.net_type.upper()} network (n={args.n}, k={args.k})"
        )
        graphs = generator.generate_networks(args.layers)

    return graphs


def run_algorithms(graphs, algorithms_to_run):
    """
    Runs a series of matching algorithms on the given networks.

    :param graphs: A list of networkx.Graph objects.
    :param algorithms_to_run: A list of algorithm names to run.
    :return: A tuple containing the results DataFrame and the initial union size.
    """
    matchings = []
    for g in graphs:
        matching = Matching(g)
        matching.HK_algorithm()
        matchings.append(matching)

    # Calculate and log the initial union size
    initial_union_size = 0
    if len(matchings) >= 2:
        initial_union_size = len(matchings[0].driver_nodes | matchings[1].driver_nodes)
    elif matchings:
        initial_union_size = len(matchings[0].driver_nodes)
    logger.info(f"Initial Union Size: {initial_union_size}")

    results = []

    # Define all available algorithms.
    # Each entry is a tuple: (function, result_key)
    # The result_key is used to extract the main result from the returned dictionary.
    available_algorithms = {
        "CLAPS": (lambda m: m.CLAPS(), "post_union"),
        "RSU": (lambda m: m.RSU(), "min_union_size"),
        "CLAPG": (lambda m: m.CLAPG(), "union_size"),
        "ILP": (lambda m: m.ILP_exact(budget_mode="auto"), "min_union_size"),
    }

    for name in algorithms_to_run:
        if name not in available_algorithms:
            logger.warning(f"Algorithm '{name}' not found, skipping.")
            continue

        multi_matching = MultiMatching(copy.deepcopy(matchings))
        algorithm_func, result_key = available_algorithms[name]

        logger.info(f"================ Running {name} ================")
        start_time = time.time()
        result_dict = algorithm_func(multi_matching)
        end_time = time.time()
        execution_time = end_time - start_time

        # Extract the main metric from the result dictionary.
        # For CLAPS, we care about post_union.
        # For RSU, CLAPG, MI, ILP, we care about the union size.
        if name == "CLAPS":
            main_result = result_dict[3]  # post_union
        else:
            main_result = result_dict

        logger.info(f"Result: {main_result}")
        logger.info(f"Execution time: {execution_time:.3f} seconds")

        results.append(
            {
                "Algorithm": name,
                "Union Size": main_result,
                "Time (s)": round(execution_time, 3),
            }
        )

    return pd.DataFrame(results), initial_union_size


def main():
    """
    Main function: parses command-line arguments, runs experiments, and prints results.
    """
    parser = argparse.ArgumentParser(
        description="Run Overlapping Community Matching algorithms.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Network parameters
    parser.add_argument(
        "-n", type=int, default=500, help="Number of nodes in the network."
    )
    parser.add_argument(
        "-k", type=float, default=3.0, help="Average degree for network generation."
    )
    parser.add_argument(
        "--net_type",
        type=str,
        default="ER",
        choices=["ER", "SF"],
        help="Type of network to generate (Erdos-Renyi or Scale-Free).",
    )
    parser.add_argument(
        "--layers", type=int, default=2, help="Number of network layers to generate."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="List of network file paths to read (overrides network generation).\nExample: --files assets/net/real/A.txt assets/net/real/B.txt",
    )

    # Algorithm selection
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["CLAPS", "RSU", "CLAPG", "ILP"],
        help='List of algorithms to run.\nAvailable: CLAPS, RSU, CLAPG, ILP.\nExample: --algos CLAPS RSU',
    )

    args = parser.parse_args()

    try:
        # 1. Get networks
        graphs = get_graphs(args)

        # 2. Run algorithms
        results_df, initial_union_size = run_algorithms(graphs, [algo.upper() for algo in args.algos])

        # 3. Display results
        print("\n================ Experiment Results ================")
        print(f"Initial Union Size: {initial_union_size}")
        print(results_df.to_string(index=False))
        print("=" * 40)

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
    
