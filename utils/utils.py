import networkx as nx
import time
import logging
import config
import warnings
import functools
import os
import datetime

from config import LOG_PATH

# 所有生成的有向网络，节点 index 从 1 开始
# DiGraph: directed graphs woth self loops.


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    return wrapper


def deprecated(message="This function is deprecated and will be removed in future versions."):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            warnings.warn(
                f"Call to deprecated function {func.__name__}. {message}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapped_func
    return decorator


def setup_logger(name, save_file=False):
    """Create a logger with the specified name."""
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(config.LOGGING_LEVEL)  # Set the minimum logging level

    if logger.handlers:
        return logger

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(config.LOGGING_LEVEL)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    if save_file:
        os.makedirs(LOG_PATH, exist_ok=True)
        log_file_path = os.path.join(LOG_PATH, f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log")
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(config.LOGGING_LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def read_network(file_path: str, n) -> nx.DiGraph:
    """
    Read a network from a file and generate a NetworkX DiGraph.

    Parameters:
    - file_path (str): Path to the file containing the network data.

    Returns:
    - nx.Graph: Generated NetworkX DiGraph.
    """
    
    graph = nx.DiGraph()
    for i in range(1, n + 1):
        graph.add_node(i)

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            nodes = line.split()
            source, target = int(nodes[0]), int(nodes[1])
            graph.add_edge(source, target)

    return graph


def save_network(graph: nx.DiGraph, file_path: str) -> None:
    """
    Save a NetworkX DiGraph to a file.

    Parameters:
    - graph (nx.DiGraph): The network graph to be saved.
    - file_path (str): Path to the file where the network data will be saved.
    """
    
    edge_list = list(graph.edges)
    with open(file_path, 'w', encoding='utf-8') as file_object:
        for edge in edge_list:
            file_object.write(str(edge[0]) + "\t" + str(edge[1]) + "\n")


def create_output_file(result_columns, output_file_name=None):
    os.makedirs(config.RESULT_PATH, exist_ok=True)
    if output_file_name is None:
        output_file_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    else:
        output_file_name = f"{output_file_name}.csv"
    with open(os.path.join(config.RESULT_PATH, output_file_name), "w", encoding="utf-8") as output_file:
        output_file.write(",".join(result_columns) + "\n")
    return os.path.join(config.RESULT_PATH, output_file_name)