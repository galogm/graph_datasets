"""Load Graph Datasets
"""
# pylint:disable=protected-access
import ssl
from typing import Tuple

import dgl
import torch

from .data_info import COLA_DATASETS
from .data_info import CRITICAL_DATASETS
from .data_info import DEFAULT_DATA_DIR
from .data_info import DGL_DATASETS
from .data_info import LINKX_DATASETS
from .data_info import OGB_DATASETS
from .data_info import PYG_DATASETS
from .data_info import SDCN_DATASETS
from .datasets import load_cola_data
from .datasets import load_critical_dataset
from .datasets import load_dgl_data
from .datasets import load_linkx_data
from .datasets import load_ogb_data
from .datasets import load_pyg_data
from .datasets import load_sdcn_data
from .utils import print_dataset_info


def load_data(
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
    verbosity: int = 0,
    source: str = "pyg",
    rm_self_loop: bool = True,
    to_simple: bool = True,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    """Load graphs.

    Args:
        dataset_name (str): Dataset name.
        directory (str, optional): Raw dir for loading or saving. \
            Defaults to DEFAULT_DATA_DIR=os.path.abspath("./data").
        verbosity (int, optional): Output debug information. \
            The greater, the more detailed. Defaults to 0.
        source (str, optional): Source for data loading. Defaults to "pyg".
        rm_self_loop (str, optional): Remove self loops. Defaults to True.
        to_simple (str, optional): Convert to a simple graph with no duplicate undirected edges.

    Raises:
        NotImplementedError: Dataset unknown.

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, int]: [graph, label, n_clusters]

    Example:
        .. code-block:: python

            from graph_datasets import load_data
            graph, label, n_clusters = load_data(
                dataset_name='cora',
                directory="./data",
                source='pyg',
                verbosity=3,
            )
    """
    dataset_name = (
        dataset_name.lower() if dataset_name not in [
            "papers100M",
            "Penn94",
            "Amherst41",
            "Cornell5",
            "Johns Hopkins55",
            "Reed98",
        ] else dataset_name
    )
    ssl._create_default_https_context = ssl._create_unverified_context

    if source == "pyg" and dataset_name in PYG_DATASETS:
        graph, label, n_clusters = load_pyg_data(
            dataset_name=dataset_name,
            directory=directory,
            verbosity=verbosity,
        )
    elif source == "dgl" and dataset_name in DGL_DATASETS:
        graph, label, n_clusters = load_dgl_data(
            dataset_name=dataset_name,
            directory=directory,
            verbosity=verbosity,
        )
    elif source == "ogb" and dataset_name in OGB_DATASETS:
        graph, label, n_clusters = load_ogb_data(
            dataset_name=dataset_name,
            directory=directory,
            verbosity=verbosity,
        )
    elif source == "sdcn" and dataset_name in SDCN_DATASETS:
        graph, label, n_clusters = load_sdcn_data(
            dataset_name=dataset_name,
            directory=directory,
            verbosity=verbosity,
        )
    elif source == "cola" and dataset_name in COLA_DATASETS:
        graph, label, n_clusters = load_cola_data(
            dataset_name=dataset_name,
            directory=directory,
            verbosity=verbosity,
        )
    elif source == "linkx" and dataset_name in LINKX_DATASETS:
        graph, label, n_clusters = load_linkx_data(
            dataset_name=dataset_name,
            directory=directory,
            verbosity=verbosity,
        )
    elif source == "critical" and dataset_name in CRITICAL_DATASETS:
        graph, label, n_clusters = load_critical_dataset(
            dataset_name=dataset_name,
            directory=directory,
            verbosity=verbosity,
        )
    else:
        raise NotImplementedError(
            f"The dataset '{dataset_name}' is not supported or the source '{source}' is incorrect. "
            f"Please check the sources or datasets on:\n"
            f"https://galogm.github.io/graph_datasets_docs/rst/table.html"
        )

    # remove self loop and turn graphs into undirected ones
    if rm_self_loop:
        graph = dgl.remove_self_loop(graph)
    if to_simple:
        graph = dgl.to_bidirected(graph, copy_ndata=True)

    # make label from 0
    uni = label.unique()
    old2new = dict(zip(uni.numpy().tolist(), list(range(len(uni)))))
    newlabel = torch.tensor(list(map(lambda x: old2new[x.item()], label)))
    graph.ndata["label"] = newlabel

    if verbosity:
        print_dataset_info(
            dataset_name=f"{source} undirected {dataset_name}\nwithout self-loops",
            n_nodes=graph.num_nodes(),
            n_edges=graph.num_edges(),
            n_feats=graph.ndata["feat"].shape[1],
            n_clusters=n_clusters,
        )

    return graph, newlabel, n_clusters


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Load Graph datasets",
        description="Load Graph datasets",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default="cora",
        help="Dataset name",
    )
    parser.add_argument(
        "-p",
        "--directory_path",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Data directory path",
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default="pyg",
        help="Dataset source",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        help="Output debug information",
        default=0,
    )
    args = parser.parse_args()

    load_data(
        dataset_name=args.dataset_name,
        directory=args.directory_path,
        source=args.source,
        verbosity=args.verbosity,
    )
