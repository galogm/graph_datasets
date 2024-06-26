"""Load Graph Datasets
"""

# pylint:disable=protected-access
import ssl
from typing import Tuple
from typing import Union

import dgl
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_dgl

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
    return_type: str = "dgl",
    raw_normalize: bool = True,
    rm_self_loop: bool = True,
    add_self_loop: bool = False,
    to_simple: bool = True,
) -> Union[Tuple[dgl.DGLGraph, torch.Tensor, int], Data]:
    """Load graphs.

    Args:
        dataset_name (str): Dataset name.
        directory (str, optional): Raw dir for loading or saving. \
            Defaults to DEFAULT_DATA_DIR=os.path.abspath("./data").
        verbosity (int, optional): Output debug information. \
            The greater, the more detailed. Defaults to 0.
        source (str, optional): Source for data loading. Defaults to "pyg".
        return_type (str, optional): Return type of the graphs within ["dgl", "pyg"]. \
            Defaults to "dgl".
        raw_normalize (str, optional): Row normalize the feature matrix. Defaults to True.
        rm_self_loop (str, optional): Remove self loops. Defaults to True.
        add_self_loop (str, optional): Add self loops no matter what rm_self_loop is. \
            Defaults to True.
        to_simple (str, optional): Convert to a simple graph with no duplicate undirected edges.

    Raises:
        NotImplementedError: Dataset unknown.

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, int]: [graph, label, n_clusters] or \
        torch_geometric.data.Data

    Example:
        .. code-block:: python

            from graph_datasets import load_data
            # dgl graph
            graph, label, n_clusters = load_data(
                dataset_name='cora',
                directory="./data",
                return_type="dgl",
                source='pyg',
                verbosity=3,
                rm_self_loop=True,
                to_simple=True,
            )
            # pyG data
            data = load_data(
                dataset_name='cora',
                directory="./data",
                return_type="pyg",
                source='pyg',
                verbosity=3,
                rm_self_loop=True,
                to_simple=True,
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

    if raw_normalize:
        graph.ndata["feat"] = F.normalize(graph.ndata["feat"], dim=1)
    if rm_self_loop:
        graph = graph.remove_self_loop()
    if add_self_loop:
        graph = graph.remove_self_loop().add_self_loop()
    if to_simple:
        graph = dgl.to_bidirected(graph, copy_ndata=True)

    # make label from 0
    uni = label.unique()
    old2new = dict(zip(uni.numpy().tolist(), list(range(len(uni)))))
    new_label = torch.tensor(list(map(lambda x: old2new[x.item()], label)))
    graph.ndata["label"] = new_label

    name = f"{dataset_name}_{source}"
    graph.name = name

    if verbosity:
        print_dataset_info(
            dataset_name=
            f"{source.upper()} undirected {dataset_name}\n  add_self_loop={add_self_loop} rm_self_loop={rm_self_loop}",
            n_nodes=graph.num_nodes(),
            n_edges=graph.num_edges(),
            n_feats=graph.ndata["feat"].shape[1],
            n_clusters=n_clusters,
        )

    if return_type == "dgl":
        return graph, new_label, n_clusters

    data = from_dgl(graph)
    data.name = name
    data.num_classes = n_clusters
    data.x = data.feat
    data.y = data.label
    data.num_nodes = graph.num_nodes()
    data.num_edges = graph.num_edges()
    data.edge_index = torch.stack(graph.edges(), dim=0)

    return data


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
