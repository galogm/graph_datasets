"""Datasets from `PyG <https://github.com/pyg-team/pytorch_geometric>`_.
"""
import os
from typing import Tuple

import dgl
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Actor
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CoraFull
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import WikiCS
from torch_geometric.datasets import WikipediaNetwork

from ..data_info import DEFAULT_DATA_DIR
from ..utils import print_dataset_info


def load_pyg_data(
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
    verbosity: int = 0,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    """Load pyG graphs.

    Args:
        dataset_name (str): Dataset name.
        directory (str, optional): Raw dir for loading or saving. \
            Defaults to DEFAULT_DATA_DIR=os.path.abspath("./data").
        verbosity (int, optional): Output debug information. \
            The greater, the more detailed. Defaults to 0.

    Raises:
        NotImplementedError: Dataset unknown.

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, int]: [graph, label, n_clusters]
    """
    path = os.path.join(os.path.abspath(directory), dataset_name)

    if dataset_name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root=path, name=dataset_name)
    elif dataset_name in ["reddit"]:
        dataset = Reddit(root=path)
    elif dataset_name in ["corafull"]:
        dataset = CoraFull(root=path)
    elif dataset_name in ["chameleon"]:
        dataset = WikipediaNetwork(root=path, name=dataset_name)
    elif dataset_name in ["squirrel"]:
        dataset = WikipediaNetwork(root=path, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ["actor"]:
        dataset = Actor(root=path)
    elif dataset_name in ["cornell", "texas", "wisconsin"]:
        dataset = WebKB(root=path, name=dataset_name)
    elif dataset_name in ["computers", "photo"]:
        dataset = Amazon(root=path, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ["cs", "physics"]:
        dataset = Coauthor(root=path, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ["wikics"]:
        dataset = WikiCS(root=path, is_undirected=False)
    else:
        raise NotImplementedError(
            f"The Dataset '{dataset_name}' is not supported."
            f"Please check the sources or datasets on:\n"
            f"https://galogm.github.io/graph_datasets_docs/rst/table.html"
        )

    data = dataset[0]

    edges = data.edge_index

    features = data.x
    labels = data.y
    n_classes = len(torch.unique(labels))

    graph = dgl.graph((edges[0, :].numpy(), edges[1, :].numpy()))
    graph.ndata["feat"] = features
    graph.ndata["label"] = labels

    if verbosity and verbosity > 1:
        print_dataset_info(
            dataset_name=f"pyG {dataset_name}",
            n_nodes=graph.num_nodes(),
            n_edges=graph.num_edges(),
            n_feats=graph.ndata["feat"].shape[1],
            n_clusters=n_classes,
        )

    return graph, labels, n_classes
