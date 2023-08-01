"""Datasets from `OGB <https://github.com/snap-stanford/ogb>`_.
"""
from typing import Tuple

import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset

from ..data_info import DEFAULT_DATA_DIR
from ..utils import print_dataset_info


def load_ogb_data(
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
    verbosity: int = 0,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    """Load OGB graphs.

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
    dataset = DglNodePropPredDataset(
        name="ogbn-" + dataset_name,
        root=directory,
    )
    graph, label = dataset[0]
    label = label.flatten()
    graph.ndata["label"] = label

    if verbosity and verbosity > 1:
        print_dataset_info(
            dataset_name=f"OGB {dataset_name}",
            n_nodes=graph.num_nodes(),
            n_edges=graph.num_edges(),
            n_feats=graph.ndata["feat"].shape[1],
            n_clusters=dataset.num_classes,
        )

    return graph, label, dataset.num_classes
