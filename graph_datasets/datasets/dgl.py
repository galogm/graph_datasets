"""Datasets from `DGL <https://github.com/dmlc/dgl>`_.
"""
from typing import Tuple

import dgl
import torch

from ..data_info import DEFAULT_DATA_DIR
from ..utils import print_dataset_info


def load_dgl_data(
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
    verbosity: int = 0,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    """Load DGL graphs.

    Args:
        dataset_name (str): Dataset name.
        directory (str, optional): Raw dir for loading or saving. \
            Defaults to DEFAULT_DATA_DIR=os.path.abspath("./data").
        verbosity (int, optional): Output debug information. \
            The greater, the more detailed. Defaults to 0.

    Raises:
        NotImplementedError: Dataset unknown.

    NOTE:
        Chameleon, Squirrel, Actor, Cornell, Texas and Wisconsin are from Geom-GCN,\
            which may be slightly different from heterophilous settings.

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, int]: [graph, label, n_clusters]
    """
    dataset_map = {
        "cora": "CoraGraphDataset",
        "citeseer": "CiteseerGraphDataset",
        "pubmed": "PubmedGraphDataset",
        "corafull": "CoraFullDataset",
        "reddit": "RedditDataset",
        "chameleon": "ChameleonDataset",
        "squirrel": "SquirrelDataset",
        "actor": "ActorDataset",
        "cornell": "CornellDataset",
        "texas": "TexasDataset",
        "wisconsin": "WisconsinDataset",
    }

    if dataset_name in ["cora", "citeseer", "pubmed"]:
        dataset = getattr(dgl.data, dataset_map[dataset_name])(
            raw_dir=directory,
            force_reload=False,
            verbose=False,
            transform=None,
            reverse_edge=False,
            reorder=False,
        )
    elif dataset_name in [
            "chameleon",
            "squirrel",
            "actor",
            "cornell",
            "texas",
            "wisconsin",
            "corafull",
            "reddit",
    ]:
        dataset = getattr(dgl.data, dataset_map[dataset_name])(
            raw_dir=directory,
            force_reload=False,
            verbose=False,
            transform=None,
        )
    else:
        raise NotImplementedError(f"load_dgl_data does not support {dataset_name}.")

    graph = dataset[0]

    if verbosity and verbosity > 1:
        print_dataset_info(
            dataset_name=f"dgl {dataset_name}",
            n_nodes=graph.num_nodes(),
            n_edges=graph.num_edges(),
            n_feats=graph.ndata["feat"].shape[1],
            n_clusters=dataset.num_classes,
        )

    return graph, graph.ndata["label"], dataset.num_classes
