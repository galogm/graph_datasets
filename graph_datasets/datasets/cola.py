"""Datasets from the paper `CoLA <https://github.com/GRAND-Lab/CoLA>`_.
"""
# pylint:disable=duplicate-code
import os
from typing import Tuple

import dgl
import numpy as np
import scipy.io as sio
import torch
import wget

from ..data_info import COLA_URL
from ..data_info import DEFAULT_DATA_DIR
from ..utils import bar_progress
from ..utils import print_dataset_info


def load_cola_data(
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
    verbosity: int = 0,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    """Load CoLA graphs.

    Args:
        dataset_name (str):  Dataset name.
        directory (str, optional): Raw dir for loading or saving. \
            Defaults to DEFAULT_DATA_DIR=os.path.abspath("./data").
        verbosity (int, optional): Output debug information. \
            The greater, the more detailed. Defaults to 0.

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, int]: [graph, label, n_clusters]
    """
    dataset = {
        "blogcatalog": "BlogCatalog",
        "flickr": "Flickr",
    }[dataset_name]

    data_file = os.path.join(directory, f"{dataset_name}.mat")

    if not os.path.exists(data_file):
        wget.download(
            f"{COLA_URL}/raw_dataset/{dataset}/{dataset}.mat?raw=true",
            out=data_file,
            bar=bar_progress,
        )

    data_mat = sio.loadmat(data_file)
    adj = data_mat["Network"]
    feat = data_mat["Attributes"]
    labels = data_mat["Label"]
    labels = labels.flatten()
    graph = dgl.from_scipy(adj)
    graph.ndata["feat"] = torch.from_numpy(feat.toarray()).to(torch.float32)
    graph.ndata["label"] = torch.from_numpy(labels).to(torch.int64)
    num_classes = len(np.unique(labels))

    if verbosity and verbosity > 1:
        print_dataset_info(
            dataset_name=f"CoLA {dataset_name}",
            n_nodes=graph.num_nodes(),
            n_edges=graph.num_edges(),
            n_feats=graph.ndata["feat"].shape[1],
            n_clusters=num_classes,
        )

    return graph, graph.ndata["label"], num_classes
