"""Datasets from the paper `SDCN <https://github.com/bdy9527/SDCN>`_.
"""
import os
from typing import Tuple

import dgl
import numpy as np
import torch
import wget

from ..data_info import DEFAULT_DATA_DIR
from ..data_info import SDCN_URL
from ..utils import bar_progress
from ..utils import print_dataset_info


def load_sdcn_data(
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
    verbosity: int = 0,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    """Load SDCN graphs.

    Args:
        dataset_name (str):  Dataset name.
        directory (str, optional): Raw dir for loading or saving. \
            Defaults to DEFAULT_DATA_DIR=os.path.abspath("./data").
        verbosity (int, optional): Output debug information. \
            The greater, the more detailed. Defaults to 0.

    NOTE:
        The last node of DBLP is an isolated node.

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, int]: [graph, label, n_clusters]
    """
    if not os.path.exists(os.path.join(directory, dataset_name)):
        os.mkdir(os.path.join(directory, dataset_name))

    adj_data_file = os.path.join(directory, dataset_name, "adj.txt")
    if not os.path.exists(adj_data_file):
        url = f"{SDCN_URL}/graph/{dataset_name}_graph.txt?raw=true"
        wget.download(url, out=adj_data_file, bar=bar_progress)

    feat_data_file = os.path.join(directory, dataset_name, "feat.txt")
    if not os.path.exists(feat_data_file):
        url = f"{SDCN_URL}/data/{dataset_name}.txt?raw=true"
        wget.download(url, out=feat_data_file, bar=bar_progress)

    label_data_file = os.path.join(directory, dataset_name, "label.txt")
    if not os.path.exists(label_data_file):
        url = f"{SDCN_URL}/data/{dataset_name}_label.txt?raw=true"
        wget.download(url, out=label_data_file, bar=bar_progress)

    feat = np.loadtxt(feat_data_file, dtype=float)
    labels = np.loadtxt(label_data_file, dtype=int)
    edges_unordered = np.genfromtxt(adj_data_file, dtype=np.int32)
    # NOTE: The last node of DBLP is an isolated node, which has no edges in adj.
    graph = dgl.graph((edges_unordered[:, 0], edges_unordered[:, 1]), num_nodes=feat.shape[0])
    graph.ndata["feat"] = torch.from_numpy(feat).to(torch.float32)
    graph.ndata["label"] = torch.from_numpy(labels).to(torch.int64)
    num_classes = len(np.unique(labels))

    if verbosity and verbosity > 1:
        print_dataset_info(
            dataset_name=f"SDCN {dataset_name}",
            n_nodes=graph.num_nodes(),
            n_edges=graph.num_edges(),
            n_feats=graph.ndata["feat"].shape[1],
            n_clusters=num_classes,
        )

    return graph, graph.ndata["label"], num_classes
