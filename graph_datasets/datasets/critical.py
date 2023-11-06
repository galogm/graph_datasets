"""Datasets from the paper \
    `A Critical Look at the Evaluation of GNNs Under Heterophily: \
        Are We Really Making Progress?\
              <https://openreview.net/pdf?id=tJbbQfw-5wv>`_.
"""
import os
from typing import Tuple

import dgl
import numpy as np
import torch
import wget

from ..data_info import CRITICAL_URL
from ..data_info import DEFAULT_DATA_DIR
from ..utils import bar_progress
from ..utils import download_tip
from ..utils import print_dataset_info


def load_critical_dataset(
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
    verbosity: int = 0,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    """Load graphs from *A Critical Look at the Evaluation of GNNs Under Heterophily:\
          Are We Really Making Progress?*

    Args:
        dataset_name (str):  Dataset name.
        directory (str, optional): Raw dir for loading or saving. \
            Defaults to DEFAULT_DATA_DIR=os.path.abspath("./data").
        verbosity (int, optional): Output debug information. \
            The greater, the more detailed. Defaults to 0.

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, int]: [graph, label, n_clusters]
    """
    if dataset_name in ["squirrel", "chameleon"]:
        dataset_name = f"{dataset_name}_filtered_directed"
    dataset_name = dataset_name.replace("-", "_")
    data_file = os.path.join(directory, f"{dataset_name}.npz")

    if not os.path.exists(data_file):
        url = f"{CRITICAL_URL}/{dataset_name}.npz?raw=true"
        info = {
            "File": os.path.basename(data_file),
            "Download URL": url,
            "Save Path": data_file,
        }
        download_tip(info)
        wget.download(url, out=data_file, bar=bar_progress)

    data = np.load(data_file)
    feat = torch.tensor(data["node_features"])
    labels = torch.tensor(data["node_labels"])
    edges = torch.tensor(data["edges"])

    graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(feat), idtype=torch.int)
    graph.ndata["feat"] = feat
    graph.ndata["label"] = labels
    num_classes = len(torch.unique(labels))

    if verbosity and verbosity > 1:
        print_dataset_info(
            dataset_name=f"Critical {dataset_name}",
            n_nodes=graph.num_nodes(),
            n_edges=graph.num_edges(),
            n_feats=graph.ndata["feat"].shape[1],
            n_clusters=num_classes,
        )

    return graph, graph.ndata["label"], num_classes
