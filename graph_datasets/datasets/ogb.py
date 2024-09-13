"""Datasets from `OGB <https://github.com/snap-stanford/ogb>`_.
"""

import traceback
from typing import Tuple

import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from torch_sparse import SparseTensor

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
    if dataset_name not in ["proteins"]:
        graph.ndata["label"] = label.flatten()
    else:
        graph.ndata["label"] = label
        # Adapted from https://github.com/qitianwu/SGFormer
        edge_index = torch.stack(graph.edges())
        edge_feat = graph.edata["feat"]
        edge_index_ = to_sparse_tensor(edge_index, edge_feat, graph.num_nodes())
        graph.ndata["feat"] = edge_index_.mean(dim=1)

    if verbosity and verbosity > 1:
        print_dataset_info(
            dataset_name=f"OGB {dataset_name}",
            n_nodes=graph.num_nodes(),
            n_edges=graph.num_edges(),
            n_feats=graph.ndata["feat"].shape[1],
            n_clusters=dataset.num_classes,
        )

    try:
        splits = dataset.get_idx_split()
    except Exception as _:
        traceback.print_exc()
        splits = None

    if splits is not None:
        graph.ndata["train_mask"] = (
            torch.zeros(graph.num_nodes()).scatter_(
                0,
                splits["train"],
                1,
            ).bool()
        )
        graph.ndata["val_mask"] = (
            torch.zeros(graph.num_nodes()).scatter_(
                0,
                splits["valid"],
                1,
            ).bool()
        )
        graph.ndata["test_mask"] = (
            torch.zeros(graph.num_nodes()).scatter_(
                0,
                splits["test"],
                1,
            ).bool()
        )

    return graph, label, dataset.num_classes


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """converts the edge_index into SparseTensor"""
    num_edges = edge_index.size(1)
    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]
    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N), is_sorted=True)
    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()
    return adj_t
