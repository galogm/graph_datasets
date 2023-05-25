"""Graph information statistics.
"""
import gc
import math

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from dgl import function as fn


def node_homo(adj: sp.spmatrix, labels: torch.Tensor) -> float:
    """Calculate node homophily.

    Args:
        adj (sp.spmatrix): adjacent matrix.
        labels (torch.Tensor): labels.

    Returns:
        float: node homophily.
    """
    adj_coo = adj.tocoo()
    adj_coo.data = (labels[adj_coo.col] == labels[adj_coo.row]).cpu().numpy().astype(int)
    n_h = np.asarray(adj_coo.sum(1)).flatten() / np.asarray(adj.sum(1)).flatten()
    del adj_coo
    del adj
    return n_h.mean(), (n_h != 1).astype(int).sum()


def edge_homo(adj: sp.spmatrix, labels: torch.Tensor) -> float:
    """Calculate edge homophily.

    Args:
        adj (sp.spmatrix): adjacent matrix.
        labels (torch.Tensor): labels.

    Returns:
        float: edge homophily.
    """
    adj_coo = adj.tocoo()
    num_intra_class_edges = ((labels[adj_coo.col] == labels[adj_coo.row]).cpu().numpy() *
                             adj.data).sum()
    num_edges = adj.sum()
    del adj_coo
    del adj
    return num_intra_class_edges / num_edges, num_intra_class_edges


def get_long_edges(graph):
    """Internal function for getting the edges of a graph as long tensors."""
    src, dst = graph.edges()
    return src.long(), dst.long()


def get_same_class_deg(graph, labels):
    with graph.local_scope():
        # Handle the case where graph is of dtype int32.
        src, dst = get_long_edges(graph)
        # Compute y_v = y_u for all edges.
        graph.edata["same_class"] = (labels[src] == labels[dst]).float()
        graph.update_all(fn.copy_e("same_class", "m"), fn.mean("m", "same_class_deg"))
        return graph.ndata["same_class_deg"]


# pylint:disable=too-many-statements
def statistics(
    graph: dgl.DGLGraph,
    labels: torch.Tensor,
    dataset_name: str = "",
    h_1=True,
    h_2=True,
) -> dict:
    """Calculate homophily metrics of graphs.

    Args:
        graph (dgl.DGLGraph): Graph
        labels (torch.Tensor): Labels
        dataset_name (str, optional): Dataset name. Defaults to ''.
        h_1 (bool, optional): 1-hop graph metrics. Defaults to True.
        h_2 (bool, optional): 2-hop graph metrics. Defaults to True.

    Raises:
        MemoryError: OOM.

    Returns:
        dict: Dict of metric results.
    """
    dic = {}
    num_edges = graph.num_edges()

    if h_1:
        dic["eh_1h"] = dgl.edge_homophily(graph, labels)
        dic["ie_1h"] = math.ceil(dic["eh_1h"] * num_edges)

        dic["nh_1h"] = dgl.node_homophily(graph, labels)
        dic["bn_1h"] = (get_same_class_deg(graph, labels) < 1).sum().item()

        dic["lh_1h"] = dgl.linkx_homophily(graph, labels)
    else:
        dic["eh_1h"] = np.nan
        dic["ie_1h"] = np.nan

        dic["nh_1h"] = np.nan
        dic["bn_1h"] = np.nan

        dic["lh_1h"] = np.nan

    try:
        if h_2:
            # pylint:disable=import-outside-toplevel
            import os

            file_path = f"./data/g.2h.{dataset_name}"
            if os.path.exists(file_path):
                graph_2h, _ = dgl.load_graphs(file_path)
                graph_2h = graph_2h[0]
            else:
                graph_2h = dgl.khop_graph(graph, k=2)
                dgl.save_graphs(file_path, [graph_2h])
            del graph
            gc.collect()

            dic["n_edges_2h"] = graph_2h.num_edges()
            dic["eh_2h"] = dgl.edge_homophily(graph_2h, labels)
            dic["ie_2h"] = math.ceil(dic["eh_2h"] * graph_2h.num_edges())
            dic["nh_2h"] = dgl.node_homophily(graph_2h, labels)
            dic["bn_2h"] = (get_same_class_deg(graph_2h, labels) < 1).sum().item()
            dic["lh_2h"] = dgl.linkx_homophily(graph_2h, labels)

            graph_2h = dgl.to_simple(dgl.remove_self_loop(graph_2h))

            dic["n_edges_2h_uns"] = graph_2h.num_edges()
            dic["eh_2h_uns"] = dgl.edge_homophily(graph_2h, labels)
            dic["ie_2h_uns"] = math.ceil(dic["eh_2h"] * graph_2h.num_edges())
            dic["nh_2h_uns"] = dgl.node_homophily(graph_2h, labels)
            dic["bn_2h_uns"] = (get_same_class_deg(graph_2h, labels) < 1).sum().item()
            dic["lh_2h_uns"] = dgl.linkx_homophily(graph_2h, labels)
        else:
            raise MemoryError("no h2")

    except MemoryError:
        # only works on variable initialization
        print("2-hop graph OOM.")
        dic["n_edges_2h"] = np.nan
        dic["eh_2h"] = np.nan
        dic["ie_2h"] = np.nan
        dic["nh_2h"] = np.nan
        dic["bn_2h"] = np.nan
        dic["lh_2h"] = np.nan
        dic["n_edges_2h_uns"] = np.nan
        dic["eh_2h_uns"] = np.nan
        dic["ie_2h_uns"] = np.nan
        dic["nh_2h_uns"] = np.nan
        dic["bn_2h_uns"] = np.nan
        dic["lh_2h_uns"] = np.nan

    return dic
