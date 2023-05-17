"""Graph information statistics.
"""
import numpy as np
import scipy.sparse as sp
import torch


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
    return num_intra_class_edges / num_edges, num_intra_class_edges


def statistics(adj: sp.spmatrix, labels: torch.Tensor) -> dict:
    dic = {
        "eh_1h": None,
        "nh_1h": None,
        "ie_1h": None,
        "bn_1h": None,
        "eh_2h": None,
        "nh_2h": None,
        "ie_2h": None,
        "bn_2h": None,
    }
    # 1 hop
    dic["eh_1h"], dic["ie_1h"] = edge_homo(adj=adj, labels=labels)
    dic["nh_1h"], dic["bn_1h"] = node_homo(adj=adj, labels=labels)

    # 2 hop
    dic["eh_2h"], dic["ie_2h"] = edge_homo(adj=adj.dot(adj), labels=labels)
    dic["nh_2h"], dic["bn_2h"] = node_homo(adj=adj.dot(adj), labels=labels)

    return dic
