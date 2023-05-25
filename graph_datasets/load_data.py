"""Load Graph Datasets

Example:
    >>> python -m src.load_data -d cora -s pyg -vv
"""
# pylint:disable=invalid-name,protected-access
import os
import ssl
from typing import List
from typing import Tuple

import dgl
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
import wget
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Actor
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CoraFull
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import WikiCS
from torch_geometric.datasets import WikipediaNetwork

from .data_info import COLA_DATASETS
from .data_info import COLA_URL
from .data_info import DEFAULT_DATA_DIR
from .data_info import DGL_DATASETS
from .data_info import LINKX_DATASETS
from .data_info import LINKX_DRIVE_ID
from .data_info import LINKX_URL
from .data_info import OGB_DATASETS
from .data_info import PYG_DATASETS
from .data_info import SDCN_DATASETS
from .data_info import SDCN_URL
from .utils import bar_progress
from .utils import download_from_google_drive
from .utils import tab_printer


def load_data(
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
    verbosity: int = 0,
    source: str = "pyg",
    rm_self_loop: bool = True,
    to_simple: bool = True,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    """Load graphs.

    Args:
        dataset_name (str): Dataset name.
        directory (str, optional): Raw dir for loading or saving. \
            Defaults to DEFAULT_DATA_DIR=os.path.abspath("./data").
        verbosity (int, optional): Output debug information. \
            The greater, the more detailed. Defaults to 0.
        source (str, optional): Source for data loading. Defaults to "pyg".
        rm_self_loop (str, optional): Remove self loops. Defaults to True.
        to_simple (str, optional): Convert to a simple graph with no duplicate undirected edges.

    Raises:
        NotImplementedError: Dataset unknown.

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, int]: [graph, label, n_clusters]

    Examples:
        >>> graph, label, n_clusters = load_data('cora')
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
    else:
        raise NotImplementedError(
            f"{dataset_name} is not supported or source {source} is incorrect."
        )

    # remove self loop and turn graphs into undirected ones
    if rm_self_loop:
        graph = dgl.remove_self_loop(graph)
    if to_simple:
        graph = dgl.to_bidirected(graph, copy_ndata=True)

    # make label from 0
    uni = label.unique()
    old2new = dict(zip(uni.numpy().tolist(), list(range(len(uni)))))
    newlabel = torch.tensor(list(map(lambda x: old2new[x.item()], label)))
    graph.ndata["label"] = newlabel

    if verbosity:
        print_dataset_info(
            dataset_name=f"undirected {dataset_name}\nwithout self-loops",
            n_nodes=graph.num_nodes(),
            n_edges=graph.num_edges(),
            n_feats=graph.ndata["feat"].shape[1],
            n_clusters=n_clusters,
        )

    return graph, newlabel, n_clusters


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
        raise NotImplementedError(f"{dataset_name} is not supported.")

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

    NOTE: The last node of DBLP is an isolated node.

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
            dataset_name=f"SDCN {dataset_name}",
            n_nodes=graph.num_nodes(),
            n_edges=graph.num_edges(),
            n_feats=graph.ndata["feat"].shape[1],
            n_clusters=num_classes,
        )

    return graph, graph.ndata["label"], num_classes


def even_quantile_labels(vals, n_classes, verbosity: int = 0):
    """partitions vals into n_classes by a quantile based split,
    where the first class is less than the 1/n_classes quantile,
    second class is less than the 2/n_classes quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int32)
    interval_lst = []
    lower = -np.inf
    for k in range(n_classes - 1):
        upper = np.nanquantile(vals, (k + 1) / n_classes)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = n_classes - 1
    interval_lst.append((lower, np.inf))
    if verbosity and verbosity > 1:
        print("Generated Class Label Intervals:")
        for class_idx, interval in enumerate(interval_lst):
            print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
    return label


def load_linkx_data(
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
    verbosity: int = 0,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    """Load LINKX graphs.

    Args:
        dataset_name (str):  Dataset name.
        directory (str, optional): Raw dir for loading or saving. \
            Defaults to DEFAULT_DATA_DIR=os.path.abspath("./data").
        verbosity (int, optional): Output debug information. \
            The greater, the more detailed. Defaults to 0.

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, int]: [graph, label, n_clusters]
    """
    data_file = os.path.abspath(os.path.join(directory, f"{dataset_name.replace('-','_')}.mat"))

    if dataset_name in ["pokec", "snap-patents", "yelp-chi"]:
        if not os.path.exists(data_file):
            download_from_google_drive(
                gid=LINKX_DRIVE_ID[dataset_name],
                output=data_file,
                file_name=dataset_name,
            )

        data_mat = sio.loadmat(data_file)

        if dataset_name == "yelp-chi":
            g = dgl.from_scipy(data_mat["homo"])
            g.ndata["feat"] = torch.tensor(data_mat["features"].todense(), dtype=torch.float)
        else:
            g = dgl.graph((data_mat["edge_index"][0], data_mat["edge_index"][1]))
            g.ndata["feat"] = torch.tensor(
                data_mat["node_feat"].todense()
                if sp.issparse(data_mat["node_feat"]) else data_mat["node_feat"],
                dtype=torch.float,
            )

        if dataset_name == "snap-patents":
            years = data_mat["years"].flatten()
            label = even_quantile_labels(years, 5, verbosity=verbosity)
            g.ndata["label"] = torch.tensor(label, dtype=torch.long)
        else:
            g.ndata["label"] = torch.tensor(data_mat["label"].flatten(), dtype=torch.long)

        num_classes = g.ndata["label"].unique().shape[0]
    elif dataset_name in [
            "cornell",
            "chameleon",
            "film",
            "squirrel",
            "texas",
            "wisconsin",
            "genius",
            "deezer-europe",
    ]:
        g, _, num_classes = load_linkx_github(
            dataset_name=dataset_name,
            directory=directory,
        )
    elif dataset_name in [
            "Penn94",
            "Amherst41",
            "Cornell5",
            "Johns Hopkins55",
            "Reed98",
    ]:
        g, _, num_classes = load_fb100_data(
            dataset_name=dataset_name,
            directory=directory,
        )
    elif dataset_name in ["wiki"]:
        g, _, num_classes = load_wiki_data(directory=directory)
    elif dataset_name in ["twitch-gamers"]:
        g, _, num_classes = load_twitch_gamers_data()
    elif dataset_name in ["arxiv-year"]:
        g, _, num_classes = load_arxiv_year_data(
            directory=directory,
            verbosity=verbosity,
        )

    if verbosity and verbosity > 1:
        print_dataset_info(
            dataset_name=f"Original {dataset_name}",
            n_nodes=g.num_nodes(),
            n_edges=g.num_edges(),
            n_feats=g.ndata["feat"].shape[1],
            n_clusters=num_classes,
        )

    return g, g.ndata["label"], num_classes


def load_linkx_github(
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    """Load LINKX graphs.

    Args:
        dataset_name (str):  Dataset name.
        directory (str, optional): Raw dir for loading or saving. \
            Defaults to DEFAULT_DATA_DIR=os.path.abspath("./data").

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, int]: [graph, label, n_clusters]
    """
    data_file = os.path.join(directory, f"{dataset_name}.mat")

    if not os.path.exists(data_file):
        url = f"{LINKX_URL}/{dataset_name}.mat?raw=true"
        info = {
            "File": os.path.basename(data_file),
            "Download URL": url,
            "Save Path": data_file,
        }
        tab_printer(info)
        wget.download(url, out=data_file, bar=bar_progress)

    data_mat = sio.loadmat(data_file)
    if dataset_name == "deezer-europe":
        graph = dgl.from_scipy(data_mat["A"])
        graph.ndata["feat"] = torch.from_numpy(data_mat["features"].toarray()).to(torch.float32)
    else:
        graph = dgl.graph((data_mat["edge_index"][0], data_mat["edge_index"][1]))
        graph.ndata["feat"] = torch.from_numpy(data_mat["node_feat"]).to(torch.float32)

    label = data_mat["label"].flatten()
    graph.ndata["label"] = torch.from_numpy(label).to(torch.int64)
    num_classes = len(np.unique(label))

    return graph, graph.ndata["label"], num_classes


def load_wiki_data(directory: str = DEFAULT_DATA_DIR) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    data_file = os.path.join(directory, "wiki")
    wiki_data = os.path.join(data_file, "wiki.pt")

    if os.path.exists(wiki_data):
        g = torch.load(wiki_data)
        num_classes = len(torch.unique(g.ndata["label"]))
        return g, g.ndata["label"], num_classes

    if not os.path.exists(data_file):
        os.mkdir(data_file)

    feat_path = os.path.join(data_file, "wiki_features.pt")
    if not os.path.exists(feat_path):
        download_from_google_drive(
            gid=LINKX_DRIVE_ID["wiki_features"],
            output=feat_path,
            quiet=False,
        )

    edge_path = os.path.join(data_file, "wiki_edges.pt")
    if not os.path.exists(edge_path):
        download_from_google_drive(
            gid=LINKX_DRIVE_ID["wiki_edges"],
            output=edge_path,
            quiet=False,
        )

    view_path = os.path.join(data_file, "wiki_views.pt")
    if not os.path.exists(view_path):
        download_from_google_drive(
            gid=LINKX_DRIVE_ID["wiki_views"],
            output=view_path,
            quiet=False,
        )

    features = torch.load(feat_path)
    edges = torch.load(edge_path).T
    label = torch.load(view_path)

    # NOTE: 1925342 nodes in edges while 1924551 nodes in features
    g = dgl.graph((edges[0], edges[1]), num_nodes=features.shape[0])
    g.ndata["feat"] = features
    g.ndata["label"] = label
    num_classes = len(np.unique(label))

    torch.save(g, wiki_data)

    return g, g.ndata["label"], num_classes


def load_twitch_gamers_data(
    task="mature",
    directory: str = DEFAULT_DATA_DIR,
    normalize: bool = True,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    data_file = os.path.join(directory, "twitch_gamers")
    twitch_gamers_data = os.path.join(data_file, "twitch_gamers.pt")

    if os.path.exists(twitch_gamers_data):
        g = torch.load(twitch_gamers_data)
        num_classes = len(torch.unique(g.ndata["label"]))
        return g, g.ndata["label"], num_classes

    if not os.path.exists(data_file):
        os.mkdir(data_file)

    feat_path = os.path.join(data_file, "twitch-gamer_feat.csv")
    if not os.path.exists(feat_path):
        download_from_google_drive(
            gid=LINKX_DRIVE_ID["twitch-gamer_feat"],
            output=feat_path,
            quiet=False,
        )

    edge_path = os.path.join(data_file, "twitch-gamer_edges.csv")
    if not os.path.exists(edge_path):
        download_from_google_drive(
            gid=LINKX_DRIVE_ID["twitch-gamer_edges"],
            output=edge_path,
            quiet=False,
        )

    edge_index = torch.tensor(pd.read_csv(edge_path).to_numpy()).t().type(torch.LongTensor)

    task = "dead_account"
    nodes = pd.read_csv(feat_path).drop("numeric_id", axis=1)
    nodes["created_at"] = nodes.created_at.replace("-", "", regex=True).astype(int)
    nodes["updated_at"] = nodes.updated_at.replace("-", "", regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes["language"].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes["language"]]
    nodes["language"] = lang_encoding

    if task is not None:
        label = torch.tensor(nodes[task].to_numpy())
        features = torch.tensor(nodes.drop(task, axis=1).to_numpy(), dtype=torch.float)

    if normalize:
        features = features - features.mean(dim=0, keepdim=True)
        features = features / features.std(dim=0, keepdim=True)

    g = dgl.graph((edge_index[0], edge_index[1]))
    g.ndata["feat"] = features
    g.ndata["label"] = label
    num_classes = label.unique().shape[0]

    torch.save(g, twitch_gamers_data)

    return g, g.ndata["label"], num_classes


def load_fb100_data(
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    data_file = os.path.join(directory, "facebook100")
    data = os.path.join(data_file, f"{dataset_name}.mat")

    if not os.path.exists(data_file):
        os.mkdir(data_file)

    if not os.path.exists(data):
        url = f"{LINKX_URL}/facebook100/{dataset_name}.mat?raw=true"
        tab_printer({
            "File": os.path.basename(data),
            "Download URL": url,
            "Save Path": data,
        })
        wget.download(url, out=data, bar=bar_progress)

    mat = sio.loadmat(data)
    A = mat["A"]
    metadata = mat["local_info"].astype(np.int32)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    # gender label, -1 means unlabeled
    label = metadata[:, 1] - 1

    # make features into one-hot encodings
    feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        features = np.hstack((features, label_binarize(feat_col, classes=np.unique(feat_col))))

    g = dgl.graph((edge_index[0], edge_index[1]))
    g.ndata["feat"] = torch.tensor(features, dtype=torch.float)
    g.ndata["label"] = torch.tensor(label)

    return g, g.ndata["label"], len(np.unique(label))


def load_arxiv_year_data(
    n_classes: int = 5,
    directory: str = DEFAULT_DATA_DIR,
    verbosity: int = 0,
) -> Tuple[dgl.DGLGraph, torch.Tensor, int]:
    graph, _ = DglNodePropPredDataset(
        name="ogbn-arxiv",
        root=directory,
    )[0]

    label = torch.as_tensor(
        even_quantile_labels(
            vals=graph.ndata["year"].numpy().flatten(),
            n_classes=n_classes,
            verbosity=verbosity,
        )
    ).reshape(-1, 1)
    graph.ndata["label"] = label

    return graph, label, label.unique().shape[0]


def print_dataset_info(
    dataset_name: str,
    n_nodes: int,
    n_edges: int,
    n_feats: int,
    n_clusters: int,
    self_loops: int = None,
    is_directed: bool = None,
    thead: List[str] = None,
) -> None:
    dic = {
        "NumNodes": n_nodes,
        "NumEdges": n_edges,
        "NumFeats": n_feats,
        "NumClasses": n_clusters,
        "Self-loops": self_loops,
        "Directed": is_directed,
    }

    if self_loops is None:
        dic.pop("Self-loops")
    if is_directed is None:
        dic.pop("Directed")

    tab_printer(
        dic,
        thead=["Dataset", dataset_name] if thead is None else thead,
        cols_align=["c", "r"],
        sort=False,
    )


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
