"""Statistics
"""
# pylint:disable=duplicate-code
import csv
import os
from pathlib import Path
from pathlib import PurePath
from typing import Tuple

from graph_datasets import load_data
from graph_datasets.data_info import DATASETS
from graph_datasets.utils import statistics


def make_parent_dirs(target_path: PurePath) -> None:
    """make all the parent dirs of the target path.

    Args:
        target_path (PurePath): target path.
    """
    if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)


def refresh_file(target_path: str = None) -> None:
    """clear target path

    Args:
        target_path (str): file path
    """
    if target_path is not None:
        target_path: PurePath = Path(target_path)
        if target_path.exists():
            target_path.unlink()

        make_parent_dirs(target_path)
        target_path.touch()


def csv2file(
    target_path: str,
    thead: Tuple[str] = None,
    tbody: Tuple = None,
    refresh: bool = False,
    is_dict: bool = False,
) -> None:
    """save csv to target_path

    Args:
        target_path (str): target path
        thead (Tuple[str], optional): csv table header, only written into the file when\
            it is not None and file is empty. Defaults to None.
        tbody (Tuple, optional): csv table content. Defaults to None.
        refresh (bool, optional): whether to clean the file first. Defaults to False.
    """
    target_path: PurePath = Path(target_path)
    if refresh:
        refresh_file(target_path)

    make_parent_dirs(target_path)

    with open(target_path, "a+", newline="", encoding="utf-8") as csvfile:
        csv_write = csv.writer(csvfile)
        if os.stat(target_path).st_size == 0 and thead is not None:
            csv_write.writerow(thead)
        if tbody is not None:
            if is_dict:
                dict_writer = csv.DictWriter(csvfile, fieldnames=tbody[0].keys())
                for elem in tbody:
                    dict_writer.writerow(elem)
            else:
                csv_write.writerow(tbody)


def main():
    display_order = {
        "src": 1,
        "ds": 2,
        "n_nodes": 3,
        "n_feats": 4,
        "n_edges": 5,
        "n_clusters": 6,
        "eh_1h": 7,
        "nh_1h": 8,
        "lh_1h": 9,
        "ie_1h": 10,
        "bn_1h": 11,
        "eh_2h": 12,
        "nh_2h": 13,
        "lh_2h": 14,
        "ie_2h": 15,
        "bn_2h": 16,
        "n_edges_2h": 17,
        "eh_2h_uns": 18,
        "nh_2h_uns": 19,
        "lh_2h_uns": 20,
        "ie_2h_uns": 21,
        "bn_2h_uns": 22,
        "n_edges_2h_uns": 23,
    }
    for source, datasets in DATASETS.items():
        if source in [
                # remove the comment of the name to enable its calculation
                "pyg",
                "dgl",
                "ogb",
                "sdcn",
                "cola",
                "linkx",
        ]:
            continue
        for dataset in datasets:
            print(dataset)
            if dataset in [
                    # "products",
                    "mag",
                    "proteins",
                    "papers100M",
                    # "reddit",
                    # "twitch-gamers",
                    # "wiki",
                    # "arxiv",
                    # "cora",
                    # "citeseer",
                    # "pubmed",
                    # "corafull",
                    # "chameleon",
                    # "squirrel",
                    # "actor",
                    # "film",
                    # "cornell",
                    # "texas",
                    # "wisconsin",
                    # "computers",
                    # "photo",
                    # "cs",
                    # "physics",
                    # "wikics",
                    # "dblp",
                    # "acm",
                    # "blogcatalog",
                    # "flickr",
                    # "snap-patents",
                    # "pokec",
                    # "genius",
                    # "arxiv-year",
                    # "Penn94",
                    # "yelp-chi",
                    # "deezer-europe",
                    # "Amherst41",
                    # "Cornell5",
                    # "Johns Hopkins55",
                    # "Reed98",
            ]:
                continue
            graph, labels, n_clusters = load_data(
                dataset_name=dataset,
                verbosity=3,
                source=source,
            )

            dic = {}

            dic["ds"] = dataset
            dic["src"] = source
            dic["n_nodes"], dic["n_feats"] = graph.ndata["feat"].shape
            dic["n_edges"] = graph.num_edges()
            dic["n_clusters"] = n_clusters

            dic.update(statistics(graph, labels, dataset_name=dataset, h_1=False, h_2=True))

            csv2file(
                target_path="results/stat.csv",
                thead=display_order.keys(),
                tbody=dict(sorted(dic.items(), key=lambda d_: display_order[d_[0]])).values(),
            )


if __name__ == "__main__":
    main()
