"""Output Management.
"""
import csv
import os
from pathlib import Path
from pathlib import PurePath
from typing import Any
from typing import List


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
    thead: List[str] = None,
    tbody: List[Any] = None,
    refresh: bool = False,
    is_dict_list: bool = False,
    sort_head: bool = True,
) -> None:
    """save data to target_path of a csv file.

    Args:
        target_path (str): target path
        thead (List[str], optional): csv table header, only written into the file when\
            it is not None and file is empty. Defaults to None.
        tbody (List, optional): csv table content. Defaults to None.
        refresh (bool, optional): whether to clean the file first. Defaults to False.
        is_dict_list (bool, optional): whether the tbody is in the format of a list of dicts. \
            Defaults to False.
        sort_head (bool, optional): whether to sort the head before writing. Defaults to True.

    Example:
        .. code-block:: python

            from graph_datasets import csv2file
            save_file = "./results/example.csv"
            final_params = {
                "dataset": "cora",
                "acc": "99.1",
                "NMI": "89.0",
            }
            thead=[]
            # list of values
            csv2file(
                target_path=save_file,
                thead=list(final_params.keys()),
                tbody=list(final_params.values()),
                refresh=False,
                is_dict_list=False,
            )
            # list of dicts
            csv2file(
                target_path=save_file,
                tbody=[
                    {
                        "a": 1,
                        "b": 2
                    },
                    {
                        "a": 2,
                        "b": 1
                    },
                ],
                is_dict_list=True,
            )
    """
    target_path: PurePath = Path(target_path)
    if refresh:
        refresh_file(target_path)

    make_parent_dirs(target_path)

    with open(target_path, "a+", newline="", encoding="utf-8") as csvfile:
        csv_write = csv.writer(csvfile)
        if tbody is not None:
            if is_dict_list:
                if sort_head:
                    keys = sorted(list(tbody[0].keys()))
                    if os.stat(target_path).st_size == 0:
                        csv_write.writerow(keys)
                    tbody = [{k: b[k] for k in keys} for b in tbody]

                dict_writer = csv.DictWriter(
                    csvfile,
                    fieldnames=tbody[0].keys(),
                )
                for elem in tbody:
                    dict_writer.writerow(elem)
            else:
                if thead is not None:
                    if sort_head:
                        thead, tbody = list(zip(*sorted(zip(thead, tbody), key=lambda x: x[0])))
                    if os.stat(target_path).st_size == 0:
                        csv_write.writerow(thead)
                csv_write.writerow(tbody)


def save_to_csv_files(
    results: dict,
    add_info: dict,
    csv_name: str,
    save_path="./results",
) -> None:
    """Save the evaluation results to a local csv file.

    Args:
        results (dict): Evaluation results document.
        add_info (dict): Additional information, such as data set name, method name.
        csv_name (str): csv file name to store.
        save_path (str, optional): Folder path to store. Defaults to './results'.

    Example:
        .. code-block:: python

            from graph_datasets import evaluate_from_embed_file
            from graph_datasets import save_to_csv_files

            method_name='orderedgnn'
            data_name='texas'

            clustering_res, classification_res = evaluate_from_embed_file(
                f'{data_name}_{method_name}_embeds.pth',
                f'{data_name}_data.pth',
                save_path='./save/',
            )

            add_info = {'data': data_name, 'method': method_name,}
            save_to_csv_files(clustering_res, add_info, 'clutering.csv')
            save_to_csv_files(classification_res, add_info, 'classification.csv')
    """
    # save to csv file
    results.update(add_info)

    # list of values
    csv2file(
        target_path=os.path.join(save_path, csv_name),
        thead=list(results.keys()),
        tbody=list(results.values()),
        refresh=False,
        is_dict_list=False,
    )
