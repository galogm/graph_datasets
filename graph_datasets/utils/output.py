"""Output Management.
"""
import csv
import os
from pathlib import Path
from pathlib import PurePath
from typing import Tuple


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
        is_dict (bool, optional): whether the tbody is in the format of a dict. Defaults to False.

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
                is_dict=False,
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
                is_dict=True,
            )
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
                dict_writer = csv.DictWriter(
                    csvfile,
                    fieldnames=tbody[0].keys(),
                )
                for elem in tbody:
                    dict_writer.writerow(elem)
            else:
                csv_write.writerow(tbody)
