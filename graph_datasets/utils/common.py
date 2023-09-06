"""Common utils.
"""
import os
import sys
import time
from typing import Any
from typing import Dict
from typing import List

import gdown
from texttable import Texttable


def get_str_time():
    """Return localtime in the format of "%m%d%H%M%S"."""
    return time.strftime("%m%d%H%M%S", time.localtime())


def format_value(value) -> Any:
    """Return number as string with comma split.

    Args:
        value (int): number.

    Returns:
        str: string of the number with comma split.
    """
    if f"{value}".isdecimal():
        return f"{value:,}"
    return value


def tab_printer(
    args: Dict,
    thead: List[str] = None,
    cols_align: List[str] = None,
    cols_valign: List[str] = None,
    cols_dtype: List[str] = None,
    sort: bool = True,
) -> None:
    """Function to print the logs in a nice tabular format.


    Args:
        args (Dict): value dict.
        thead (List[str], optional): table head. Defaults to None.
        cols_align (List[str], optional): horizontal alignment of the columns. Defaults to None.
        cols_valign (List[str], optional): vertical alignment of the columns. Defaults to None.
        cols_dtype (List[str], optional): value types of the columns. Defaults to None.
        sort (bool, optional): whether to sort the keys. Defaults to True.

    Returns:
        str: table string to print.
    """
    args = vars(args) if hasattr(args, "__dict__") else args
    keys = sorted(args.keys()) if sort else args.keys()
    table = Texttable()
    table.set_precision(5)
    params = [[] if thead is None else thead]
    params.extend(
        [
            [
                k.replace("_", " "),
                f"{args[k]}" if isinstance(args[k], bool) else format_value(args[k]),
            ] for k in keys
        ]
    )
    if cols_align is not None:
        table.set_cols_align(cols_align)
    if cols_valign is not None:
        table.set_cols_valign(cols_valign)
    if cols_dtype is not None:
        table.set_cols_dtype(cols_dtype)
    table.add_rows(params)

    print(table.draw())

    return table.draw()


def download_tip(info: Dict) -> None:
    """Tips for Downloading datasets

    Args:
        data_file (str): filepath.
        url (str): url for downloading.
    """
    info["Tip"] = "If the download fails, \
use the 'Download URL' to download manually and move the file to the 'Save Path'."

    tab_printer(info)


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


def bar_progress(current, total, _):
    """create this bar_progress method which is invoked automatically from wget"""
    progress_message = f"Downloading: {current / total * 100}% [{current} / {total}] bytes"
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download_from_google_drive(
    gid: str,
    output: str,
    quiet: bool = False,
    file_name: str = None,
) -> None:
    """Download data from google drive.

    Args:
        id (str): Id for google drive url.
        output (str): Path to save data.
        quiet (bool): Suppress terminal output. Default is False.
        file_name (str): File name. Default to None.
    """
    if not quiet:
        info = {
            "File": file_name if file_name is not None else os.path.basename(output),
            "Drive ID": gid,
            "Download URL": f"https://drive.google.com/uc?id={gid}",
            "Save Path": output,
        }
        download_tip(info)

    gdown.download(
        id=gid,
        output=output,
        quiet=quiet,
    )
