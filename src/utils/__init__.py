"""Utils
"""
# pylint:disable=invalid-name
import os
import sys
from typing import Any
from typing import Dict
from typing import List

import gdown
from texttable import Texttable


def format_value(value) -> Any:
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
        args (Dict): Parameters used for the model.
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
        tab_printer(info)

    gdown.download(
        id=gid,
        output=output,
        quiet=quiet,
    )
