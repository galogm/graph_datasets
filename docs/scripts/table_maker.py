"""Dataset Statistics
"""
# pylint:disable=duplicate-code,line-too-long
import re

import pandas as pd
from texttable import Texttable

from graph_datasets import load_data
from graph_datasets.data_info import DATASETS
from graph_datasets.utils import format_value

idx = 0


def main(_source, _dataset):
    graph, _, n_clusters = load_data(
        dataset_name=_dataset,
        directory="./data",
        source=_source,
        verbosity=1,
    )

    global idx
    idx = idx + 1
    n_nodes, n_feats = graph.ndata["feat"].shape
    tbody.append(
        [
            idx,
            source,
            dataset,
            format_value(n_nodes),
            format_value(n_feats),
            format_value(graph.num_edges()),
            format_value(n_clusters),
        ]
    )


if __name__ == "__main__":
    table = Texttable()
    thead = [
        'idx',
        'source',
        'dataset',
        'n_nodes',
        'n_feats',
        'n_edges',
        'n_clusters',
    ]
    tbody = []
    table.set_cols_dtype(['i', 't', 't', 'i', 'i', 'i', 'i'])
    table.set_cols_align(['r', 'c', 'c', 'r', 'r', 'r', 'r'])
    for source, datasets in DATASETS.items():
        for dataset in datasets:
            main(source, dataset)

    with open('./docs/rst/table.rst', 'w') as tf:
        tf.writelines("""statistics
===========
""")
        tbody.insert(0, thead)
        table.add_rows(tbody)
        tbody.pop(0)
        tf.write(table.draw())

    with open('./README.md', 'r+', encoding='utf-8') as tf:
        txt = tf.read()
        tf.seek(0)
        tf.truncate(0)
        tf.write(
            re.sub(
                r"(?<=<!-- Statistics begins -->\n## Statistics\n)(.*)(?=\n<!-- Statistics ends -->)",
                pd.DataFrame(tbody, columns=thead).to_markdown(index=False),
                txt,
                flags=re.S,
            )
        )
