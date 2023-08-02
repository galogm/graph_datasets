"""Dataset Statistics
"""
# pylint:disable=duplicate-code
import re

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
        verbosity=0,
    )

    global idx
    idx = idx + 1
    n_nodes, n_feat = graph.ndata["feat"].shape
    table.add_row(
        [
            idx,
            source,
            dataset,
            format_value(n_nodes),
            format_value(n_feat),
            format_value(graph.num_edges()),
            format_value(n_clusters),
        ]
    )


if __name__ == "__main__":
    table = Texttable()
    table.add_row([
        'idx',
        'source',
        'dataset',
        'n_nodes',
        'n_feat',
        'n_edges',
        'n_clusters',
    ])
    table.set_cols_dtype(['i', 't', 't', 'i', 'i', 'i', 'i'])
    table.set_cols_align(['r', 'c', 'c', 'r', 'r', 'r', 'r'])
    for source, datasets in DATASETS.items():
        for dataset in datasets:
            main(source, dataset)
        break

    with open('./docs/rst/table.rst', 'w') as tf:
        tf.writelines("""
statistics
===========
""")
        tf.write(table.draw())

    with open('./README.md', 'r+', encoding='utf-8') as tf:
        txt = tf.read()
        tf.seek(0)
        tf.truncate(0)
        nt = table.draw().split('\n')
        nt = [i + '\n' for i in nt]
        for i in range(0, len(nt), 2):
            if (i == 2):
                nt[i] = nt[i].replace('+', '|')
            else:
                nt[i] = ""
        nt = "".join(nt)
        tf.write(
            re.sub(
                r"(?<=<!-- Statistics begins -->\n## Statistics\n)(.*)(?=\n<!-- Statistics ends -->)",
                nt,
                txt,
                flags=re.S,
            )
        )
