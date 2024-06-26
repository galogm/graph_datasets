"""Test
"""

# pylint:disable=duplicate-code
from graph_datasets import load_data
from graph_datasets.data_info import DATASETS


def main(_source, _dataset):
    load_data(
        dataset_name=_dataset,
        directory="./data",
        source=_source,
        verbosity=3,
        raw_normalize=True,
        rm_self_loop=True,
        add_self_loop=True,
        to_simple=True,
    )

    # import argparse
    # parser = argparse.ArgumentParser(
    #     prog="Load Graph datasets",
    #     description="Load Graph datasets",
    # )
    # parser.add_argument(
    #     "-d",
    #     "--dataset_name",
    #     type=str,
    #     default="cora",
    #     help="Dataset name",
    # )
    # parser.add_argument(
    #     "-p",
    #     "--directory_path",
    #     type=str,
    #     default="./data",
    #     help="Data directory path",
    # )
    # parser.add_argument(
    #     "-s",
    #     "--source",
    #     type=str,
    #     default="pyg",
    #     help="Dataset source",
    # )
    # parser.add_argument(
    #     "-v",
    #     "--verbosity",
    #     action="count",
    #     help="Output debug information",
    #     default=0,
    # )
    # args = parser.parse_args()

    # load_data(
    #     dataset_name=args.dataset_name,
    #     directory=args.directory_path,
    #     source=args.source,
    #     verbosity=args.verbosity,
    # )


if __name__ == "__main__":
    for source, datasets in DATASETS.items():
        for dataset in datasets:
            main(source, dataset)
