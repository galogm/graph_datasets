"""Test
"""
import argparse

from graph_datasets.load_data import load_data


def main():
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
        default="./data",
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


if __name__ == "__main__":
    main()
