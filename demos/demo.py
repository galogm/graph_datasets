"""Demo
"""
from graph_datasets import load_data

graph, label, n_clusters = load_data(
    dataset_name="cora",
    directory="./data",
    source="dgl",
    verbosity=2,
)
