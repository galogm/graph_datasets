# Graph Datasets

<div align="center">

[![PYPI](https://img.shields.io/pypi/v/graph_datasets?style=flat)](https://pypi.org/project/graph-datasets/)  [![Latest Release](https://img.shields.io/github/v/tag/galogm/graph_datasets)](https://github.com/galogm/graph_datasets/tags)

</div>

## Installation

```bash
$ python -m pip install graph_datasets
```

## Usage

```python
from graph_datasets import load_data

graph, label, n_clusters = load_data(
    dataset_name='cora',
    directory='./data',
    source='pyg',
    verbosity=1,
)
```

<!-- - DEV

```bash
# install cuda 11.3 if necessary
$ sudo bash scripts/cuda.sh
# see installation logs in logs/install.log
$ nohup bash scripts/install-dev.sh && bash scripts/install.sh > logs/install-dev.log &
```

- PROD

```bash
# see installation logs in logs/install.log
$ nohup bash scripts/install.sh > logs/install.log &
``` -->

<!-- Statistics begins -->
## Statistics
| idx | source |  dataset  | n_nodes | n_feat |     n_edges | n_clusters |
|-----|--------|-----------|---------|--------|-------------|------------|
|   1 |  pyg   |   cora    |   2,708 |  1,433 |      10,556 |          7 |
|   2 |  pyg   | citeseer  |   3,327 |  3,703 |       9,104 |          6 |
|   3 |  pyg   |  pubmed   |  19,717 |    500 |      88,648 |          3 |
|   4 |  pyg   | corafull  |  19,793 |  8,710 |     126,842 |         70 |
|   5 |  pyg   |  reddit   | 232,965 |    602 | 114,615,892 |         41 |
|   6 |  pyg   | chameleon |   2,277 |  2,325 |      62,742 |          5 |
|   7 |  pyg   | squirrel  |   5,201 |  2,089 |     396,706 |          5 |
|   8 |  pyg   |   actor   |   7,600 |    932 |      53,318 |          5 |
|   9 |  pyg   |  cornell  |     183 |  1,703 |         554 |          5 |
|  10 |  pyg   |   texas   |     183 |  1,703 |         558 |          5 |
|  11 |  pyg   | wisconsin |     251 |  1,703 |         900 |          5 |

<!-- Statistics ends -->

## Requirements

See in `requirements-dev.txt` and `requirements.txt`.
