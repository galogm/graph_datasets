# Template

## Installation

```sh
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

## Requirements

See in `requirements-dev.txt` and `requirements.txt`.
