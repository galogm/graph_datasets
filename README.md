# Graph Datasets

<div align="center">

[![PYPI](https://img.shields.io/pypi/v/graph_datasets?style=flat)](https://pypi.org/project/graph-datasets/)  [![Latest Release](https://img.shields.io/github/v/tag/galogm/graph_datasets)](https://github.com/galogm/graph_datasets/tags)

</div>

## Installation

- python>=3.8
- torch>=1.12
- torch_geometric>=2.0
- dgl>=1.1

```bash
$ python -m pip install graph_datasets
```

## Usage
See [Graph Datasets](https://galogm.github.io/graph_datasets_docs) for docs.

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
|   idx |  source  |     dataset     |   n_nodes |   n_feats |     n_edges |   n_clusters |
|------:|:--------:|:---------------:|----------:|----------:|------------:|-------------:|
|     1 |   pyg    |      cora       |     2,708 |     1,433 |      10,556 |            7 |
|     2 |   pyg    |    citeseer     |     3,327 |     3,703 |       9,104 |            6 |
|     3 |   pyg    |     pubmed      |    19,717 |       500 |      88,648 |            3 |
|     4 |   pyg    |    corafull     |    19,793 |     8,710 |     126,842 |           70 |
|     5 |   pyg    |     reddit      |   232,965 |       602 | 114,615,892 |           41 |
|     6 |   pyg    |    chameleon    |     2,277 |     2,325 |      62,742 |            5 |
|     7 |   pyg    |    squirrel     |     5,201 |     2,089 |     396,706 |            5 |
|     8 |   pyg    |      actor      |     7,600 |       932 |      53,318 |            5 |
|     9 |   pyg    |     cornell     |       183 |     1,703 |         554 |            5 |
|    10 |   pyg    |      texas      |       183 |     1,703 |         558 |            5 |
|    11 |   pyg    |    wisconsin    |       251 |     1,703 |         900 |            5 |
|    12 |   pyg    |    computers    |    13,752 |       767 |     491,722 |           10 |
|    13 |   pyg    |      photo      |     7,650 |       745 |     238,162 |            8 |
|    14 |   pyg    |       cs        |    18,333 |     6,805 |     163,788 |           15 |
|    15 |   pyg    |     physics     |    34,493 |     8,415 |     495,924 |            5 |
|    16 |   pyg    |     wikics      |    11,701 |       300 |     431,206 |           10 |
|    17 |   dgl    |      cora       |     2,708 |     1,433 |      10,556 |            7 |
|    18 |   dgl    |    citeseer     |     3,327 |     3,703 |       9,104 |            6 |
|    19 |   dgl    |     pubmed      |    19,717 |       500 |      88,648 |            3 |
|    20 |   dgl    |    corafull     |    19,793 |     8,710 |     126,842 |           70 |
|    21 |   dgl    |     reddit      |   232,965 |       602 | 114,615,892 |           41 |
|    22 |   dgl    |    chameleon    |     2,277 |     2,325 |      62,742 |            5 |
|    23 |   dgl    |    squirrel     |     5,201 |     2,089 |     396,706 |            5 |
|    24 |   dgl    |      actor      |     7,600 |       932 |      53,318 |            5 |
|    25 |   dgl    |     cornell     |       183 |     1,703 |         554 |            5 |
|    26 |   dgl    |      texas      |       183 |     1,703 |         558 |            5 |
|    27 |   dgl    |    wisconsin    |       251 |     1,703 |         900 |            5 |
|    28 |   ogb    |    products     | 2,449,029 |       100 | 123,718,024 |           47 |
|    29 |   ogb    |      arxiv      |   169,343 |       128 |   2,315,598 |           40 |
|    30 |   sdcn   |      dblp       |     4,057 |       334 |       7,056 |            4 |
|    31 |   sdcn   |       acm       |     3,025 |     1,870 |      26,256 |            3 |
|    32 |   cola   |   blogcatalog   |     5,196 |     8,189 |     343,486 |            6 |
|    33 |   cola   |     flickr      |     7,575 |    12,047 |     479,476 |            9 |
|    34 |  linkx   |  snap-patents   | 2,923,922 |       269 |  27,945,090 |            5 |
|    35 |  linkx   |      pokec      | 1,632,803 |        65 |  44,603,928 |            3 |
|    36 |  linkx   |     genius      |   421,961 |        12 |   1,845,736 |            2 |
|    37 |  linkx   |   arxiv-year    |   169,343 |       128 |   2,315,598 |            5 |
|    38 |  linkx   |     Penn94      |    41,554 |     4,814 |   2,724,458 |            3 |
|    39 |  linkx   |  twitch-gamers  |   168,114 |         7 |  13,595,114 |            2 |
|    40 |  linkx   |      wiki       | 1,925,342 |       600 | 485,014,138 |            6 |
|    41 |  linkx   |     cornell     |       183 |     1,703 |         554 |            5 |
|    42 |  linkx   |    chameleon    |     2,277 |     2,325 |      62,742 |            5 |
|    43 |  linkx   |      film       |     7,600 |       932 |      53,318 |            5 |
|    44 |  linkx   |    squirrel     |     5,201 |     2,089 |     396,706 |            5 |
|    45 |  linkx   |      texas      |       183 |     1,703 |         558 |            5 |
|    46 |  linkx   |    wisconsin    |       251 |     1,703 |         900 |            5 |
|    47 |  linkx   |    yelp-chi     |    45,954 |        32 |   7,693,958 |            2 |
|    48 |  linkx   |  deezer-europe  |    28,281 |    31,241 |     185,504 |            2 |
|    49 |  linkx   |    Amherst41    |     2,235 |     1,193 |     181,908 |            3 |
|    50 |  linkx   |    Cornell5     |    18,660 |     4,735 |   1,581,554 |            3 |
|    51 |  linkx   | Johns Hopkins55 |     5,180 |     2,406 |     373,172 |            3 |
|    52 |  linkx   |     Reed98      |       962 |       745 |      37,624 |            3 |
|    53 | critical |  roman-empire   |    22,662 |       300 |      65,854 |           18 |
|    54 | critical | amazon-ratings  |    24,492 |       300 |     186,100 |            5 |
|    55 | critical |   minesweeper   |    10,000 |         7 |      78,804 |            2 |
|    56 | critical |    tolokers     |    11,758 |        10 |   1,038,000 |            2 |
|    57 | critical |    questions    |    48,921 |       301 |     307,080 |            2 |
|    58 | critical |    squirrel     |     2,223 |     2,089 |      93,996 |            5 |
|    59 | critical |    chameleon    |       890 |     2,325 |      17,708 |            5 |
<!-- Statistics ends -->

## Requirements

See [requirements-dev.txt](./requirements-dev.txt), [requirements.txt](./requirements.txt) and [pyproject.toml:dependencies](./pyproject.toml).

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).
