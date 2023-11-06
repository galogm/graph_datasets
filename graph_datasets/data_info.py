"""Data source information.
"""
# pylint:disable=duplicate-code
#: Default directory for data saving.
DEFAULT_DATA_DIR = "./data"

#: Downloading url of datasets in paper
#: `SDCN <https://github.com/bdy9527/SDCN>`_.
SDCN_URL = "https://github.com/bdy9527/SDCN/blob/da6bb007b7"

#: Downloading url of datasets in paper
#: `CoLA <https://github.com/GRAND-Lab/CoLA>`_.
COLA_URL = "https://github.com/GRAND-Lab/CoLA/blob/main"

#: Downloading url of datasets in paper
#: `LINKX <https://github.com/CUAI/Non-Homophily-Large-Scale>`_.
LINKX_URL = "https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c/data"
LINKX_DRIVE_ID = {
    "snap-patents": "1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia",
    "pokec": "1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y",
    "yelp-chi": "1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ",
    "wiki_views": "1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP",  # Wiki 1.9M
    "wiki_edges": "14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u",  # Wiki 1.9M
    "wiki_features": "1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK",  # Wiki 1.9M
    "twitch-gamer_feat": "1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR",
    "twitch-gamer_edges": "1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0",
}

#: Downloading url of datasets in paper
#: `A Critical Look at the Evaluation of GNNs Under Heterophily:
#: Are We Really Making Progress?* <https://openreview.net/pdf?id=tJbbQfw-5wv>`_.
CRITICAL_URL = "https://github.com/yandex-research/heterophilous-graphs/blob/a431395/data"

#: Supported datasets of pyG.
#:
#: NOTE:
#:      main difference of dgl and pyG datasets
#:          - dgl has self-loops while pyG removes them.
#:          - dgl row normalizes features while pyG does not.
PYG_DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "corafull",
    "reddit",
    "chameleon",
    "squirrel",
    "actor",
    "cornell",
    "texas",
    "wisconsin",
    "computers",
    "photo",
    "cs",
    "physics",
    "wikics",
]

#: Supported datasets of dgl.
DGL_DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "corafull",
    "reddit",
    "chameleon",
    "squirrel",
    "actor",
    "cornell",
    "texas",
    "wisconsin",
]
OGB_DATASETS = [
    "products",
    "arxiv",
    # "mag",
    # "proteins",
    # "papers100M",
]
#: Datasets in paper
#: `SDCN <https://github.com/bdy9527/SDCN>`_.
SDCN_DATASETS = [
    "dblp",
    "acm",
]
#: Datasets in paper
#: `CoLA <https://github.com/GRAND-Lab/CoLA>`_.
COLA_DATASETS = [
    "blogcatalog",
    "flickr",
]
#: Datasets in paper
#: `LINKX <https://github.com/CUAI/Non-Homophily-Large-Scale>`_.
LINKX_DATASETS = [
    "snap-patents",
    "pokec",
    "genius",
    "arxiv-year",
    "Penn94",
    "twitch-gamers",
    "wiki",
    "cornell",
    "chameleon",
    # dataset film is also called as actor
    "film",
    "squirrel",
    "texas",
    "wisconsin",
    "yelp-chi",
    "deezer-europe",
    "Amherst41",
    "Cornell5",
    "Johns Hopkins55",
    "Reed98",
]

#: Datasets in paper
#: `A Critical Look at the Evaluation of GNNs Under Heterophily:
#: Are We Really Making Progress?* <https://openreview.net/pdf?id=tJbbQfw-5wv>`_.
CRITICAL_DATASETS = [
    "roman-empire",
    "amazon-ratings",
    "minesweeper",
    "tolokers",
    "questions",
    "squirrel",
    "chameleon",
]

DATASETS = {
    "pyg": PYG_DATASETS,
    "dgl": DGL_DATASETS,
    "ogb": OGB_DATASETS,
    "sdcn": SDCN_DATASETS,
    "cola": COLA_DATASETS,
    "linkx": LINKX_DATASETS,
    "critical": CRITICAL_DATASETS,
}
