"""Data information.
"""
# pylint:disable=duplicate-code
import os

DEFAULT_DATA_DIR = os.path.abspath("./data")

# paper [SDCN](https://github.com/bdy9527/SDCN)
SDCN_URL = "https://github.com/bdy9527/SDCN/blob/da6bb007b7"

# paper [CoLA](https://github.com/GRAND-Lab/CoLA)
COLA_URL = "https://github.com/GRAND-Lab/CoLA/blob/main"

# paper [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale)
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

# NOTE: main difference of dgl and pyG datasets
# NOTE: - dgl has self-loops while pyG removes them.
# NOTE: - dgl row normalizes features while pyG does not.
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
    "mag",
    "proteins",
    "papers100M",
]
SDCN_DATASETS = [
    "dblp",
    "acm",
]
COLA_DATASETS = [
    "blogcatalog",
    "flickr",
]
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
