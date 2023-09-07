"""Evaluation utils for graph tasks.
"""
# pylint: disable=invalid-name,too-many-locals
import os
from typing import Dict
from typing import Tuple

import torch

from .eval_tools import evaluate_results_nc


def load_from_file(file_name):
    # Load the embeddings from the .pth file
    try:
        embeddings = torch.load(file_name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File '{file_name}' not found.") from exc
    except Exception as el:
        raise RuntimeError(f"Error occurred while loading *.pth from '{file_name}'") from el

    return embeddings


def evaluate_from_embed_file(
    embedding_file: str,
    data_file: str,
    save_path: str = "./tmp/",
    quiet: bool = True,
) -> Tuple[Dict, Dict]:
    """Evaluation of representation quality using clustering and classification tasks.

    Args:
        embedding_file (str): Embedded file name.
        data_file (str): Data file name.
        save_path (str, optional): Folder path to store. Defaults to './tmp/'.
        quiet (bool, optional): Whether to print results. Defaults to True.

    Returns:
        Tuple[Dict, Dict]: Two dicts are included, \
            which are the evaluation results of clustering and classification.

    Example:
        .. code-block:: python

            from graph_datasets import evaluate_from_embed_file

            method_name='orderedgnn'
            data_name='texas'

            clustering_res, classification_res = evaluate_from_embed_file(
                f'{data_name}_{method_name}_embeds.pth',
                f'{data_name}_data.pth',
                save_path='./save/',
            )
    """
    embedding_file = os.path.join(save_path, embedding_file)
    data_file = os.path.join(save_path, data_file)

    embeddings = load_from_file(embedding_file).cpu().detach()
    data = load_from_file(data_file)

    # Call the evaluate_results_nc function with the loaded embeddings
    (
        svm_macro_f1_list,
        svm_micro_f1_list,
        acc_mean,
        acc_std,
        nmi_mean,
        nmi_std,
        ami_mean,
        ami_std,
        ari_mean,
        ari_std,
        f1_mean,
        f1_std,
    ) = evaluate_results_nc(
        data,
        embeddings,
        quiet=quiet,
        method="both",
    )

    # Format the output as desired
    clustering_results = {
        "ACC": f"{acc_mean * 100:.2f}±{acc_std * 100:.2f}",
        "NMI": f"{nmi_mean * 100:.2f}±{nmi_std * 100:.2f}",
        "AMI": f"{ami_mean * 100:.2f}±{ami_std * 100:.2f}",
        "ARI": f"{ari_mean * 100:.2f}±{ari_std * 100:.2f}",
        "Macro F1": f"{f1_mean * 100:.2f}±{f1_std * 100:.2f}",
    }

    svm_macro_f1_list = [f"{res[0] * 100:.2f}±{res[1] * 100:.2f}" for res in svm_macro_f1_list]
    svm_micro_f1_list = [f"{res[0] * 100:.2f}±{res[1] * 100:.2f}" for res in svm_micro_f1_list]

    classification_results = {}
    for i, percent in enumerate(["10%", "20%", "30%", "40%"]):
        classification_results[f"{percent}_Macro-F1"] = svm_macro_f1_list[i]
        classification_results[f"{percent}_Micro-F1"] = svm_micro_f1_list[i]

    return clustering_results, classification_results


# if __name__ == "__main__":
#     method_name = 'orderedgnn'  # 'selene' 'greet' 'hgrl' 'nwr-gae' 'orderedgnn'
#     data_name = 'texas'  # 'actor' 'chameleon' 'cornell' 'squirrel' 'texas' 'wisconsin'
#     print(method_name, data_name)

#     clu_res, cls_res = evaluate_from_embed_file(
#         f'{data_name}_{method_name}_embeds.pth',
#         f'{data_name}_data.pth',
#         save_path='/data/gnn/heter/save/',
#         quiet=True,
#     )
#     from graph_datasets.utils import tab_printer, save_to_csv_files
#     tab_printer(clu_res, sort=False)
#     tab_printer(cls_res, sort=False)

#     add_info = {
#         'data': data_name,
#         'method': method_name,
#     }
#     save_to_csv_files(clu_res, add_info, 'clutering.csv')
#     save_to_csv_files(cls_res, add_info, 'classification.csv')

#     # for data in ['squirrel', 'actor', 'cornell', 'texas', 'wisconsin', 'chameleon']:
#     #     print(method, data)
#     #     print(
#     #         evaluate_from_embed_file(
#     #             f'/data/gnn/heter/save/{data}_{method}_embeds.pth',
#     #             f'/data/gnn/heter/save/{data}_data.pth',
#     #         )
#     #     )
