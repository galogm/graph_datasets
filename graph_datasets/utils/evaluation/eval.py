from .eval_tools import evaluate_results_nc

import os
import torch

import warnings
warnings.filterwarnings("ignore")


def load_from_file(file_name):
    # Load the embeddings from the .pth file
    try:
        embeddings = torch.load(file_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_name}' not found.")
    except Exception as e:
        raise RuntimeError(f"Error occurred while loading *.pth from '{file_name}': {e}")

    return embeddings


def evaluate_embed_file(embedding_file: str, data_file: str, save_path='./tmp/'):
    """Evaluation of representation quality using clustering and classification tasks.

    Args:
        embedding_file (str): Embedded file name.
        data_file (str): Data file name.
        save_path (str, optional): Folder path to store. Defaults to './tmp/'.

    Returns:
        tuple: Two dict are included, which are the evaluation results of clustering and classification.

    Example:'''
        method_name='orderedgnn' # 'selene' 'greet' 'hgrl' 'nwr-gae' 'orderedgnn'
        data_name='texas' # 'actor' 'chameleon' 'cornell' 'squirrel' 'texas' 'wisconsin'
        print(method_name, data_name)

        clu_res, cls_res = evaluate_embed_file(
            f'{data_name}_{method_name}_embeds.pth', f'{data_name}_data.pth', save_path='./save/')
        print(clu_res, cls_res)
    '''
    """
    embedding_file = os.path.join(save_path, embedding_file)
    data_file = os.path.join(save_path, data_file)

    embeddings = load_from_file(embedding_file).cpu().detach()
    data = load_from_file(data_file)

    # Call the evaluate_results_nc function with the loaded embeddings
    svm_macro_f1_list, svm_micro_f1_list, acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std = \
        evaluate_results_nc(data, embeddings, quiet=False, method='both')

    # Format the output as desired
    clustering_results = {
        'ACC': f"{acc_mean:.4f}±{acc_std:.4f}",
        'NMI': f"{nmi_mean:.4f}±{nmi_std:.4f}",
        'ARI': f"{ari_mean:.4f}±{ari_std:.4f}",
        'F1': f"{f1_mean:.4f}±{f1_std:.4f}",
    }

    svm_macro_f1_list = [f"{res[0]:.4f}±{res[1]:.4f}" for res in svm_macro_f1_list]
    svm_micro_f1_list = [f"{res[0]:.4f}±{res[1]:.4f}" for res in svm_micro_f1_list]

    classification_results = {}
    for i, percent in enumerate(["10%", "20%", "30%", "40%"]):
        classification_results[f"{percent}_Macro-F1"] = svm_macro_f1_list[i]
        classification_results[f"{percent}_Micro-F1"] = svm_micro_f1_list[i]

    return clustering_results, classification_results


def save_to_csv_files(results: dict, add_info: dict, csv_name: str, save_path='.'):
    """Save the evaluation results to a local csv file.

    Args:
        results (dict): Evaluation results document.
        add_info (dict): Additional information, such as data set name, method name.
        csv_name (str): csv file name to store.
        save_path (str, optional): Folder path to store. Defaults to '.'.

    Example:'''
        method_name='orderedgnn' # 'selene' 'greet' 'hgrl' 'nwr-gae' 'orderedgnn'
        data_name='texas' # 'actor' 'chameleon' 'cornell' 'squirrel' 'texas' 'wisconsin'
        print(method_name, data_name)

        clu_res, cls_res = evaluate_embed_file(
            f'{data_name}_{method_name}_embeds.pth', f'{data_name}_data.pth', save_path='./save/')
        print(clu_res, cls_res)

        add_info = {'data': data_name, 'method': method_name,}
        save_to_csv_files(clu_res, add_info, 'clutering.csv')
        save_to_csv_files(cls_res, add_info, 'classification.csv')
    '''
    """
    # save to csv file
    results.update(add_info)

    from graph_datasets.utils import csv2file
    # list of values
    csv2file(
        target_path=os.path.join(save_path, csv_name),
        thead=list(results.keys()),
        tbody=list(results.values()),
        refresh=False,
        is_dict=False,
    )


if __name__ == "__main__":
    method_name='orderedgnn' # 'selene' 'greet' 'hgrl' 'nwr-gae' 'orderedgnn'
    data_name='texas' # 'actor' 'chameleon' 'cornell' 'squirrel' 'texas' 'wisconsin'
    print(method_name, data_name)

    clu_res, cls_res = evaluate_embed_file(
        f'{data_name}_{method_name}_embeds.pth', f'{data_name}_data.pth', save_path='./save/')
    print(clu_res, cls_res)

    add_info = {'data': data_name, 'method': method_name,}
    save_to_csv_files(clu_res, add_info, 'clutering.csv')
    save_to_csv_files(cls_res, add_info, 'classification.csv')


    # for data in ['squirrel', 'actor', 'cornell', 'texas', 'wisconsin', 'chameleon']:
    #     print(method, data)
    #     print(evaluate_embed_file(f'/data/gnn/heter/save/{data}_{method}_embeds.pth', f'/data/gnn/heter/save/{data}_data.pth'))
