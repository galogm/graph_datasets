"""Utils for evaluation.
"""
# pylint: disable=invalid-name,invalid-name,too-many-locals
import os
import random
from datetime import datetime

import numpy as np
import torch
from munkres import Munkres
from six.moves import cPickle as pickle
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import f1_score as F1
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.svm import LinearSVC

from ..common import tab_printer


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


def save_dict(di_, filename_):
    # Get the directory path from the filename
    dir_path = os.path.dirname(filename_)

    # Create the directory if it does not exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def split_train_test_nodes(data, train_ratio, valid_ratio, data_name, split_id=0, fixed_split=True):
    if fixed_split:
        file_path = f"../input/fixed_splits/{data_name}-{train_ratio}-{valid_ratio}-splits.npy"
        if not os.path.exists(file_path):
            print("There is no generated fixed splits")
            print("Generating fixed splits...")
            splits = {}
            for idx in range(10):
                # set up train val and test
                shuffle = list(range(data.num_nodes))
                random.shuffle(shuffle)
                train_nodes = shuffle[:int(data.num_nodes * train_ratio / 100)]
                val_nodes = shuffle[
                    int(data.num_nodes * train_ratio /
                        100):int(data.num_nodes * (train_ratio + valid_ratio) / 100)]
                test_nodes = shuffle[int(data.num_nodes * (train_ratio + valid_ratio) / 100):]
                splits[idx] = {"train": train_nodes, "valid": val_nodes, "test": test_nodes}
            save_dict(di_=splits, filename_=file_path)
        else:
            splits = load_dict(filename_=file_path)
        split = splits[split_id]
        train_nodes, val_nodes, test_nodes = split["train"], split["valid"], split["test"]
    else:
        # set up train val and test
        shuffle = list(range(data.num_nodes))
        random.shuffle(shuffle)
        train_nodes = shuffle[:int(data.num_nodes * train_ratio / 100)]
        val_nodes = shuffle[int(data.num_nodes * train_ratio /
                                100):int(data.num_nodes * (train_ratio + valid_ratio) / 100)]
        test_nodes = shuffle[int(data.num_nodes * (train_ratio + valid_ratio) / 100):]

    return np.array(train_nodes), np.array(val_nodes), np.array(test_nodes)


def cluster_eval(y_true, y_pred):
    """code source: https://github.com/bdy9527/SDCN"""
    y_true = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)

    # NOTE: NOT force to assign a random node into a missing class
    # fill out missing classes
    # ind = 0
    # if numclass1 != numclass2:
    #     for i in l1:
    #         if i in l2:
    #             pass
    #         else:
    #             y_pred[ind] = i
    #             ind += 1

    # l2 = list(set(y_pred))
    # numclass2 = len(l2)

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = ACC(y_true, new_predict)
    f1_macro = F1(y_true, new_predict, average="macro")
    nmi = NMI(y_true, new_predict, average_method="arithmetic")
    ami = AMI(y_true, new_predict, average_method="arithmetic")
    ari = ARI(y_true, new_predict)
    return acc, nmi, ami, ari, f1_macro


def unsup_eval(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    acc, nmi, ami, ari, f1_macro = cluster_eval(y_true, y_pred)
    return acc, nmi, ami, ari, f1_macro


def kmeans_test(X, y, n_clusters, repeat=10):
    y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

    acc_list = []
    nmi_list = []
    ami_list = []
    ari_list = []
    f1_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        (
            acc_score,
            nmi_score,
            ami_score,
            ari_score,
            macro_f1,
        ) = unsup_eval(
            y_true=y,
            y_pred=y_pred,
        )
        acc_list.append(acc_score)
        nmi_list.append(nmi_score)
        ami_list.append(ami_score)
        ari_list.append(ari_score)
        f1_list.append(macro_f1)
    return (
        np.mean(acc_list),
        np.std(acc_list),
        np.mean(nmi_list),
        np.std(nmi_list),
        np.mean(ami_list),
        np.std(ami_list),
        np.mean(ari_list),
        np.std(ari_list),
        np.mean(f1_list),
        np.std(f1_list),
    )


def svm_test(data, embeddings, labels, train_ratios=(10, 20, 30, 40), repeat=10):
    result_macro_f1_list = []
    result_micro_f1_list = []
    for train_ratio in train_ratios:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            train_idx, val_idx, test_idx = split_train_test_nodes(
                data=data,
                train_ratio=train_ratio,
                valid_ratio=train_ratio,
                data_name=data.name,
                split_id=i,
            )
            X_train, X_test = embeddings[np.concatenate([train_idx, val_idx])], embeddings[test_idx]
            y_train, y_test = labels[np.concatenate([train_idx, val_idx])], labels[test_idx]
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = F1(y_test, y_pred, average="macro")
            micro_f1 = F1(y_test, y_pred, average="micro")
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(
    data,
    embeddings,
    quiet=False,
    method="unsup",
    alpha: float = 2.0,
    beta: float = 2.0,
):
    labels = data.y.detach().cpu().numpy()
    num_classes = data.num_classes
    num_nodes = data.num_nodes
    if embeddings.shape[0] > num_nodes:
        z_1 = embeddings[:num_nodes]
        z_2 = embeddings[num_nodes:]
        if (alpha <= 1) and (beta <= 1):
            embeddings = alpha * z_1 + beta * z_2
        else:
            embeddings = torch.cat((z_1, z_2), 1)

    if method in ("both", "sup"):
        (
            svm_macro_f1_list,
            svm_micro_f1_list,
        ) = svm_test(
            data=data,
            embeddings=embeddings,
            labels=labels,
        )
        if not quiet:
            print("SVM test")
            tab_printer(
                {
                    "Macro F1":
                        "\n".join(
                            [
                                f"{macro_f1_mean * 100:.2f}±{macro_f1_std * 100:.2f} ({train_size:.1f})"
                                for (macro_f1_mean, macro_f1_std
                                    ), train_size in zip(svm_macro_f1_list, [10, 20, 30, 40])
                            ]
                        ),
                    "Micro F1":
                        "\n".join(
                            [
                                f"{micro_f1_mean * 100:.2f}±{micro_f1_std * 100:.2f} ({train_size:.1f})"
                                for (micro_f1_mean, micro_f1_std
                                    ), train_size in zip(svm_micro_f1_list, [10, 20, 30, 40])
                            ]
                        ),
                },
                sort=False,
            )

    if method in ("both", "unsup"):
        (
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
        ) = kmeans_test(
            embeddings,
            labels,
            num_classes,
        )
        if not quiet:
            print("K-means test")
            tab_printer(
                {
                    "ACC": f"{acc_mean * 100:.2f}±{acc_std * 100:.2f}",
                    "NMI": f"{nmi_mean * 100:.2f}±{nmi_std * 100:.2f}",
                    "AMI": f"{ami_mean * 100:.2f}±{ami_std * 100:.2f}",
                    "ARI": f"{ari_mean * 100:.2f}±{ari_std * 100:.2f}",
                    "Macro F1": f"{f1_mean * 100:.2f}±{f1_std * 100:.2f}",
                },
                sort=False,
            )

    if method == "sup":
        acc_mean = acc_std = nmi_mean = nmi_std = ari_mean = ari_std = f1_mean = f1_std = 0
    elif method == "unsup":
        svm_macro_f1_list = svm_micro_f1_list = [(0, 0), (0, 0), (0, 0), (0, 0)]

    return (
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
    )


def save_embedding(
    node_embeddings: torch.tensor,
    dataset_name: str,
    model_name: str,
    params: dict,
    save_dir: str = "./save",
    verbose: bool or int = True,
):
    dataset_name = dataset_name.replace("_", "-")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{dataset_name.lower()}_{model_name.lower()}_embeds_{timestamp}.pth"
    file_path = os.path.join(save_dir, file_name)

    result = {
        "node_embeddings": node_embeddings.cpu().detach(),
        "hyperparameters": params,
    }

    torch.save(result, file_path)

    if verbose:
        print(f"Embeddings and hyperparameters saved to {file_path}")
