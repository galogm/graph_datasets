"""Model Management.
"""
import os
import random
from pathlib import Path
from pathlib import PurePath
from typing import Tuple

import dgl
import numpy as np
import torch


def get_modelfile_path(model_filename: str) -> PurePath:
    model_path: PurePath = Path(f".checkpoints/{model_filename}.pt")
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
    return model_path


def check_modelfile_exists(model_filename: str) -> bool:
    if get_modelfile_path(model_filename).exists():
        return True
    return False


def save_model(
    model_filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    current_epoch: int,
    loss: float,
) -> None:
    """Save model, optimizer, current_epoch, loss to ``.checkpoints/${model_filename}.pt``.

    Args:
        model_filename (str): filename to save model.
        model (torch.nn.Module): model.
        optimizer (torch.optim.Optimizer): optimizer.
        current_epoch (int): current epoch.
        loss (float): loss.
    """
    model_path = get_modelfile_path(model_filename)
    torch.save(
        {
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        model_path,
    )


def load_model(
    model_filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """Load model from ``.checkpoints/${model_filename}.pt``.

    Args:
        model_filename (str): filename to load model.
        model (torch.nn.Module): model.
        optimizer (torch.optim.Optimizer): optimizer.

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
            [model, optimizer, epoch, loss]
    """

    model_path = get_modelfile_path(model_filename)
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return model, optimizer, epoch, loss


def set_seed(seed: int = 4096) -> None:
    """Set random seed.

    NOTE:
        !!! `conv` and `neighborSampler` of dgl are somehow nondeterministic !!!

        Set seeds according to:
            - `pytorch doc <https://pytorch.org/docs/1.9.0/notes/randomness.html>`_
            - `cudatoolkit doc \
                <https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility>`_
            - `dgl issue <https://github.com/dmlc/dgl/issues/3302>`_

    Args:
        seed (int, optional): random seed. Defaults to 4096.
    """
    if seed is not False:
        os.environ["PYTHONHASHSEED"] = str(seed)
        # required by torch: Deterministic behavior was enabled with either
        # `torch.use_deterministic_algorithms(True)` or
        # `at::Context::setDeterministicAlgorithms(true)`,
        # but this operation is not deterministic because it uses CuBLAS and you have
        # CUDA >= 10.2. To enable deterministic behavior in this case,
        # you must set an environment variable before running your PyTorch application:
        # CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.
        # For more information, go to
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # if you are using multi-GPU.
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
        # NOTE: dgl.seed will occupy cuda:0 no matter which gpu is set if seed is set before device
        # see the issue：https://github.com/dmlc/dgl/issues/3054
        dgl.seed(seed)


def set_device(gpu: str = "0") -> torch.device:
    """Set torch device.

    Args:
        gpu (str): args.gpu. Defaults to '0'.

    Returns:
        torch.device: torch device. `device(type='cuda: x')` or `device(type='cpu')`.
    """
    max_device = torch.cuda.device_count() - 1
    if gpu == "none":
        print("Use CPU.")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        if not gpu.isnumeric():
            raise ValueError(
                f"args.gpu:{gpu} is not a single number for gpu setting."
                f"Multiple GPUs parallelism is not supported."
            )

        if int(gpu) <= max_device:
            print(f"GPU available. Use cuda:{gpu}.")
            device = torch.device(f"cuda:{gpu}")
            torch.cuda.set_device(device)
        else:
            print(f"cuda:{gpu} is not in available devices [0, {max_device}]. Use CPU instead.")
            device = torch.device("cpu")
    else:
        print("GPU is not available. Use CPU instead.")
        device = torch.device("cpu")
    return device
