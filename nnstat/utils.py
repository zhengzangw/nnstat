"""useful functions for nnstat"""

import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LogNorm

logging.basicConfig()
logger = logging.getLogger("nnstat")
CACHE_DIR = "_cache_nnstat"

__all__ = ["itemize", "set_cache_dir", "get_cache_dir", "plot_line", "plot_hist", "plot_ecdf", "plot_heatmap"]


def get_time():
    return datetime.today().strftime("%y%m%d%H%M%S")


def make_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_obj(obj, name: str, suffix: str = "pickle"):
    with open(f"{name}.{suffix}", "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {name}.{suffix}")


def load_obj(name: str):
    with open(f"{name}.pickle", "rb") as handle:
        obj = pickle.load(handle)
    return obj


# region: format
def str2list(s: Union[str, List[str]]) -> List[str]:
    if s is None:
        return []
    elif isinstance(s, str):
        return [s]
    else:
        return s


def itemize(
    t: Union[float, torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
) -> Union[float, List[float], Dict[str, float]]:
    """convert tensor to float

    Args:
        t (Union[float, torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]): tensor to convert

    Returns:
        Union[float, List[float], Dict[str, float]]: converted tensor
    """
    if isinstance(t, torch.Tensor) and t.numel() == 1:
        return t.item()
    if isinstance(t, list):
        return [itemize(x) for x in t]
    if isinstance(t, dict):
        return {k: itemize(v) for k, v in t.items()}
    return t


def print_title(name: str, width: int = 0):
    print(f"[{' '+name+' ':=^{width}}]")


# endregion: format


# region: plot
def set_cache_dir(path: str):
    """set cache directory, default is `cache_nnstat`

    Args:
        path (str): cache directory path
    """
    global CACHE_DIR
    CACHE_DIR = path
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_dir() -> str:
    """get cache directory

    Returns:
        str: cache directory path
    """
    return CACHE_DIR


def save_fig(name, directory=None, suffix="png", log=True):
    directory = CACHE_DIR if directory is None else os.path.join(CACHE_DIR, directory)

    make_dir(directory)
    fig_path = os.path.join(directory, f"{name}.{suffix}")
    plt.savefig(fig_path)
    plt.clf()
    if log:
        logger.warning(f"saved {fig_path}")


def plot_line(df: pd.DataFrame, name="tmp", **kwargs):
    x = df.columns[0]
    y = df.columns[1]
    sns.lineplot(data=df, x=x, y=y)

    save_fig(name, **kwargs)


def plot_hist(x, bins="auto", name="tmp", directory=None, logx=False, logy=False, **kwargs):
    """plot histogram and save

    Args:
        x (torch.Tensor): data
        directory (str, optional): directory. Defaults to None.
        bins (str, optional): bins. Defaults to "auto".
        name (str, optional): file name. Defaults to "tmp".
        logx (bool, optional): log scale x. Defaults to False.
        logy (bool, optional): log scale y. Defaults to False.
    """
    sns.histplot(x.flatten(), bins=bins, log_scale=(logx, logy), **kwargs)
    plt.title(name + f" {list(x.shape)}")
    save_fig(name, directory)


def plot_ecdf(x, name="tmp", directory=None, logx=False, logy=False, **kwargs):
    """plot ecdf and save

    Args:
        x (torch.Tensor): data
        name (str, optional): file name. Defaults to "tmp".
        directory (str, optional): directory. Defaults to None.
        logx (bool, optional): log scale x. Defaults to False.
        logy (bool, optional): log scale y. Defaults to False.
    """
    sns.ecdfplot(x.flatten(), log_scale=(logx, logy), **kwargs)
    plt.title(name + f" {list(x.shape)}")
    save_fig(name, directory)


def plot_heatmap(x, name="tmp", directory=None, vmin=None, vmax=None, log=False, **kwargs):
    """plot heatmap and save

    Args:
        x (torch.Tensor): data
        directory (str, optional): directory. Defaults to None.
        name (str, optional): file name. Defaults to "tmp".
        vmin ([type], optional): min value. Defaults to None.
        vmax ([type], optional): max value. Defaults to None.
        log (bool, optional): log scale. Defaults to False.
    """
    ori_shape = x.shape
    modified = False
    if x.dim() > 2:
        logger.warning(f"flattened {x.shape}")
        x = x.flatten(1)
        modified = True
    elif x.dim() == 1:
        logger.warning(f"unsqueeze {x.shape}")
        x = x.unsqueeze(0)
    sns.heatmap(x, vmin=vmin, vmax=vmax, norm=LogNorm() if log else None, **kwargs)
    if modified:
        plt.title(name + f" {ori_shape}->{list(x.shape)}")
    else:
        plt.title(name + f" {ori_shape}")
    save_fig(name, directory)


# endregion: plot
