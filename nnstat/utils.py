import logging
import os
import pickle
import re
from datetime import datetime
from typing import Dict, List, Union, Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LogNorm

logging.basicConfig()
logger = logging.getLogger("nnstat")
CACHE_DIR = "cache_nnstat"


def set_cache_dir(path: str):
    global CACHE_DIR
    CACHE_DIR = path
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_dir():
    return CACHE_DIR


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


def pattern_filter(data_dict: Dict, pattern: Union[str, List[str]]):
    if pattern is None:
        return data_dict
    pattern = str2list(pattern)
    patterns = [re.compile(p) for p in pattern]
    return {k: v for k, v in data_dict.items() if any(p.match(k) for p in patterns)}


def math_reduction(math_func: Callable):
    def func(self, *args, layerwise: bool = False, pattern: Union[str, List[str]] = None, **kwargs):
        if layerwise:
            return pattern_filter({k: math_func(self, v, *args, **kwargs) for k, v in self.items()}, pattern)
        return math_func(self, self.flatten(), *args, **kwargs)

    return func


# region: format


def str2list(s: Union[str, List[str]]) -> List[str]:
    if s is None:
        return []
    if isinstance(s, str):
        return [s]
    return s


def itemize(t: Union[float, torch.Tensor]):
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


def plot_hist(x, bins="auto", name="tmp", logx=False, logy=False, **kwargs):
    shape = list(x.shape)
    x = x.flatten()
    sns.histplot(x, bins=bins, log_scale=(logx, logy))
    plt.title(name + f" {shape}")

    save_fig(name, **kwargs)


def plot_ecdf(x, name="tmp", logx=False, logy=False, **kwargs):
    shape = list(x.shape)
    x = x.flatten()
    sns.ecdfplot(x, log_scale=(logx, logy))
    plt.title(name + f" {shape}")

    save_fig(name, **kwargs)


def plot_heatmap(x, name="tmp", vmin=None, vmax=None, log=False, **kwargs):
    assert x.dim() <= 2
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if log:
        sns.heatmap(x, vmin=vmin, vmax=vmax, norm=LogNorm())
    else:
        sns.heatmap(x, vmin=vmin, vmax=vmax)
    plt.title(name + f" {list(x.shape)}")

    save_fig(name, **kwargs)


# endregion: plot
