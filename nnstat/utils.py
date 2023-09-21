import logging
import os
import pickle
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

logging.basicConfig()
logger = logging.getLogger("nnstat")
cache_dir = "cache_nnstat"


def set_cache_dir(path):
    global cache_dir
    cache_dir = path
    os.makedirs(cache_dir, exist_ok=True)


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def get_time():
    return datetime.today().strftime("%y%m%d%H%M%S")


def save_obj(obj, name, suffix="pickle"):
    with open(f"{name}.{suffix}", "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {name}.{suffix}")


def load_obj(name):
    with open(f"{name}.pickle", "rb") as handle:
        obj = pickle.load(handle)
    return obj


def print_title(name: str):
    print(f"[> {name} <]")


def process_table_element(value):
    if isinstance(value, float):
        return f"{value:.5g}"
    return value


def op_str(s: str, value):
    assert s in ["abs", "square", "identity"]
    if s == "abs":
        return value.abs()
    elif s == "square":
        return value**2
    else:
        return value


def save_fig(name, directory=None, suffix="png", log=True):
    if directory is None:
        directory = cache_dir
    else:
        directory = os.path.join(cache_dir, directory)

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


def exclude_from_columns(columns: List[str], exclude: Union[str, List[str]] = None):
    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        for e in exclude:
            if e in columns:
                columns.remove(e)
    return columns


def pattern_filter(keys, pattern):
    if pattern is None:
        return keys
    if isinstance(pattern, str):
        pattern = [pattern]
    return [key for key in keys if any([inc in key for inc in pattern])]
