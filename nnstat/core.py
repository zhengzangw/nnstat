import logging
import os
import pickle
from collections import OrderedDict
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LogNorm
from prettytable import PrettyTable
from scipy import stats

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


stats_ops = dict(
    # - norm
    num_params=lambda x: x.numel(),
    L1_norm=lambda x: x.norm(1).item(),
    L2_norm=lambda x: x.norm(2).item(),
    # L_+inf_norm
    abs_min=lambda x: x.abs().min().item(),
    # L_-inf_norm
    abs_max=lambda x: x.abs().max().item(),
    # - others
    L2_squared=lambda x: x.norm(2).item() ** 2,
    max=lambda x: x.max().item(),
    min=lambda x: x.min().item(),
)
layerwise_stats_ops = dict(
    # - moment
    mean=lambda x: x.mean().item(),
    sq_mean=lambda x: x.pow(2).mean().item(),
    variance=lambda x: x.var().item(),
    skew=lambda x: stats.skew(x, axis=None),
    kurtosis=lambda x: stats.kurtosis(x, axis=None),
    # - mean
    abs_mean=lambda x: x.norm(1).item() / x.numel(),
    std=lambda x: x.std().item(),
    sq_mean_std=lambda x: x.pow(2).mean().item() ** 0.5,
)
columns_group = dict(
    p1=[
        "num_params",
        "L1_norm",
        "L2_norm",
        "abs_min",
        "abs_max",
        "max",
        "min",
    ],
    p2=[
        "mean",
        "sq_mean",
        "variance",
        "skew",
        "kurtosis",
    ],
    p3=[
        "abs_mean",
        "std",
        "L2_squared",
        "sq_mean_std",
    ],
    default=[
        "L1_norm",
        "L2_norm",
        "mean",
        "variance",
        "sq_mean",
    ],
)


class NetDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: torch.Tensor):
        super().__setitem__(key, value)

    def __str__(self):
        s = f"{self.__class__.__name__}[L1={self.norm(1)}, L2={self.norm(2)}](\n"
        keys = list(self.keys())
        shapes = [str(tuple(value.shape)) for value in self.values()]
        max_key_len = max(len(key) for key in keys)
        max_shape_len = max(len(shape) for shape in shapes)
        for i, (key, shape) in enumerate(zip(keys, shapes)):
            s += f"  {i:0>2}: {key:<{max_key_len}} {shape:>{max_shape_len}}\n"
        s += ")"
        return s

    def __repr__(self):
        return self.__str__()

    def at(self, index):
        return self[list(self.keys())[index]]

    def save(self, path):
        save_obj(self, path)

    @classmethod
    def load(cls, path):
        return load_obj(path)

    def clone(self):
        new_dict = self.__class__()
        for key, value in self.items():
            new_dict[key] = value.clone()
        return new_dict

    def is_compatible(self, other):
        return self.keys() == other.keys()

    def __add__(self, other):
        if isinstance(other, self.__class__):
            assert self.is_compatible(other)
            return self.__class__(
                (key, value + other[key]) for key, value in self.items()
            )
        else:
            return self.__class__((key, value + other) for key, value in self.items())

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            assert self.is_compatible(other)
            return self.__class__(
                (key, value * other[key]) for key, value in self.items()
            )
        else:
            return self.__class__((key, value * other) for key, value in self.items())

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            assert self.is_compatible(other)
            return self.__class__(
                (key, value - other[key]) for key, value in self.items()
            )
        else:
            return self.__class__((key, value - other) for key, value in self.items())

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            assert self.is_compatible(other)
            return self.__class__(
                (key, value / other[key]) for key, value in self.items()
            )
        else:
            return self.__class__((key, value / other) for key, value in self.items())

    def apply(self, func):
        return self.__class__((key, func(value)) for key, value in self.items())

    def apply_(self, func):
        for key, value in self.items():
            self[key] = func(value)
        return self

    def zero(self):
        new_dict = self.__class__()
        for key, value in self.items():
            new_dict[key] = torch.zeros_like(value)
        return new_dict

    def zero_(self):
        return self.apply_(lambda x: x.zero_())

    def abs(self):
        return self.apply(lambda x: x.abs())

    def abs_(self):
        return self.apply_(lambda x: x.abs_())

    def sign(self):
        return self.apply(lambda x: x.sign())

    def sign_(self):
        return self.apply_(lambda x: x.sign_())

    def sqrt(self):
        return self.apply(lambda x: x.sqrt())

    def sqrt_(self):
        return self.apply_(lambda x: x.sqrt_())

    def numel(self):
        return sum(v.numel() for v in self.values())

    def norm(self, p=2):
        norm = 0
        for v in self.values():
            norm += v.norm(p) ** p
        norm = norm ** (1 / p)
        return norm

    def max(self):
        return max(v.max() for v in self.values())

    def min(self):
        return min(v.min() for v in self.values())

    def to(self, device=None):
        # for values on cuda, it is changable but in place
        # for values on cpu, it is not changable
        for key, value in self.items():
            self[key] = value.to(device)

    def op(self, op: str):
        assert op in stats_ops
        return stats_ops[op](self)

    def op_layerwise(self, op: str, pattern: Union[str, List[str]] = None):
        assert (
            op in layerwise_stats_ops
            or op in stats_ops
            or op in ["id", "layer_name", "shape"]
        )

        ret = []
        for i, (key, value) in enumerate(self.items()):
            if pattern is not None and not any([inc in key for inc in pattern]):
                continue
            if op in stats_ops:
                ret.append(stats_ops[op](value))
            elif op == "id":
                ret.append(i)
            elif op == "layer_name":
                ret.append(key)
            elif op == "shape":
                ret.append(list(value.shape))
            elif op in layerwise_stats_ops:
                ret.append(layerwise_stats_ops[op](value))
        return ret

    def describe(
        self,
        layerwise: bool = True,
        display: bool = True,
        group: str = "default",
        exclude: Union[str, List[str]] = None,
        # valid only for layerwise
        pattern: Union[str, List[str]] = None,
        include: pd.DataFrame = None,
    ):
        if isinstance(pattern, str):
            pattern = [pattern]

        if group is not None:
            assert group in columns_group
            columns = columns_group[group]
            if not layerwise:
                columns = [c for c in columns if c in stats_ops]
        else:
            columns = list(stats_ops.keys())
            if layerwise:
                columns = list(layerwise_stats_ops.keys()) + columns
        columns = exclude_from_columns(columns, exclude)
        if layerwise:
            columns = ["layer_name", "shape"] + columns
        ret = pd.DataFrame(columns=columns)

        for k in ret.columns:
            if not layerwise:
                ret[k] = [self.op(k)]
            else:
                ret[k] = self.op_layerwise(k, pattern)

        if include is not None and layerwise:
            if "layer_name" in include.columns:
                include = include.drop(columns=["layer_name"])
            ret = pd.concat([ret, include], axis=1)

        if display:
            print_title("Stats")
            print(ret)
        else:
            return ret

    def hist(
        self,
        op: str = "identity",
        bins: int = 100,
        logx: bool = False,
        logy: bool = False,
        layerwise: bool = True,
        pattern: Union[str, List[str]] = None,
    ):
        pattern_keys = pattern_filter(list(self.keys()), pattern)

        if not layerwise:
            logger.warning("Histogram of all parameters comsumes a lot of time.")
            values = []
            for key, value in self.items():
                if key in pattern_keys:
                    values.append(value.flatten())
            values = torch.cat(values)
            values = op_str(op, values)
            plot_hist(values, bins=bins, name=f"{op}_all_hist")
        else:
            directory = f"{op}_hist_{get_time()}"
            for key, value in self.items():
                if key in pattern_keys:
                    value = op_str(op, value)
                    plot_hist(
                        value,
                        bins=bins,
                        logx=logx,
                        logy=logy,
                        directory=directory,
                        name=f"{key}",
                    )

    def ecdf(
        self,
        op: str = "identity",
        bins: int = 100,
        logx: bool = False,
        logy: bool = False,
        layerwise: bool = True,
        pattern: Union[str, List[str]] = None,
    ):
        if not layerwise:
            raise NotImplementedError("heatmap of all parameters is not implemented")
        pattern_keys = pattern_filter(list(self.keys()), pattern)

        directory = f"{op}_ecdf_{get_time()}"
        for key, value in self.items():
            if key in pattern_keys:
                value = op_str(op, value)
                plot_ecdf(
                    value,
                    logx=logx,
                    logy=logy,
                    directory=directory,
                    name=key,
                )

    def heatmap(
        self,
        op: str = "identity",
        vmin: float = None,
        vmax: float = None,
        log: bool = False,
        layerwise: bool = True,
        pattern: Union[str, List[str]] = None,
    ):
        if not layerwise:
            raise NotImplementedError("heatmap of all parameters is not implemented")
        pattern_keys = pattern_filter(list(self.keys()), pattern)

        directory = f"{op}_heatmap_{get_time()}"
        for key, value in self.items():
            if key in pattern_keys:
                value = op_str(op, value)
                plot_heatmap(
                    value,
                    vmin=vmin,
                    vmax=vmax,
                    log=log,
                    directory=directory,
                    name=key,
                )

    @classmethod
    def get_weight(cls, model, requires_grad=False, cpu=True):
        ret = cls(
            (name, param.data)
            for name, param in model.named_parameters()
            if param.requires_grad or not requires_grad
        )
        if cpu:
            ret.to("cpu")
        return ret

    @classmethod
    def get_grad(cls, model, requires_grad=False, cpu=True):
        ret = cls(
            (name, param.grad.data)
            for name, param in model.named_parameters()
            if param.requires_grad or not requires_grad
        )
        if cpu:
            ret.to("cpu")
        return ret

    @classmethod
    def get_optimizer_state(cls, model, optimizer, optimizer_key="exp_avg", cpu=True):
        # adam: exp_avg, exp_avg_sq
        # sophia: exp_avg, hessian
        ret = cls(
            (name, optimizer.state[param][optimizer_key].cpu())
            for name, param in model.named_parameters()
            if param in optimizer.state
        )
        assert len(ret) > 0
        if cpu:
            ret.to("cpu")
        return ret


get_weight = NetDict.get_weight
get_grad = NetDict.get_grad
get_optimizer_state = NetDict.get_optimizer_state


def get_update(w0: NetDict, w1: NetDict, lr: float = 1.0):
    return (w1 - w0) / lr


def get_adam_update(m: NetDict, v: NetDict, eps: float = 1e-6):
    return m / (v.sqrt() + eps)
