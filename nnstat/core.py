import logging
import os
import pickle
from collections import OrderedDict
from pprint import pprint
from typing import Callable, Dict, List, Tuple, Union
from datetime import datetime

import matplotlib.pyplot as plt
import torch
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


def print_title(name: str, width: int = 50, symbol="="):
    print(f"{' ' + name + ' ':=^{width}}")


def process_table_element(value):
    if isinstance(value, float):
        return f"{value:.5g}"
    return value


def print_table(table_dict):
    tab = PrettyTable()
    tab.align = "l"
    tab.field_names = list(table_dict.keys())
    length = len(list(table_dict.values())[0])
    for i in range(length):
        row = [process_table_element(value[i]) for value in table_dict.values()]
        tab.add_row(row)
    print(tab)


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


def plot_curve(x, y: dict, name="tmp", x_label=None, y_label=None, **kwargs):
    for key, value in y.items():
        plt.plot(x, value, label=key)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    save_fig(name, **kwargs)


def plot_hist(x, bins=100, name="tmp", **kwargs):
    plt.hist(x, bins=bins)
    plt.title(name)

    save_fig(name, **kwargs)


class ResultDict(OrderedDict):
    def __init__(self, columns: List[str], exclude: Union[str, List[str]] = None):
        super().__init__()
        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            for e in exclude:
                if e in columns:
                    columns.remove(e)

        self.columns = columns
        for c in columns:
            self[c] = []


stats_ops = dict(
    num_params=lambda x: x.numel(),
    # first-order origin moment
    L1_norm=lambda x: x.norm(1).item(),
    # second-order central moment
    L2_squared=lambda x: x.norm(2).item() ** 2,
    L2_norm=lambda x: x.norm(2).item(),
    max=lambda x: x.max().item(),
    min=lambda x: x.min().item(),
    abs_mean=lambda x: x.norm(1).item() / x.numel(),
    abs_max=lambda x: x.abs().max().item(),
)
layerwise_stats_ops = dict(
    id=None,
    layer_name=None,
    shape=lambda x: list(x.shape),
    # first-order central moment
    mean=lambda x: x.mean().item(),
    # second-order central moment
    variance=lambda x: x.var().item(),
    std=lambda x: x.std().item(),
    # third-order central moment
    skew=lambda x: stats.skew(x, axis=None),
    # fourth-order central moment
    kurtosis=lambda x: stats.kurtosis(x, axis=None),
)
columns_group = dict(
    stats=[
        "num_params",
        "L1_norm",
        "L2_norm",
    ],
    default=[
        "id",
        "layer_name",
        "shape",
        "L1_norm",
        "L2_squared",
        "mean",
        "variance",
        "skew",
        "kurtosis",
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
        assert self.is_compatible(other)
        return self.__class__((key, value + other[key]) for key, value in self.items())

    def __mul__(self, other):
        assert self.is_compatible(other)
        return self.__class__((key, value * other[key]) for key, value in self.items())

    def __sub__(self, other):
        assert self.is_compatible(other)
        return self.__class__((key, value - other[key]) for key, value in self.items())

    def __truediv__(self, other):
        assert self.is_compatible(other)
        return self.__class__((key, value / other[key]) for key, value in self.items())

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
        assert op in layerwise_stats_ops or op in stats_ops

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
        layerwise: bool = False,
        display: bool = True,
        group: str = "default",
        exclude: Union[str, List[str]] = None,
        # valid only for layerwise
        pattern: Union[str, List[str]] = None,
        include=None,
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
        ret = ResultDict(columns, exclude)

        for k in ret.columns:
            if not layerwise:
                ret[k] = [self.op(k)]
            else:
                ret[k] = self.op_layerwise(k, pattern)

        if include is not None and layerwise:
            for key, value in include.items():
                if key not in columns:
                    ret[key] = value

        if display:
            print_title("Stats")
            print_table(ret)
        else:
            return ret

    def hist(
        self,
        op: str = "identity",
        bins: int = 100,
        layerwise: bool = False,
        pattern: Union[str, List[str]] = None,
    ):
        if isinstance(pattern, str):
            pattern = [pattern]
        assert op in ["identity", "abs", "square"]

        if not layerwise:
            logger.warning("Histogram of all parameters comsumes a lot of time.")
            values = []
            for key, value in self.items():
                if pattern is not None and not any([inc in key for inc in pattern]):
                    continue
                values.append(value.flatten())
            values = torch.cat(values)

            if op == "abs":
                values = values.abs()
            elif op == "square":
                values = values**2

            plot_hist(values, bins=bins, name=f"{op}_all_hist")
        else:
            directory = f"{op}_hist_{get_time()}"
            for key, value in self.items():
                if pattern is not None and not any([inc in key for inc in pattern]):
                    continue
                if op == "abs":
                    value = value.abs()
                elif op == "square":
                    value = value**2
                plot_hist(
                    value.flatten(),
                    bins=bins,
                    directory=directory,
                    name=f"{key}{list(value.shape)}",
                )

    def heatmap(
        self,
        op: str = "identity",
        layerwise: bool = False,
        pattern: Union[str, List[str]] = None,
    ):
        assert layerwise
        assert op in ["identity", "abs", "square"]

        for key, value in self.items():
            if pattern is not None and not any([inc in key for inc in pattern]):
                continue
            if op == "abs":
                value = value.abs()
            elif op == "square":
                value = value**2

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
            if param.grad is not None
        )
        if cpu:
            ret.to("cpu")
        return ret


get_weight = NetDict.get_weight
get_grad = NetDict.get_grad
get_optimizer_state = NetDict.get_optimizer_state


def get_update(w0: NetDict, w1: NetDict, lr: float = 1.0):
    return (w1 - w0) / lr
