import logging
import os
import pickle
from collections import OrderedDict
from pprint import pprint
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import torch
from prettytable import PrettyTable

logging.basicConfig()
logger = logging.getLogger("nnstat")
cache_dir = "nnstat_cache"


def set_cache_dir(path):
    global cache_dir
    cache_dir = path
    os.makedirs(cache_dir, exist_ok=True)


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


def plot_curve(x, y: dict, name="torchmonitor", x_label=None, y_label=None):
    set_cache_dir(cache_dir)
    plt.clf()
    for key, value in y.items():
        plt.plot(x, value, label=key)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    fig_path = os.path.join(cache_dir, f"{name}.png")
    plt.savefig(fig_path)
    logger.warning(f"saved {fig_path}")


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

    @property
    def num_params(self):
        return sum(v.numel() for v in self.values())

    def norm(self, p=2):
        norm = 0
        for v in self.values():
            norm += v.norm(p) ** p
        norm = norm ** (1 / p)
        return norm.item()

    def max_value(self):
        return max(v.max() for v in self.values()).item()

    def min_value(self):
        return min(v.min() for v in self.values()).item()

    def to(self, device=None):
        # for values on cuda, it is changable but in place
        # for values on cpu, it is not changable
        for key, value in self.items():
            self[key] = value.to(device)

    def describe(
        self,
        display=True,
        exclude: Union[str, List[str]] = None,
    ):
        # columns
        columns = [
            "num_params",
            "L1_norm",
            "L2_norm",
            "L1_avg",
            "L2_squared",
        ]

        ret = ResultDict(columns, exclude)

        if "num_params" in columns:
            ret["num_params"].append(self.num_params)
        if "L1_norm" in columns:
            ret["L1_norm"].append(self.norm(1))
        if "L2_norm" in columns:
            ret["L2_norm"].append(self.norm(2))
        if "L1_avg" in columns:
            ret["L1_avg"].append(self.norm(1) / self.num_params)
        if "L2_squared" in columns:
            ret["L2_squared"].append(self.norm(2) ** 2)

        if display:
            print_title("Stats")
            print_table(ret)
        else:
            return ret

    def describe_layers(
        self,
        display=True,
        pattern: Union[str, List[str]] = None,
        exclude: Union[str, List[str]] = None,
        include=None,
    ):
        # default columns
        columns = [
            "id",
            "layer_name",
            "shape",
            "L1_norm",
            "L2_norm",
            "L1_avg",
            "L2_squared",
            "max_abs",
        ]

        ret = ResultDict(columns, exclude)

        for i, (key, value) in enumerate(self.items()):
            if pattern is not None and not any([inc in key for inc in pattern]):
                continue

            if "id" in columns:
                ret["id"].append(i)
            if "layer_name" in columns:
                ret["layer_name"].append(key)
            if "shape" in columns:
                ret["shape"].append(list(value.shape))
            if "L1_norm" in columns:
                ret["L1_norm"].append(value.norm(1).item())
            if "L2_norm" in columns:
                ret["L2_norm"].append(value.norm(2).item())
            if "L1_avg" in columns:
                ret["L1_avg"].append(value.norm(1).item() / value.numel())
            if "L2_squared" in columns:
                ret["L2_squared"].append(value.norm(2).item() ** 2)
            if "max_abs" in columns:
                ret["max_abs"].append(value.abs().max().item())

        if include is not None:
            for key, value in include.items():
                if key not in columns:
                    ret[key] = value

        if display:
            print_title("Layer-wise stats")
            print_table(ret)
        else:
            return ret

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
