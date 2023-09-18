import pickle
from collections import OrderedDict
from pprint import pprint
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import torch
from prettytable import PrettyTable


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


def plot_curve(x, y: dict, name="torchmonitor", x_label=None, y_label=None):
    plt.clf()
    for key, value in y.items():
        plt.plot(x, value, label=key)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"{name}.png")
    print(f"[torchmonitor] saved {name}.png")


class NetDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: torch.Tensor):
        super().__setitem__(key, value)

    def __str__(self):
        s = f"{self.__class__.__name__}(\n"
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

    def clone(self):
        new_dict = self.__class__()
        for key, value in self.items():
            new_dict[key] = value.clone()
        return new_dict

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

    @classmethod
    def load(cls, path):
        return load_obj(path)

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

    def describe(self, display=True):
        info = dict()
        info["num params"] = self.num_params
        info["L1 norm"] = self.norm(1)
        info["L1 norm averaged"] = self.norm(1) / self.num_params
        info["L2 norm"] = self.norm(2)
        info["L2 norm squared"] = self.norm(2) ** 2

        if display:
            pprint(info, sort_dicts=False)

        return info

    def describe_layers(self, display=True, include: Union[str, List[str]] = None):
        infos = [
            "id",
            "layer name",
            "shape",
            "L1 norm",
            "L2 norm",
            "L1 norm averaged",
            "L2 norm squared",
            "max abs value",
        ]
        ret = OrderedDict()
        tab = PrettyTable()
        tab.field_names = infos
        tab.align = "l"

        if isinstance(include, str):
            include = [include]

        for i, (key, value) in enumerate(self.items()):
            if include is not None and not any([inc in key for inc in include]):
                continue

            L1_norm = value.norm(1).item()
            L1_norm_averaged = L1_norm / value.numel()
            L2_norm = value.norm(2).item()
            L2_norm_squared = L2_norm**2
            max_abs_value = value.abs().max().item()
            shape = list(value.shape)
            info_values = [
                i,
                key,
                shape,
                L1_norm,
                L2_norm,
                L1_norm_averaged,
                L2_norm_squared,
                max_abs_value,
            ]
            ret[key] = dict(zip(infos, info_values))

            for i in range(3, len(info_values)):
                info_values[i] = f"{info_values[i]:.5g}"
            tab.add_row(info_values)

        if display:
            print(tab)

        return ret

    def is_compatible(self, other):
        return self.keys() == other.keys()


get_weight = NetDict.get_weight
get_grad = NetDict.get_grad
get_optimizer_state = NetDict.get_optimizer_state


def get_update(w0: NetDict, w1: NetDict):
    return w1 - w0
