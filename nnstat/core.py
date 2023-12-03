"""Core functions and classes of nnstat."""

import logging
from collections import OrderedDict
from typing import Dict, List, Union

import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from .utils import itemize, print_title, str2list, pattern_filter, math_reduction

__all__ = [
    "from_weight",
    "from_grad",
    "from_optimizer_state",
    "StateDict",
    "zeros_like",
    "ones_like",
    "load",
    "compute_update",
    "compute_adam_update",
]

logging.basicConfig()
logger = logging.getLogger("nnstat")
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

columns_group = dict(
    all=[
        "name",
        "shape",
        "numel",
        "norm1",
        "norm1_mean",
        "norm2",
        "sum",
        "mean",
        "var",
        "std",
        "skew",
        "kurtosis",
        "max",
        "min",
        "abs_min",
        "abs_max",
    ],
    default=[
        "name",
        "shape",
        "numel",
        "norm1",
        "norm1_mean",
        "norm2",
    ],
)


class StateDict(OrderedDict):
    """A state dict is a dictionary of tensors."""

    # region: basic
    def __init__(self, *args, name: str = None, device: str = "cpu", copy: bool = True, **kwargs):
        self.device = device
        self.copy = copy
        super().__init__(*args, **kwargs)
        self._name = name if name is not None else self.__class__.__name__
        self._flattened = None

    def __setitem__(self, key: str, value: torch.Tensor):
        assert isinstance(value, torch.Tensor)
        if self.copy:
            value = value.detach().clone()
        value = value.to(self.device)
        super().__setitem__(key, value)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = f"{self._name}[L1={self.norm(1):.4g}, L2={self.norm(2):.4g}, Numel={self.numel():,d}]\n"
        keys = list(self.keys())
        shapes = [str(tuple(value.shape)) for value in self.values()]
        max_key_len = max(len(key) for key in keys)
        max_shape_len = max(len(shape) for shape in shapes)

        s += "(\n"
        for i, (key, shape) in enumerate(zip(keys, shapes)):
            s += f"\t{i:0>2}: {key:<{max_key_len}} {shape:>{max_shape_len}}\n"
        s += ")"
        return s

    # endregion: basic
    # region:basic-functions
    def set_name(self, name: str) -> "StateDict":
        """Set the name of the state dict."""
        if name is not None:
            self._name = name
        return self

    def to(self, device: str) -> "StateDict":
        """Move the state dict to a device."""
        self.device = device
        for key in self:
            self[key] = self[key].to(device)
        return self

    def at(self, index: int) -> torch.Tensor:
        """Get the value at a given index."""
        return self[list(self.keys())[index]]

    def clone(self) -> "StateDict":
        """Clone the state dict."""
        return StateDict({key: self[key].clone() for key in self})

    def StateDict(self):
        """Return a state dict of type dict."""
        return {key: self[key] for key in self}

    def save(self, path: str):
        """Save a state dict to a file."""
        torch.save(self.StateDict(), path)

    def is_compatible(self, other: "StateDict") -> bool:
        """Check if two state dicts are compatible.

        Args:
            other (StateDict): another state dict

        Returns:
            bool: True if compatible
        """
        return set(self.keys()) == set(other.keys())

    def flatten(self, lazy=True) -> torch.Tensor:
        """Flatten the state dict into a single tensor."""
        if lazy:
            if self._flattened is None:
                self._flattened = torch.cat([value.flatten() for value in self.values()])
            return self._flattened
        return torch.cat([value.flatten() for value in self.values()])

    # endregion: basic-functions
    # region: basic-math
    def __neg__(self) -> "StateDict":
        return StateDict({key: -value for key, value in self.items()}, device=self.device, copy=self.copy)

    def __add__(self, other: Union["StateDict", float, int]) -> "StateDict":
        if not isinstance(other, StateDict):
            return StateDict({key: value + other for key, value in self.items()})
        assert self.is_compatible(other)
        return StateDict({key: value + other[key] for key, value in self.items()}, device=self.device, copy=self.copy)

    def __radd__(self, other: Union["StateDict", float, int]) -> "StateDict":
        return self + other

    def __mul__(self, other: Union[float, int]) -> "StateDict":
        return StateDict({key: value * other for key, value in self.items()}, device=self.device, copy=self.copy)

    def __rmul__(self, other: Union[float, int]) -> "StateDict":
        return self * other

    def __sub__(self, other: Union["StateDict", float, int]) -> "StateDict":
        return self + (-other)

    def __rsub__(self, other: Union["StateDict", float, int]) -> "StateDict":
        return other + (-self)

    def __truediv__(self, other: Union[float, int]) -> "StateDict":
        return self * (1 / other)

    def __rtruediv__(self, other: Union[float, int]) -> "StateDict":
        return other * (1 / self)

    def __pow__(self, other: Union[float, int]) -> "StateDict":
        return StateDict({key: value**other for key, value in self.items()}, device=self.device, copy=self.copy)

    def sqrt(self) -> "StateDict":
        return StateDict({key: value.sqrt() for key, value in self.items()}, device=self.device, copy=self.copy)

    def abs(self) -> "StateDict":
        return StateDict({key: value.abs() for key, value in self.items()}, device=self.device, copy=self.copy)

    def abs_(self) -> "StateDict":
        for key, value in self.items():
            self[key] = value.abs()
        return self

    def apply(self, func) -> "StateDict":
        return StateDict({key: func(value) for key, value in self.items()}, device=self.device, copy=self.copy)

    def apply_(self, func) -> "StateDict":
        for key, value in self.items():
            self[key] = func(value)
        return self

    def zero(self) -> "StateDict":
        return StateDict(
            {key: torch.zeros_like(value) for key, value in self.items()}, device=self.device, copy=self.copy
        )

    def zero_(self) -> "StateDict":
        for key, value in self.items():
            self[key] = torch.zeros_like(value)
        return self

    def sign(self) -> "StateDict":
        return StateDict({key: value.sign() for key, value in self.items()}, device=self.device, copy=self.copy)

    def sign_(self) -> "StateDict":
        for key, value in self.items():
            self[key] = value.sign()
        return self

    # endregion: basic-math
    # region: math
    def max(self, layerwise: bool = False) -> Union[float, OrderedDict]:
        if layerwise:
            return OrderedDict({key: value.max() for key, value in self.items()})
        return self.flatten().max()

    def min(self, layerwise: bool = False) -> Union[float, OrderedDict]:
        if layerwise:
            return OrderedDict({key: value.min() for key, value in self.items()})
        return self.flatten().min()

    def abs_max(self, layerwise: bool = False) -> Union[float, OrderedDict]:
        if layerwise:
            return OrderedDict({key: value.abs().max() for key, value in self.items()})
        return self.flatten().abs().max()

    def abs_min(self, layerwise: bool = False) -> Union[float, OrderedDict]:
        if layerwise:
            return OrderedDict({key: value.abs().min() for key, value in self.items()})
        return self.flatten().abs().min()

    def name(self, layerwise: bool = False) -> Union[str, OrderedDict]:
        """Return the name of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of names per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[str, OrderedDict]: name
        """
        if layerwise:
            return OrderedDict({key: key for key in self.keys()})
        return self._name

    def numel(self, layerwise: bool = False) -> Union[int, OrderedDict]:
        """Compute the total number of elements in the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of number of elements per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[int, OrderedDict]: number of elements
        """

        if layerwise:
            return OrderedDict({key: value.numel() for key, value in self.items()})
        return self.flatten().numel()

    @math_reduction
    def shape(self, layerwise: bool = False) -> Union[torch.Size, OrderedDict]:
        """Compute the shape of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of shapes per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[torch.Size, OrderedDict]: shape
        """
        if layerwise:
            return OrderedDict({key: value.shape for key, value in self.items()})
        return self.flatten().shape

    @math_reduction
    def norm(self, v: torch.Tensor, p: float = 2):
        """Compute the norm of the state dict.

        Args:
            p (float, optional): norm. Defaults to 2.
            layerwise (bool, optional): If True, return a dict of norms per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: norm
        """
        return v.norm(p)

    @math_reduction
    def sum(self, v: torch.Tensor) -> Union[float, OrderedDict]:
        """Compute the sum of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of sums per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: sum
        """
        return self.flatten().sum()

    def norm1(self, *args, **kwargs) -> Union[float, OrderedDict]:
        """Compute the L1 norm of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of L1 norms per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: L1 norm
        """
        return self.norm(*args, p=1, **kwargs)

    def norm2(self, *args, **kwargs) -> Union[float, OrderedDict]:
        """Compute the L2 norm of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of L2 norms per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: L2 norm
        """
        return self.norm(*args, p=2, **kwargs)

    def norm1_mean(self, layerwise: bool = False) -> Union[float, OrderedDict]:
        """Compute the mean of the L1 norm of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of means of L1 norms per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: mean of L1 norm
        """
        if layerwise:
            return OrderedDict({key: value.norm(1) / value.numel() for key, value in self.items()})
        return self.flatten().norm(1) / self.flatten().numel()

    def mean(self, layerwise: bool = False) -> Union[float, OrderedDict]:
        """Compute the mean of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of means per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: mean
        """
        if layerwise:
            return OrderedDict({key: value.mean() for key, value in self.items()})
        return self.flatten().mean()

    def var(self, layerwise: bool = False) -> Union[float, OrderedDict]:
        """Compute the variance of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of variances per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: variance
        """
        if layerwise:
            return OrderedDict({key: value.var() for key, value in self.items()})
        return self.flatten().var()

    def std(self, layerwise: bool = False) -> Union[float, OrderedDict]:
        """Compute the standard deviation of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of standard deviations per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: standard deviation
        """
        if layerwise:
            return OrderedDict({key: value.std() for key, value in self.items()})
        return self.flatten().std()

    def skew(self, layerwise: bool = False) -> Union[float, OrderedDict]:
        """Compute the skewness of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of skewness per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: skewness
        """
        if layerwise:
            return OrderedDict({key: stats.skew(value, axis=None) for key, value in self.items()})
        return stats.skew(self.flatten(), axis=None)

    def kurtosis(self, layerwise: bool = False) -> Union[float, OrderedDict]:
        """Compute the kurtosis of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of kurtosis per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: kurtosis
        """
        if layerwise:
            return OrderedDict({key: stats.kurtosis(value, axis=None) for key, value in self.items()})
        return stats.kurtosis(self.flatten(), axis=None)

    # endregion: math

    def describe(
        self,
        layerwise: bool = False,
        display: bool = True,
        pattern: Union[str, List[str]] = None,
        group: str = None,
        include_keys: Union[str, List[str]] = None,
        exlude_keys: Union[str, List[str]] = None,
        additional_info: Dict[str, torch.Tensor] = None,
    ) -> Union[None, Dict[str, Dict[str, float]]]:
        """Display a summary of the state dict.

        Args:
            layerwise (bool, optional): If True, display layerwise stats. Defaults to False.
            display (bool, optional): If True, display the summary. Defaults to True.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.
            group (str, optional): group of keys. Defaults to None.
            include_keys (Union[str, List[str]], optional): keys to include. Defaults to None.
            exlude_keys (Union[str, List[str]], optional): keys to exclude. Defaults to None.
            additional_info (Dict[str, torch.Tensor], optional): additional info to display. It should be like {"norm1": {"layer1": value1, "layer2": value2}, "norm2": {...}} Defaults to None.

        Returns:
            Union[None, Dict[str, Dict[str, float]]]: stats
        """
        include_keys = str2list(include_keys)
        exlude_keys = str2list(exlude_keys)

        if group is None:
            columns = []
        else:
            assert group in columns_group
            columns = columns_group[group]
        # exclude keys from columns
        for k in exlude_keys:
            if k in columns:
                columns.remove(k)
        # include keys in columns
        for k in include_keys:
            if k not in columns:
                columns.append(k)

        if display:
            ret = pd.DataFrame(columns=columns)
        else:
            ret = dict()
            for k in self.keys():
                ret[k] = dict()
        for k in columns:
            stat = getattr(self, k)(layerwise=layerwise)
            stat = itemize(stat)
            if layerwise:
                stat = pattern_filter(stat, pattern)
            if not layerwise:
                ret[k] = [stat] if display else stat
            else:
                if display:
                    ret[k] = stat.values()
                else:
                    for key, value in stat.items():
                        ret[key][k] = value

        if additional_info is not None:
            assert layerwise
            for k, result in additional_info.items():
                stat = itemize(result)
                stat = pattern_filter(stat, pattern)
                if display:
                    ret[k] = stat.values()
                else:
                    for key, value in stat.items():
                        ret[key][k] = value

        if display:
            ret_str = ret.to_string(
                formatters=dict(name=lambda x: f"{x:<{max([len(n) for n in ret.name])}}"), justify="left"
            )
            print_title(f"Stats {self._name}", width=len(ret_str.split("\n")[0]))
            print(ret_str)
        else:
            return ret


# region: factory
def StateDict(
    raw_StateDict: Dict[str, torch.Tensor], device: str = "cpu", copy: bool = True, name: str = None
) -> StateDict:
    """Create a StateDict from a raw state dict.

    Args:
        raw_StateDict (Dict[str, torch.Tensor]): raw state dict
        device (str, optional): device. Defaults to "cpu".
        copy (bool, optional): copy and detach tensors. Defaults to True.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict
    """
    return StateDict(raw_StateDict, device=device, copy=copy, name=name)


def from_weight(
    model: nn.Module, requires_grad: bool = False, device: str = "cpu", copy: bool = True, name: str = None
) -> StateDict:
    """Create a StateDict from a model.

    Args:
        model (nn.Module): model
        requires_grad (bool, optional): keep track of gradient-needed ones. Defaults to False.
        device (str, optional): device. Defaults to "cpu".
        copy (bool, optional): copy and detach tensors. Defaults to True.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict of weights
    """
    state = StateDict(device=device, copy=copy)
    for key, param in model.named_parameters():
        if param.requires_grad or not requires_grad:
            state[key] = param
    if name is None:
        name = f"{model.__class__.__name__}_weights"
    state.set_name(name)
    return state


def from_grad(model: nn.Module, device: str = "cpu", copy: bool = True, name: str = None) -> StateDict:
    """Create a StateDict from a model.

    Args:
        model (nn.Module): model
        device (str, optional): device. Defaults to "cpu".
        copy (bool, optional): copy and detach tensors. Defaults to True.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict of gradients
    """
    state = StateDict(device=device, copy=copy)
    for key, param in model.named_parameters():
        if param.grad is not None:
            state[key] = param.grad
    if name is None:
        name = f"{model.__class__.__name__}_grad"
    state.set_name(name)
    return state


def from_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_key: str = "exp_avg",
    device: str = "cpu",
    copy: bool = True,
    name: str = None,
) -> StateDict:
    """Create a StateDict from a model and an optimizer.

    Args:
        model (nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        optimizer_key (str, optional): key of optimizer state. Defaults to "exp_avg".
        device (str, optional): device. Defaults to "cpu".
        copy (bool, optional): copy and detach tensors. Defaults to True.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict of optimizer state
    """
    state = StateDict(device=device, copy=copy)
    for key, param in model.named_parameters():
        if param in optimizer.state:
            state[key] = optimizer.state[param][optimizer_key]
    if name is None:
        name = f"{model.__class__.__name__}_optimizer_state"
    state.set_name(name)
    return state


def zeros_like(state: StateDict, name: str = None) -> StateDict:
    """Create a StateDict with the same keys as the input state dict, but filled with zeros.

    Args:
        state (StateDict): state dict
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict of zeros
    """
    return StateDict(
        {key: torch.zeros_like(state[key]) for key in state}, device=state.device, copy=state.copy, name=name
    )


def ones_like(state: StateDict, name: str = None) -> StateDict:
    """Create a StateDict with the same keys as the input state dict, but filled with ones.

    Args:
        state (StateDict): state dict
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict of ones
    """
    return StateDict(
        {key: torch.ones_like(state[key]) for key in state}, device=state.device, copy=state.copy, name=name
    )


def load(path: str, device: str = "cpu", copy: bool = True, name: str = None) -> StateDict:
    """Load a state dict from a file.

    Args:
        path (str): path to file
        device (str, optional): device. Defaults to "cpu".
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict
    """
    return StateDict(torch.load(path, map_location=device), device=device, copy=copy, name=name)


# endregion: factory


def compute_update(w0: StateDict, w1: StateDict, alpha: float) -> StateDict:
    """Compute the update of w0 to w1.

    Args:
        w0 (StateDict): initial state dict
        w1 (StateDict): final state dict
        alpha (float): step size

    Returns:
        StateDict: update
    """
    assert w0.is_compatible(w1)
    return (w1 - w0) / alpha


def compute_adam_update(m: StateDict, v: StateDict, eps: float = 1e-6):
    """Compute the update of Adam.

    Args:
        m (StateDict): first moment
        v (StateDict): second moment
        eps (float, optional): epsilon. Defaults to 1e-6.

    Returns:
        StateDict: update
    """
    return m / (v.sqrt() + eps)
