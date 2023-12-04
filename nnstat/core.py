"""
Core functions and classes of NNstat.
"""

import logging
from collections import OrderedDict
from typing import Callable, Dict, List, Union

import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from .utils import itemize, pattern_filter, print_title, str2list

__all__ = [
    "StateDict",
    "from_state_dict",
    "from_weight",
    "from_grad",
    "from_optimizer_state",
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
    """
    `StateDict` is the main class of nnstat. It is a wrap over dictionary of tensors, such as the returns from `model.state_dict()`. It provides many useful functions to analyze the state dict and support basic math operations.

    For example:

    .. code-block:: python

        > import nnstat
        > from torchvision import models
        > state = nnstat.from_state_dict(models.resnet18().state_dict())
        # or recommended way
        > state = nnstat.from_weight(models.resnet18())
        > print(state)
        StateDict[L1=2.407e+05, L2=132.6, Numel=11,699,132]
        (
            00: conv1.weight                (64, 3, 7, 7)
            01: bn1.weight                  (64,)
            02: bn1.bias                    (64,)
            03: bn1.running_mean            (64,)
            04: bn1.running_var             (64,)
            05: bn1.num_batches_tracked     ()
            (...truncated)
        )

    The math functions can be categorized into two types. The first type applys element-wise operations to each tensor in the state dict. For example, to calculate adam update for analysis, we can do

    .. code-block:: python

        m = nnstat.from_optimizer_state(model, optimizer, optimizer_key="exp_avg")
        v = nnstat.from_optimizer_state(model, optimizer, optimizer_key="exp_avg_sq")
        update = m / (v.sqrt() + 1e-8)


    The second type applys reduction operations to the flattened state dict. These methods have a `layerwise` argument, which can be set to True to return a dict of results per layer. For example, to calculate the L1 norm of the state dict, we can do

    .. code-block:: python

        > state.norm1(p=2)
        tensor(113.1479)
        > state.norm1(p=2, layerwise=True)
        {'conv1.weight': tensor(2.4467), 'bn1.weight': tensor(8.), 'bn1.bias': tensor(0.), 'layer1.0.conv1.weight': tensor(11.2160), (...truncated)}
        # use regex to filter keys
        > state.norm1(p=2, layerwise=True, pattern=".*conv.*")
        {'conv1.weight': tensor(2.4467), 'layer1.0.conv1.weight': tensor(11.2160), 'layer1.0.conv2.weight': tensor(11.3411), (...truncate)}

    We provide three ways to effectively examine the state dict status. The first is to use `describe` method for debugging in terminal, which can display a summary of the state dict.

    .. code-block:: bash

        > state.describe()
        [============================ Stats ResNet_weights ============================]
        name            shape         numel     norm1         norm1_mean  norm2
        0  ResNet_weights  (11689512,)  11689512  235774.53125  0.02017     113.108963
        > state.describe(layerwise=True, pattern=".*conv.*")
        [=========================== Stats ResNet_weights ===========================]
            name                   shape              numel    norm1        norm2
        0   conv1.weight              (64, 3, 7, 7)     9408    188.525146   2.438898
        1   layer1.0.conv1.weight    (64, 64, 3, 3)    36864   1736.147461  11.318304
        2   layer1.0.conv2.weight    (64, 64, 3, 3)    36864   1736.389282  11.336217
        3   layer1.1.conv1.weight    (64, 64, 3, 3)    36864   1733.171997  11.316651
        4   layer1.1.conv2.weight    (64, 64, 3, 3)    36864   1724.930786  11.236347
        (...truncated)

    The second way is to export the statistics for tools such as tensorboard and wandb. To export the statistics, do the following:

    .. code-block:: bash

        > state.describe(display=False, include_keys=['norm1', 'norm2'])
        {'norm1': 235752.828125, 'norm2': 113.10881805419922}

    The third way is to visualize the statistics by plotting.
    """

    # region: basic
    def __init__(
        self,
        *args,
        name: str = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        copy: bool = True,
        **kwargs,
    ):
        """`StateDict` is a subclass of OrderedDict. We recommend create StateDict using the factory functions such as `from_state_dict`, `from_weight`, `from_grad`, `from_optimizer_state`.

        Args:
            name (str, optional): The name of the state dict. Defaults to None.
            device (str, optional): The device of the state dict. All tensors in state dict are assumed to be on the same device. All tensors and tensors added in the future will be moved to the device. Defaults to "cpu".
            dtype (torch.dtype, optional): The dtype of the state dict. All tensors in state dict are assumed to be of the same dtype. All tensors and tensors added in the future will be casted to the dtype. Defaults to torch.float32.
            copy (bool, optional): Make a copy of the original tensors. This can avoid modifying original values. Defaults to True.
        """
        self.device = device
        self.dtype = dtype
        self.copy = copy
        super().__init__(*args, **kwargs)
        self._name = name if name is not None else self.__class__.__name__
        self._flattened = None

    def __setitem__(self, key: str, value: torch.Tensor):
        assert isinstance(value, torch.Tensor)
        if self.copy:
            value = value.detach().clone()
        value = value.to(dtype=self.dtype, device=self.device)
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
        """Set the name of the state dict.

        Args:
            name (str): name

        Returns:
            StateDict: self
        """
        if name is not None:
            self._name = name
        return self

    def to(self, device: str = None, dtype: torch.dtype = None) -> "StateDict":
        """Move the state dict to a device and cast to a dtype.

        Args:
            device (str, optional): device. Defaults to None.
            dtype (torch.dtype, optional): dtype. Defaults to None.

        Returns:
            StateDict: self
        """

        self.device = device
        self.dtype = dtype
        for key in self:
            self[key] = self[key].to(device=device, dtype=dtype)
        return self

    def at(self, index: int) -> torch.Tensor:
        """Get the value at a given index.

        Args:
            index (int): index

        Returns:
            torch.Tensor: value
        """
        return self[list(self.keys())[index]]

    def clone(self) -> "StateDict":
        """Clone the state dict.

        Returns:
            StateDict: cloned state dict with the same device and dtype
        """
        return StateDict({key: self[key].clone() for key in self}, device=self.device, dtype=self.dtype, copy=True)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a state dict of type dict, which can be loaded by `torch.load`.

        Returns:
            dict[str, torch.Tensor]: state dict
        """
        return {key: self[key] for key in self}

    def save(self, path: str):
        """Save a state dict to a file.

        Args:
            path (str): path to file
        """
        torch.save(self.state_dict(), path)

    def is_compatible(self, other: "StateDict") -> bool:
        """Check if two state dicts are compatible.

        Args:
            other (StateDict): another state dict

        Returns:
            bool: True if compatible
        """
        return set(self.keys()) == set(other.keys())

    def flatten(self, lazy=True) -> torch.Tensor:
        """Flatten the state dict into a single tensor.

        Args:
            lazy (bool, optional): If True, return the cached flattened tensor. Defaults to True.

        Returns:
            torch.Tensor: flattened tensor
        """
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
        """Compute the square root of the state dict.

        Returns:
            StateDict: square root element-wise
        """
        return StateDict({key: value.sqrt() for key, value in self.items()}, device=self.device, copy=self.copy)

    def abs(self) -> "StateDict":
        """Compute the absolute value of the state dict.

        Returns:
            StateDict: absolute value element-wise
        """
        return StateDict({key: value.abs() for key, value in self.items()}, device=self.device, copy=self.copy)

    def abs_(self) -> "StateDict":
        """Compute the absolute value of the state dict inplace.

        Returns:
            StateDict: absolute value element-wise
        """
        for key, value in self.items():
            self[key] = value.abs()
        return self

    def apply(self, func: Callable[torch.Tensor, torch.Tensor]) -> "StateDict":
        """Apply a function to each tensor in state dict.

        Args:
            func (Callable[torch.Tensor, torch.Tensor]): function

        Returns:
            StateDict: state dict with function applied
        """
        return StateDict({key: func(value) for key, value in self.items()}, device=self.device, copy=self.copy)

    def apply_(self, func: Callable[torch.Tensor, torch.Tensor]) -> "StateDict":
        """Apply a function to each tensor in state dict inplace.

        Args:
            func (Callable[torch.Tensor, torch.Tensor]): function

        Returns:
            StateDict: state dict with function applied
        """
        for key, value in self.items():
            self[key] = func(value)
        return self

    def zero(self) -> "StateDict":
        """Return a state dict of zeros with the same keys and shapes as the input state dict.

        Returns:
            StateDict: state dict of zeros
        """
        return StateDict(
            {key: torch.zeros_like(value) for key, value in self.items()}, device=self.device, copy=self.copy
        )

    def zero_(self) -> "StateDict":
        """Fill the state dict with zeros inplace.

        Returns:
            StateDict: state dict of zeros
        """
        for key, value in self.items():
            self[key] = torch.zeros_like(value)
        return self

    def sign(self) -> "StateDict":
        """Compute the sign of the state dict.

        Returns:
            StateDict: sign
        """
        return StateDict({key: value.sign() for key, value in self.items()}, device=self.device, copy=self.copy)

    def sign_(self) -> "StateDict":
        """Compute the sign of the state dict inplace.

        Returns:
            StateDict: sign
        """
        for key, value in self.items():
            self[key] = value.sign()
        return self

    # endregion: basic-math
    # region: math
    def apply_reduction(
        self, func: Callable, layerwise: bool = False, pattern: Union[str, List[str]] = None, **kwargs
    ) -> Union[float, Dict]:
        """Apply a reduction function to the state dict.

        Args:
            func (Callable): reduction function
            layerwise (bool, optional): If True, return a dict of results per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: result
        """
        if layerwise:
            return pattern_filter({key: func(value, **kwargs) for key, value in self.items()}, pattern)
        return func(self.flatten(), **kwargs)

    def register_reduction(self, name: str, func: Callable):
        """Register a reduction function to the state dict.

        Args:
            name (str): name of the function
            func (Callable): reduction function
        """
        assert name not in columns_group["all"]
        columns_group["all"].append(name)
        setattr(self, name, lambda **kwargs: self.apply_reduction(func, **kwargs))

    def max(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[float, Dict]:
        """Compute the maximum of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of maximums per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: maximum
        """
        return self.apply_reduction(lambda x: x.max(), layerwise=layerwise, pattern=pattern)

    def min(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[float, Dict]:
        """Compute the minimum of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of minimums per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: minimum
        """
        return self.apply_reduction(lambda x: x.min(), layerwise=layerwise, pattern=pattern)

    def name(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[str, Dict]:
        """Get the name of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of names per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[str, Dict]: name
        """
        if layerwise:
            names = {key: key for key in self.keys()}
            return pattern_filter(names, pattern)
        return self._name

    def numel(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[int, Dict]:
        """Compute the number of elements of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of number of elements per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[int, Dict]: number of elements
        """
        return self.apply_reduction(lambda x: x.numel(), layerwise=layerwise, pattern=pattern)

    def shape(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[tuple, Dict]:
        """Get the shape of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of shapes per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[tuple, Dict]: shape
        """
        return self.apply_reduction(lambda x: x.shape, layerwise=layerwise, pattern=pattern)

    def norm(self, p: float = 2, *, layerwise: bool = False, pattern: Union[str, List[str]] = None):
        """Compute the Lp norm of the state dict.

        Args:
            p (float, optional): p. Defaults to 2.
            layerwise (bool, optional): If True, return a dict of Lp norms per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: Lp norm
        """
        return self.apply_reduction(lambda x: x.norm(p), layerwise=layerwise, pattern=pattern)

    def norm1(self, **kwargs) -> Union[float, Dict]:
        """Compute the L1 norm of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of L1 norms per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: L1 norm
        """
        return self.norm(p=1, **kwargs)

    def norm2(self, **kwargs) -> Union[float, Dict]:
        """Compute the L2 norm of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of L2 norms per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: L2 norm
        """
        return self.norm(p=2, **kwargs)

    def sum(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[float, Dict]:
        """Compute the sum of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of sums per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: sum
        """
        return self.apply_reduction(lambda x: x.sum(), layerwise=layerwise, pattern=pattern)

    def norm1_mean(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[float, Dict]:
        """Compute the mean of the L1 norm of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of means per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: mean of L1 norm
        """
        return self.apply_reduction(lambda x: x.norm(1) / x.numel(), layerwise=layerwise, pattern=pattern)

    def mean(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[float, Dict]:
        """Compute the mean of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of means per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: mean
        """
        return self.apply_reduction(lambda x: x.mean(), layerwise=layerwise, pattern=pattern)

    def var(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[float, Dict]:
        """Compute the variance of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of variances per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: variance
        """
        return self.apply_reduction(lambda x: x.var(), layerwise=layerwise, pattern=pattern)

    def std(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[float, Dict]:
        """Compute the standard deviation of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of standard deviations per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: standard deviation
        """
        return self.apply_reduction(lambda x: x.std(), layerwise=layerwise, pattern=pattern)

    def skew(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[float, Dict]:
        """Compute the skewness of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of skewnesses per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: skewness
        """
        return self.apply_reduction(lambda x: stats.skew(x, axis=None), layerwise=layerwise, pattern=pattern)

    def kurtosis(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[float, Dict]:
        """Compute the kurtosis of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of kurtosis per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: kurtosis
        """
        return self.apply_reduction(lambda x: stats.kurtosis(x, axis=None), layerwise=layerwise, pattern=pattern)

    def abs_max(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[float, Dict]:
        """Compute the absolute maximum of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of absolute maximums per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: absolute maximum
        """
        return self.apply_reduction(lambda x: x.abs().max(), layerwise=layerwise, pattern=pattern)

    def abs_min(self, *, layerwise: bool = False, pattern: Union[str, List[str]] = None) -> Union[float, Dict]:
        """Compute the absolute minimum of the state dict.

        Args:
            layerwise (bool, optional): If True, return a dict of absolute minimums per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: absolute minimum
        """
        return self.apply_reduction(lambda x: x.abs().min(), layerwise=layerwise, pattern=pattern)

    # endregion: math

    def describe(
        self,
        display: bool = True,
        layerwise: bool = False,
        pattern: Union[str, List[str]] = None,
        group: str = None,
        include_keys: Union[str, List[str]] = None,
        exlude_keys: Union[str, List[str]] = None,
        additional_info: Dict[str, torch.Tensor] = None,
    ) -> Union[None, Dict[str, Dict[str, float]]]:
        """Display a summary of the state dict. The pre-defined groups are `all` and `default`.
        `all`: ["name", "shape", "numel", "norm1", "norm1_mean", "norm2", "sum", "mean", "var", "std", "skew", "kurtosis", "max", "min", "abs_min", "abs_max"],
        `default`: ["name", "shape", "numel", "norm1", "norm1_mean", "norm2"]

        Args:
            display (bool, optional): If True, display the summary and return None. Defaults to True.
            layerwise (bool, optional): If True, display layerwise stats. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.
            group (str, optional): Group of keys. Defaults to None. If None and no include_keys are provided,  use the `default` group.
            include_keys (Union[str, List[str]], optional): Additional keys to include. Defaults to None.
            exlude_keys (Union[str, List[str]], optional): Keys to exclude. Defaults to None.
            additional_info (Dict[str, torch.Tensor], optional): additional info to display. It should be like {"norm1": {"layer1": value1, "layer2": value2}, "norm2": {...}} Defaults to None.

        Returns:
            Union[None, Dict[str, Dict[str, float]]]: stats
        """
        if group is None:
            columns = []
            if include_keys is None:
                columns = columns_group["default"]
        else:
            assert group in columns_group
            columns = columns_group[group]

        include_keys = str2list(include_keys)
        exlude_keys = str2list(exlude_keys)
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
            if layerwise:
                for k in self.keys():
                    ret[k] = dict()
        for k in columns:
            stat = getattr(self, k)(layerwise=layerwise, pattern=pattern)
            stat = itemize(stat)
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
def from_state_dict(
    raw_state_dict: Dict[str, torch.Tensor],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    copy: bool = True,
    name: str = None,
) -> StateDict:
    """Create a StateDict from a raw state dict.

    Args:
        raw_state_dict (Dict[str, torch.Tensor]): raw state dict
        device (str, optional): device. Defaults to "cpu".
        dtype (torch.dtype, optional): dtype. Defaults to torch.float32.
        copy (bool, optional): copy and detach tensors. Defaults to True.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict
    """
    return StateDict(raw_state_dict, device=device, dtype=dtype, copy=copy, name=name)


def from_weight(
    model: nn.Module,
    requires_grad: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    copy: bool = True,
    name: str = None,
) -> StateDict:
    """Create a StateDict from a model.

    Args:
        model (nn.Module): model
        requires_grad (bool, optional): keep track of gradient-needed ones. Defaults to False.
        device (str, optional): device. Defaults to "cpu".
        dtype (torch.dtype, optional): dtype. Defaults to torch.float32.
        copy (bool, optional): copy and detach tensors. Defaults to True.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict of weights
    """
    state = StateDict(device=device, dtype=dtype, copy=copy)
    for key, param in model.named_parameters():
        if param.requires_grad or not requires_grad:
            state[key] = param
    if name is None:
        name = f"{model.__class__.__name__}_weights"
    state.set_name(name)
    return state


def from_grad(
    model: nn.Module,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    copy: bool = True,
    name: str = None,
) -> StateDict:
    """Create a StateDict from a model.

    Args:
        model (nn.Module): model
        device (str, optional): device. Defaults to "cpu".
        dtype (torch.dtype, optional): dtype. Defaults to torch.float32.
        copy (bool, optional): copy and detach tensors. Defaults to True.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict of gradients
    """
    state = StateDict(device=device, dtype=dtype, copy=copy)
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
    dtype: torch.dtype = torch.float32,
    copy: bool = True,
    name: str = None,
) -> StateDict:
    """Create a StateDict from a model and an optimizer.

    Args:
        model (nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        optimizer_key (str, optional): key of optimizer state. Defaults to "exp_avg".
        device (str, optional): device. Defaults to "cpu".
        dtype (torch.dtype, optional): dtype. Defaults to torch.float32.
        copy (bool, optional): copy and detach tensors. Defaults to True.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict of optimizer state
    """
    state = StateDict(device=device, dtype=dtype, copy=copy)
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
        {key: torch.zeros_like(state[key]) for key in state},
        device=state.device,
        dtype=state.dtype,
        copy=state.copy,
        name=name,
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
        {key: torch.ones_like(state[key]) for key in state},
        device=state.device,
        dtype=state.dtype,
        copy=state.copy,
        name=name,
    )


def load(
    path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    copy: bool = True,
    name: str = None,
) -> StateDict:
    """Load a state dict from a file.

    Args:
        path (str): path to file
        device (str, optional): device. Defaults to "cpu".
        dtype (torch.dtype, optional): dtype. Defaults to torch.float32.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict
    """
    return StateDict(
        torch.load(path, map_location=device),
        device=device,
        dtype=dtype,
        copy=copy,
        name=name,
    )


# endregion: factory


def compute_update(w0: StateDict, w1: StateDict, alpha: float = 1.0) -> StateDict:
    """Compute the update of w0 to w1.

    Args:
        w0 (StateDict): initial state dict
        w1 (StateDict): final state dict
        alpha (float, optional): step size. Defaults to 1.0.

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
