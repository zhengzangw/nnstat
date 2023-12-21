"""
See :doc:`/tutorials/getting_started` for basic usage.
"""

import logging
import re
from collections import OrderedDict
from typing import Callable, Dict, List, Union

import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from . import utils

__all__ = [
    "from_weight",
    "from_grad",
    "from_optimizer_state",
    "from_together",
    "from_state_dict",
    "StateDict",
    "load",
]

logging.basicConfig()
logger = logging.getLogger("nnstat")
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

columns_group = dict(
    all=[
        "no",
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
        "no",
        "name",
        "shape",
        "numel",
        "norm1",
        "norm1_mean",
        "norm2",
    ],
    _light_dict=[
        "no",
        "name",
        "value",
    ],
)


class Pattern:
    def __init__(self, pattern: Union[None, str]):
        if pattern is None:
            self.pattern = None
        else:
            self.pattern = re.compile(pattern)

    def match(self, s: str):
        if self.pattern is None:
            return True
        return self.pattern.match(s) is not None


# region: factory
def from_state_dict(
    raw_state_dict: Dict[str, torch.Tensor],
    pattern: Union[None, str] = None,
    *,
    clone: bool = False,
    device: str = None,
    dtype: torch.dtype = None,
    name: str = None,
) -> "StateDict":
    """Create a StateDict from a raw state dict.

    Args:
        raw_state_dict (Dict[str, torch.Tensor]): raw state dict
        device (str, optional): device. Defaults to None.
        dtype (torch.dtype, optional): dtype. Defaults to None.
        copy (bool, optional): copy and detach tensors. Defaults to False.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict
    """
    assert (device is None and dtype is None) or clone, "device and dtype can only be set when clone is True"
    state = StateDict()
    pattern = Pattern(pattern)

    for k, v in raw_state_dict.items():
        if pattern.match(k):
            state[k] = v
    if clone:
        state = state.clone(device=device, dtype=dtype)
    state.set_name(name)
    return state


def from_weight(
    model: nn.Module,
    pattern: Union[None, str] = None,
    *,
    requires_grad: bool = False,
    clone: bool = False,
    device: str = None,
    dtype: torch.dtype = None,
    name: str = None,
) -> "StateDict":
    """Create a StateDict from a model.

    Args:
        model (nn.Module): model
        requires_grad (bool, optional): keep track of gradient-needed ones. Defaults to False.
        device (str, optional): device. Defaults to None.
        dtype (torch.dtype, optional): dtype. Defaults to None.
        copy (bool, optional): copy and detach tensors. Defaults to False.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict of weights
    """
    assert (device is None and dtype is None) or clone, "device and dtype can only be set when clone is True"
    state = StateDict()
    pattern = Pattern(pattern)

    for k, param in model.named_parameters():
        if (param.requires_grad or not requires_grad) and pattern.match(k):
            state[k] = param

    if clone:
        state = state.clone(device=device, dtype=dtype)
    if name is None:
        name = f"{model.__class__.__name__}_weights"
    state.set_name(name)
    return state


def from_grad(
    model: nn.Module,
    pattern: Union[None, str] = None,
    *,
    clone: bool = False,
    device: str = None,
    dtype: torch.dtype = None,
    name: str = None,
) -> "StateDict":
    """Create a StateDict from a model. gradient is different.

    Args:
        model (nn.Module): model
        device (str, optional): device. Defaults to None.
        dtype (torch.dtype, optional): dtype. Defaults to None.
        copy (bool, optional): copy and detach tensors. Defaults to False.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict of gradients
    """
    assert (device is None and dtype is None) or clone, "device and dtype can only be set when clone is True"
    state = StateDict()
    pattern = Pattern(pattern)

    for k, param in model.named_parameters():
        if param.grad is not None and pattern.match(k):
            state[k] = param.grad

    if clone:
        state = state.clone(device=device, dtype=dtype)
    if name is None:
        name = f"{model.__class__.__name__}_grad"
    state.set_name(name)
    return state


def from_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_key: str = None,
    pattern: Union[None, str] = None,
    *,
    clone: bool = False,
    device: str = None,
    dtype: torch.dtype = None,
    name: str = None,
) -> "StateDict":
    """Create a StateDict from a model and an optimizer.

    Args:
        model (nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        optimizer_key (str, optional): k of optimizer state. Defaults to None.
        device (str, optional): device. Defaults to None.
        dtype (torch.dtype, optional): dtype. Defaults to None.
        copy (bool, optional): copy and detach tensors. Defaults to False.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict of optimizer state
    """
    assert (device is None and dtype is None) or clone, "device and dtype can only be set when clone is True"
    state = StateDict()
    pattern = Pattern(pattern)

    for k, param in model.named_parameters():
        if param in optimizer.state and pattern.match(k):
            state[k] = optimizer.state[param][optimizer_key]
    if clone:
        state = state.clone(device=device, dtype=dtype)
    if name is None:
        name = f"{model.__class__.__name__}_optimizer_state"
    state.set_name(name)
    return state


def from_together(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_keys: List[str] = None,
    pattern: Union[None, str] = None,
    *,
    clone: bool = False,
    device: str = None,
    dtype: torch.dtype = None,
) -> tuple:
    """Create weight, grad, optim status state dicts from a model and an optimizer.

    Args:
        model (nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        optimizer_keys (List[str], optional): keys of optimizer state. Defaults to None. For adam, ``["exp_avg", "exp_avg_sq"]`` can be used.

    Returns:
        tuple: weight state, grad state, optimizer states
    """
    weight_state = from_weight(model, pattern=pattern, clone=clone, device=device, dtype=dtype)
    grad_state = from_grad(model, pattern=pattern, clone=clone, device=device, dtype=dtype)
    optimizer_states = []
    for k in optimizer_keys:
        optimizer_states.append(
            from_optimizer_state(
                model, optimizer, optimizer_key=k, pattern=pattern, clone=clone, device=device, dtype=dtype
            )
        )
    return weight_state, grad_state, optimizer_states


def load(path: str, name: str = None) -> "StateDict":
    """Load a state dict from a file.

    Args:
        path (str): path to file
        device (str, optional): device. Defaults to None.
        dtype (torch.dtype, optional): dtype. Defaults to None.
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict
    """
    return StateDict(torch.load(path, map_location="cpu"), name=name)


# endregion: factory


class StateDict(OrderedDict):
    """
    ``StateDict`` is the main class of nnstat. ``StateDict`` is a subclass of ``OrderedDict``, which has the same structure as the return of ``model.state_dict()``. It provides many useful functions to analyze the state dict and supports basic math operations. All values will be detached automatically. For example:

    >>> import nnstat
    >>> from torchvision import models
    >>> state = nnstat.from_state_dict(models.resnet18().state_dict())
    # or in a recommended way
    >>> state = nnstat.from_weight(models.resnet18())
    >>> print(state)
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

    Math Functions
    ----------------

    The math functions can be categorized into two types. The first type applys element-wise operations to each tensor in the state dict. For example, to calculate adam update for analysis, we can do

    .. code-block:: python

        m = nnstat.from_optimizer_state(model, optimizer, optimizer_key="exp_avg")
        v = nnstat.from_optimizer_state(model, optimizer, optimizer_key="exp_avg_sq")
        update = m / (v.sqrt() + 1e-8)


    The second type applys reduction operations to the flattened state dict. These methods have a ``lw`` argument, which can be set to True to return a dict of results per layer. For example, to calculate the L1 norm of the state dict, we can do

    >>> state.norm1(p=2)
    tensor(113.1479)
    >>> state.norm1(p=2, lw=True)
    {'conv1.weight': tensor(2.4467), 'bn1.weight': tensor(8.), 'bn1.bias': tensor(0.), 'layer1.0.conv1.weight': tensor(11.2160), (...truncated)}
    # use regex to filter keys
    >>> state.norm1(p=2, lw=True, pattern=".*conv.*")
    {'conv1.weight': tensor(2.4467), 'layer1.0.conv1.weight': tensor(11.2160), 'layer1.0.conv2.weight': tensor(11.3411), (...truncate)}

    Visualization
    ----------------

    We provide three ways to effectively examine the state dict status. The first is to use ``describe`` method for debugging in terminal, which can display a summary of the state dict.

    >>> state.describe()
    [============================ Stats ResNet_weights ============================]
    name            shape         numel     norm1         norm1_mean  norm2
    0  ResNet_weights  (11689512,)  11689512  235774.53125  0.02017     113.108963
    >>> state.describe(lw=True, pattern=".*conv.*")
    [=========================== Stats ResNet_weights ===========================]
        name                   shape              numel    norm1        norm2
    0   conv1.weight              (64, 3, 7, 7)     9408    188.525146   2.438898
    1   layer1.0.conv1.weight    (64, 64, 3, 3)    36864   1736.147461  11.318304
    2   layer1.0.conv2.weight    (64, 64, 3, 3)    36864   1736.389282  11.336217
    3   layer1.1.conv1.weight    (64, 64, 3, 3)    36864   1733.171997  11.316651
    4   layer1.1.conv2.weight    (64, 64, 3, 3)    36864   1724.930786  11.236347
    (...truncated)

    The second way is to export the statistics for tools such as tensorboard and wandb. To export the statistics, do the following:

    >>> state.describe(display=False, include_keys=['norm1', 'norm2'])
    {'norm1': 235752.828125, 'norm2': 113.10881805419922}

    The third way is to visualize the statistics by plotting. The plotting API automatically plot and save the figures to :file:`_nnstat_cache` folder.

    .. code-block:: python

        # plot the histogram
        state.hist()
        # plot the ecdf for each layer whose name contains "conv"
        state.ecdf(lw=True, pattern=".*conv.*")
        # change the cahce directory
        nnstat.utils.set_cache_dir("my_cache_dir")
        # plot the heatmap for each layer whose name contains "conv"
        state.abs().heatmap(lw=True, pattern=".*conv.*")

    """

    # region: basic
    def __init__(
        self,
        *args,
        name: str = None,
        **kwargs,
    ):
        """Avoid calling class ``__init__`` directly. Instead, create StateDict using the factory functions such as :func:`from_state_dict`, :func:`from_weight`, :func:`from_grad`, :func:`from_optimizer_state`, :func:`from_together`.

        .. caution::

            Use ``copy=True`` to avoid modifying the original values.

            Use ``copy=True`` and ``device="cpu"`` to save GPU memory.

            Use ``copy=False`` to save memory and accelerate computing. When ``copy=False``, you :abbr:`cannot` set ``device`` and ``dtype``.

        Args:
            name (str, optional): The name of the state dict. Defaults to None.
            copy (bool, optional): Make a copy of the original tensors. This can avoid modifying original values. Defaults to False.
            device (str, optional): The device of the state dict. All tensors in state dict are assumed to be on the same device. All tensors and tensors added in the future will be moved to the device. Defaults to None.
            dtype (torch.dtype, optional): The dtype of the state dict. All tensors in state dict are assumed to be of the same dtype. All tensors and tensors added in the future will be casted to the dtype. Defaults to None.
        """
        self._name = name if name is not None else self.__class__.__name__
        self._flattened = None
        self.device, self.dtype, self._light_dict, self._is_tensor = None, None, None, None
        super().__init__(*args, **kwargs)

    def __setitem__(self, k: str, v):
        _is_tensor = isinstance(v, torch.Tensor)
        # some layers have only one parameter
        _light_dict = not _is_tensor or (v.numel() == 1 and self._light_dict is not False)
        assert self._light_dict is None or self._light_dict == _light_dict
        assert self._is_tensor is None or self._is_tensor == _is_tensor

        self._light_dict = _light_dict
        self._is_tensor = _is_tensor

        if self._is_tensor:
            if self.device is not None:
                v = v.to(device=self.device)
            self.device = v.device
            v = v.detach()
        super().__setitem__(k, v)

    def clone(self, device: str = None, dtype: torch.dtype = None) -> "StateDict":
        """Clone the state dict.

        Returns:
            StateDict: cloned state dict with the same device and dtype
        """
        assert self._is_tensor or (device is None and dtype is None)
        state_dict = StateDict(name=f"{self._name}_clone")
        for k in self:
            if self._is_tensor:
                state_dict[k] = self[k].detach().clone().to(dtype=dtype, device=device)
            else:
                state_dict[k] = self[k]
        return state_dict

    def to(self, device: str = None, dtype: torch.dtype = None) -> "StateDict":
        """Move the state dict to a device and cast to a dtype.

        .. warning::

            This method will modify the state dict inplace and force all future tensors to be on the same device and of the same dtype.

        Args:
            device (str, optional): device. None means the current device. Defaults to None.
            dtype (torch.dtype, optional): dtype. None means the current dtype. Defaults to None.

        Returns:
            StateDict: self
        """

        self.device = device
        self.dtype = dtype
        for k in self:
            self[k] = self[k].to(device=device, dtype=dtype)
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        keys = list(self.keys())

        if len(keys) == 0:
            return f"{self._name}[Empty]()"
        if self._light_dict:
            s = f"{self._name}\n(\n"
            for i, (k, v) in enumerate(self.items()):
                s += f"\t{i:0>2}: {k:<{len(k)}} = {v}\n"
            s += ")"
        else:
            s = f"{self._name}[L2={self.norm(2):.4g}, Numel={self.numel():,d}, device={self.device}, dtype={self.dtype}]\n"
            shapes = [str(tuple(v.shape)) for v in self.values()]
            max_key_len = max(len(k) for k in keys)
            max_shape_len = max(len(shape) for shape in shapes)

            s += "(\n"
            for i, (k, shape) in enumerate(zip(keys, shapes)):
                s += f"\t{i:0>2}: {k:<{max_key_len}} {shape:>{max_shape_len}}\n"
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

    def at(self, index: int) -> torch.Tensor:
        """Get the v at a given index.

        Args:
            index (int): index

        Returns:
            torch.Tensor: v
        """
        return self[list(self.keys())[index]]

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a state dict of type dict, which can be loaded by ``torch.load``.

        Returns:
            dict[str, torch.Tensor]: state dict
        """
        return {k: self[k] for k in self}

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

    def flatten(
        self,
        lazy=True,
        pattern: Union[None, str] = None,
    ) -> torch.Tensor:
        """Flatten the state dict into a single tensor.

        .. warning::

            For now, reduction functions on the whole state dict are implemented on the flattened tensor. This means that the flattened tensor will be computed every time you call a reduction function. If you want to use reduction functions on the whole state dict frequently, you can set ``lazy=False`` to cache the flattened tensor. This will increase the memory usage.

        Args:
            lazy (bool, optional): If True, return the cached flattened tensor. Defaults to True. This cannot be used together with pattern.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            torch.Tensor: flattened tensor
        """
        if pattern is not None or not self._is_tensor:
            lazy = False
        if lazy:
            if self._flattened is None:
                self._flattened = torch.cat([v.flatten() for v in self.values()])
            return self._flattened

        pattern = Pattern(pattern)
        if self._is_tensor:
            return torch.cat([v.flatten() for k, v in self.items() if pattern.match(k)])
        else:
            return [v for k, v in self.items() if pattern.match(k)]

    # endregion: basic-functions
    # region: math-update
    def apply(self, func: Callable[torch.Tensor, torch.Tensor]) -> "StateDict":
        """Apply a function to each tensor in state dict.

        Args:
            func (Callable[torch.Tensor, torch.Tensor]): function

        Returns:
            StateDict: state dict with function applied
        """
        return StateDict({k: func(v) for k, v in self.items()})

    def apply_(self, func: Callable[torch.Tensor, torch.Tensor]) -> "StateDict":
        """Inplace version of :meth:`apply`."""
        for k, v in self.items():
            self[k] = func(v)
        return self

    def filter(self, pattern: Union[None, str]) -> "StateDict":
        """Filter the state dict by a pattern.

        Args:
            pattern (Union[None, str]): pattern

        Returns:
            StateDict: filtered state dict
        """
        pattern = Pattern(pattern)
        return StateDict({k: v for k, v in self.items() if pattern.match(k)})

    def __neg__(self) -> "StateDict":
        return self.apply(lambda x: -x)

    def __add__(self, other: Union["StateDict", float, int]) -> "StateDict":
        if not isinstance(other, StateDict):
            return StateDict({k: v + other for k, v in self.items()})
        assert self.is_compatible(other)
        return StateDict({k: v + other[k] for k, v in self.items()})

    def __radd__(self, other: Union["StateDict", float, int]) -> "StateDict":
        return self + other

    def __sub__(self, other: Union["StateDict", float, int]) -> "StateDict":
        return self + (-other)

    def __rsub__(self, other: Union["StateDict", float, int]) -> "StateDict":
        return other + (-self)

    def __mul__(self, other: Union["StateDict", float, int]) -> "StateDict":
        if not isinstance(other, StateDict):
            return StateDict({k: v * other for k, v in self.items()})
        assert self.is_compatible(other)
        return StateDict({k: v * other[k] for k, v in self.items()})

    def __rmul__(self, other: Union["StateDict", float, int]) -> "StateDict":
        return self * other

    def __truediv__(self, other: Union["StateDict", float, int]) -> "StateDict":
        if not isinstance(other, StateDict):
            return StateDict({k: v / other for k, v in self.items()})
        assert self.is_compatible(other)
        return StateDict({k: v / other[k] for k, v in self.items()})

    def __rtruediv__(self, other: Union["StateDict", float, int]) -> "StateDict":
        if not isinstance(other, StateDict):
            return StateDict({k: other / v for k, v in self.items()})
        assert self.is_compatible(other)
        return StateDict({k: other[k] / v for k, v in self.items()})

    def __pow__(self, other: Union[float, int]) -> "StateDict":
        return StateDict({k: v**other for k, v in self.items()})

    def sqrt(self) -> "StateDict":
        """Compute the square root of the state dict.

        Returns:
            StateDict: square root element-wise
        """
        return self.apply(lambda x: x**0.5)

    def abs(self) -> "StateDict":
        """Compute the absolute v of the state dict.

        Returns:
            StateDict: absolute v element-wise
        """
        if self._is_tensor:
            return self.apply(torch.abs)
        else:
            return self.apply(abs)

    def abs_(self) -> "StateDict":
        """Inplace version of :meth:`abs`."""
        if self._is_tensor:
            return self.apply_(torch.abs)
        else:
            return self.apply_(abs)
        return self

    def zeros(self) -> "StateDict":
        """Return a state dict of zeros with the same keys and shapes as the input state dict.

        Returns:
            StateDict: state dict of zeros
        """
        if self._is_tensor:
            return self.apply(torch.zeros_like)
        else:
            return self.apply(lambda x: 0 if isinstance(x, int) else 0.0)

    def zeros_(self) -> "StateDict":
        """Inplace version of :meth:`zero`."""
        if self._is_tensor:
            return self.apply_(torch.zeros_like)
        else:
            return self.apply_(lambda x: 0 if isinstance(x, int) else 0.0)
        return self

    def ones(self) -> "StateDict":
        """Return a state dict of ones with the same keys and shapes as the input state dict.

        Returns:
            StateDict: state dict of ones
        """
        if self._is_tensor:
            return self.apply(torch.ones_like)
        else:
            return self.apply(lambda x: 1 if isinstance(x, int) else 1.0)

    def ones_(self) -> "StateDict":
        """Inplace version of :meth:`ones`."""
        if self._is_tensor:
            return self.apply_(torch.ones_like)
        else:
            return self.apply_(lambda x: 1 if isinstance(x, int) else 1.0)
        return self

    def sign(self) -> "StateDict":
        """Compute the sign of the state dict.

        Returns:
            StateDict: sign
        """
        if self._is_tensor:
            return self.apply(torch.sign)
        else:
            return self.apply(lambda x: 1.0 if x >= 0 else -1.0)

    def sign_(self) -> "StateDict":
        """Inplace version of :meth:`sign`."""
        if self._is_tensor:
            return self.apply_(torch.sign)
        else:
            return self.apply_(lambda x: 1.0 if x >= 0 else -1.0)
        return self

    def clip(self, min: float = None, max: float = None) -> "StateDict":
        """Clip the state dict inplace.

        Args:
            min (float, optional): min. Defaults to None.
            max (float, optional): max. Defaults to None.

        Returns:
            StateDict: clipped state dict
        """
        if self._is_tensor:
            return self.apply(lambda x: x.clip(min=min, max=max))
        else:
            return self.apply(lambda x: min if x < min else max if x > max else x)

    def clip_(self, min: float = None, max: float = None) -> "StateDict":
        """Inplace version of :meth:`clip`."""
        if self._is_tensor:
            return self.apply_(lambda x: x.clip(min=min, max=max))
        else:
            return self.apply_(lambda x: min if x < min else max if x > max else x)
        return self

    # endregion: basic-math
    # region: math-reduction
    def apply_reduction(
        self,
        func: Callable,
        *,
        lw: bool = False,
        pattern: Union[None, str] = None,
    ) -> Union[float, "StateDict"]:
        """Apply a reduction function to the state dict.

        Args:
            func (Callable): reduction function
            lw (bool, optional): If True, return a dict of results per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: result
        """
        if lw:
            pattern = Pattern(pattern)
            state_dict = StateDict(name=f"{self._name}_stat")
            for k, v in self.items():
                if pattern.match(k):
                    state_dict[k] = func(v)
            return state_dict
        return func(self.flatten(pattern=pattern))

    def register_reduction(self, name: str, func: Callable):
        """Register a reduction function to the state dict.

        Args:
            name (str): name of the function
            func (Callable): reduction function
        """
        assert name not in columns_group["all"]
        columns_group["all"].append(name)
        setattr(self, name, lambda **kwargs: self.apply_reduction(func, **kwargs))

    def max(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union[float, Dict]:
        """Compute the maximum of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of maximums per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: maximum
        """
        return self.apply_reduction(lambda x: x.max(**kwargs), lw=lw, pattern=pattern)

    def min(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union[float, Dict]:
        """Compute the minimum of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of minimums per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: minimum
        """
        return self.apply_reduction(lambda x: x.min(**kwargs), lw=lw, pattern=pattern)

    def no(self, *, lw: bool = False, pattern: Union[None, str] = None) -> Union[int, Dict]:
        """If lw is True, return a dict of number ids per layer. Otherwise, return the number of tensors.

        Args:
            lw (bool, optional): If True, return a dict of numbers per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[int, Dict]: number of tensors
        """
        pattern = Pattern(pattern)
        if lw:
            return {k: i for i, k in enumerate(self.keys()) if pattern.match(k)}
        return len(self)

    def name(self, *, lw: bool = False, pattern: Union[None, str] = None) -> Union[str, Dict]:
        """Get the name of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of names per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[str, Dict]: name
        """
        pattern = Pattern(pattern)
        if lw:
            return {k: k for k in self.keys() if pattern.match(k)}
        return self._name

    def numel(self, *, lw: bool = False, pattern: Union[None, str] = None) -> Union[int, Dict]:
        """Compute the number of elements of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of number of elements per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[int, Dict]: number of elements
        """
        return self.apply_reduction(lambda x: x.numel(), lw=lw, pattern=pattern)

    def shape(self, *, lw: bool = False, pattern: Union[None, str] = None) -> Union[tuple, Dict]:
        """Get the shape of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of shapes per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[tuple, Dict]: shape
        """
        return self.apply_reduction(lambda x: x.shape, lw=lw, pattern=pattern)

    def norm(self, p: float = 2, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs):
        """Compute the Lp norm of the state dict.

        Args:
            p (float, optional): p. Defaults to 2.
            lw (bool, optional): If True, return a dict of Lp norms per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: Lp norm
        """
        return self.apply_reduction(lambda x: x.norm(p, **kwargs), lw=lw, pattern=pattern)

    def norm1(self, **kwargs) -> Union[float, Dict]:
        """Compute the L1 norm of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of L1 norms per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: L1 norm
        """
        return self.norm(p=1, **kwargs)

    def norm2(self, **kwargs) -> Union[float, Dict]:
        """Compute the L2 norm of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of L2 norms per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, OrderedDict]: L2 norm
        """
        return self.norm(p=2, **kwargs)

    def sum(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union[float, Dict]:
        """Compute the sum of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of sums per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: sum
        """
        return self.apply_reduction(lambda x: x.sum(**kwargs), lw=lw, pattern=pattern)

    def norm1_mean(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union[float, Dict]:
        """Compute the mean of the L1 norm of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of means per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: mean of L1 norm
        """
        return self.apply_reduction(lambda x: x.norm(1, **kwargs) / x.numel(**kwargs), lw=lw, pattern=pattern)

    def mean(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union[float, Dict]:
        """Compute the mean of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of means per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: mean
        """
        return self.apply_reduction(lambda x: x.mean(**kwargs), lw=lw, pattern=pattern)

    def var(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union[float, Dict]:
        """Compute the variance of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of variances per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: variance
        """
        return self.apply_reduction(lambda x: x.var(**kwargs), lw=lw, pattern=pattern)

    def std(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union[float, Dict]:
        """Compute the standard deviation of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of standard deviations per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: standard deviation
        """
        return self.apply_reduction(lambda x: x.std(**kwargs), lw=lw, pattern=pattern)

    def skew(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union[float, Dict]:
        """Compute the skewness of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of skewnesses per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: skewness
        """
        return self.apply_reduction(lambda x: stats.skew(x, axis=None, **kwargs), lw=lw, pattern=pattern)

    def kurtosis(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union[float, Dict]:
        """Compute the kurtosis of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of kurtosis per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: kurtosis
        """
        return self.apply_reduction(lambda x: stats.kurtosis(x, axis=None, **kwargs), lw=lw, pattern=pattern)

    def abs_max(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union[float, Dict]:
        """Compute the absolute maximum of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of absolute maximums per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: absolute maximum
        """
        return self.apply_reduction(lambda x: x.abs().max(**kwargs), lw=lw, pattern=pattern)

    def abs_min(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union[float, Dict]:
        """Compute the absolute minimum of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of absolute minimums per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: absolute minimum
        """
        return self.apply_reduction(lambda x: x.abs().min(**kwargs), lw=lw, pattern=pattern)

    def value(self, *, lw: bool = False, pattern: Union[None, str] = None) -> Union[float, Dict]:
        """Get the value of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of values per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[float, Dict]: value
        """
        if not lw:
            return str(self)
        return self.apply_reduction(lambda x: x, lw=lw, pattern=pattern)

    # endregion: math

    def describe(
        self,
        pattern: Union[None, str] = None,
        group: str = None,
        lw: bool = False,
        display: bool = True,
        include_keys: Union[str, List[str]] = None,
        exlude_keys: Union[str, List[str]] = None,
        additional_info: Dict[str, torch.Tensor] = None,
    ) -> Union[None, Dict[str, Dict[str, float]]]:
        """Display a summary of the state dict. The pre-defined groups are ``all`` and ``default``.
        ``all``: ["name", "shape", "numel", "norm1", "norm1_mean", "norm2", "sum", "mean", "var", "std", "skew", "kurtosis", "max", "min", "abs_min", "abs_max"],
        ``default``: ["name", "shape", "numel", "norm1", "norm1_mean", "norm2"]

        Args:
            display (bool, optional): If True, display the summary and return None. Defaults to True.
            lw (bool, optional): If True, display lw stats. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.
            group (str, optional): Group of keys. Defaults to None. If None and no include_keys are provided,  use the ``default`` group.
            include_keys (Union[str, List[str]], optional): Additional keys to include. Defaults to None.
            exlude_keys (Union[str, List[str]], optional): Keys to exclude. Defaults to None.
            additional_info (Dict[str, torch.Tensor], optional): additional info to display. It should be like {"norm1": {"layer1": value1, "layer2": value2}, "norm2": {...}} Defaults to None.

        Returns:
            Union[None, Dict[str, Dict[str, float]]]: stats
        """
        if self._light_dict:
            group = "_light_dict"
        if group is None:
            columns = []
            if include_keys is None:
                columns = columns_group["default"]
        else:
            assert group in columns_group
            columns = columns_group[group]

        include_keys = utils.str2list(include_keys)
        exlude_keys = utils.str2list(exlude_keys)
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
            pattern_ = Pattern(pattern)
            ret = dict()
            if lw:
                for k in self.keys():
                    if pattern_.match(k):
                        ret[k] = dict()
        for k in columns:
            stat = getattr(self, k)(lw=lw, pattern=pattern)
            stat = utils.itemize(stat)
            if not lw:
                ret[k] = [stat] if display else stat
            else:
                if display:
                    ret[k] = stat.values()
                else:
                    for k, v in stat.items():
                        ret[k][k] = v

        if additional_info is not None:
            assert lw
            for k, result in additional_info.items():
                stat = utils.itemize(result)
                pattern = Pattern(pattern)
                stat = {k: v for k, v in stat.items() if pattern.match(k)}
                if display:
                    ret[k] = stat.values()
                else:
                    for k, v in stat.items():
                        ret[k][k] = v

        if display:
            ret_str = ret.to_string(
                formatters=dict(name=lambda x: f"{x:<{max([len(n) for n in ret.name])}}"), justify="left"
            )
            utils.print_title(f"Stats {self._name}", width=len(ret_str.split("\n")[0]))
            print(ret_str)
        else:
            return ret

    def hist(
        self,
        lw: bool = False,
        pattern: Union[None, str] = None,
        name: str = None,
        *,
        bins: int = 100,
        logx: bool = False,
        logy: bool = False,
        **kwargs,
    ):
        """Plot histogram of the state dict.

        Args:
            lw (bool, optional): If True, plot histograms per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.
            name (str, optional): name. Defaults to None.
            bins (int, optional): number of bins. Defaults to 100.
            logx (bool, optional): log scale of x axis. Defaults to False.
            logy (bool, optional): log scale of y axis. Defaults to False.
        """
        assert not self._light_dict, "Cannot plot histograms of light state dict."
        if name is None:
            if self._name is None:
                name = f"hist_{utils.get_time()}"
            else:
                name = f"{self._name}_hist_{utils.get_time()}"
        if not lw:
            logger.warning("Histogram of all parameters comsumes a lot of time.")
            values = self.flatten(pattern=pattern)
            utils.plot_hist(values, bins=bins, logx=logx, logy=logy, name=name, **kwargs)
        else:
            for k, v in self.items():
                utils.plot_hist(v, bins=bins, logx=logx, logy=logy, directory=name, name=k, **kwargs)

    def ecdf(
        self,
        lw: bool = False,
        pattern: Union[None, str] = None,
        name: str = None,
        *,
        logx: bool = False,
        logy: bool = False,
    ):
        """Plot Empirical Cumulative Distribution Function (ECDF) of the state dict.

        Args:
            lw (bool, optional): If True, plot ECDFs per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.
            name (str, optional): name. Defaults to None.
            logx (bool, optional): log scale of x axis. Defaults to False.
            logy (bool, optional): log scale of y axis. Defaults to False.
        """
        assert not self._light_dict, "Cannot plot ECDFs of light state dict."
        if name is None:
            if self._name is None:
                name = f"ecdf_{utils.get_time()}"
            else:
                name = f"{self._name}_ecdf_{utils.get_time()}"
        if not lw:
            logger.warning("ECDF of all parameters comsumes a lot of time.")
            values = self.flatten(pattern=pattern)
            utils.plot_ecdf(values, logx=logx, logy=logy, name=name)
        else:
            for k, v in self.items():
                utils.plot_ecdf(v, logx=logx, logy=logy, directory=name, name=k)

    def heatmap(
        self,
        lw: bool = False,
        pattern: Union[None, str] = None,
        name: str = None,
        *,
        vmin: float = None,
        vmax: float = None,
        log: bool = False,
        square: bool = False,
        **kwargs,
    ):
        """Plot heatmap of the state dict.

        Args:
            lw (bool, optional): If True, plot heatmaps per layer. Defaults to False.
            pattern (Union[str, List[str]], optional): pattern to filter keys. Defaults to None.
            name (str, optional): name. Defaults to None.
            vmin (float, optional): min v. Defaults to None.
            vmax (float, optional): max v. Defaults to None.
            log (bool, optional): log scale. Defaults to False.
            square (bool, optional): square shape. Defaults to True.
        """
        assert not self._light_dict, "Cannot plot heatmaps of light state dict."
        if name is None:
            if self._name is None:
                name = f"heatmap_{utils.get_time()}"
            else:
                name = f"{self._name}_heatmap_{utils.get_time()}"
        if not lw:
            logger.warning("Heatmap of all parameters comsumes a lot of time.")
            values = self.flatten(pattern=pattern)
            utils.plot_heatmap(values, vmin=vmin, vmax=vmax, log=log, square=square, name=name, **kwargs)
        else:
            for k, v in self.items():
                utils.plot_heatmap(v, vmin=vmin, vmax=vmax, log=log, square=square, directory=name, name=k, **kwargs)
