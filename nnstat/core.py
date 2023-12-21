"""
See :doc:`/tutorials/getting_started` for basic usage.
"""

import logging
import re
from collections import OrderedDict
from typing import Callable, Dict, List, Union, Any

import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from . import utils

__all__ = [
    "from_state_dict",
    "from_weight",
    "from_grad",
    "from_optimizer_state",
    "from_together",
    "from_activation",
    "from_activation_grad",
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
    _light=[
        "no",
        "name",
        "value",
    ],
    custom=[],
)


class Pattern:
    def __init__(self, pattern: Union[None, str]):
        if pattern is None:
            self.pattern = None
        else:
            self.pattern = re.compile(pattern)

    def match(self, s: str) -> bool:
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

    >>> state = nnstat.from_state_dict(model.state_dict())

    Args:
        raw_state_dict (Dict[str, torch.Tensor]): raw state dict
        pattern (Union[None, str], optional): regex pattern to filter keys. Only keys that match the pattern will be kept. Defaults to None, which means no filtering.
        clone (bool, optional): ``clone=False`` means that the tensors in returned state dict shares the same memory with the one in the model except being detached. Use ``clone=True`` to make a snapshot of the state dict. Use ``device="cpu"`` to save GPU memory. When ``copy=False``, you :abbr:`cannot` set ``device`` and ``dtype``.. Defaults to False.
        device (str, optional): device used for moving. Defaults to None, which means no moving. This cannot be used together with ``clone=False``.
        dtype (torch.dtype, optional): dtype used for casting. Defaults to None, which means no casting. This cannot be used together with ``clone=False``.
        name (str, optional): name of the state dict. Defaults to None.

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
    """Create a StateDict containing weights (parameters) from PyTorch a model.

    >>> state = nnstat.from_weight(model)

    Args:
        model (nn.Module): model
        pattern (Union[None, str], optional): regex pattern to filter keys. Only keys that match the pattern will be kept. Defaults to None, which means no filtering.
        requires_grad (bool, optional): only keep parameters with ``requires_grad=True``. Defaults to False.
        clone (bool, optional): ``clone=False`` means that the tensors in returned state dict shares the same memory with the one in the model except being detached. Use ``clone=True`` to make a snapshot of the state dict. Use ``device="cpu"`` to save GPU memory. When ``copy=False``, you :abbr:`cannot` set ``device`` and ``dtype``.. Defaults to False.
        device (str, optional): device used for moving. Defaults to None, which means no moving. This cannot be used together with ``clone=False``.
        dtype (torch.dtype, optional): dtype used for casting. Defaults to None, which means no casting. This cannot be used together with ``clone=False``.
        name (str, optional): name of the state dict. Defaults to None.

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
    """Create a StateDict containing gradients from PyTorch a model.

    >>> loss.backward()
    >>> state = nnstat.from_grad(model)

    Args:
        model (nn.Module): model
        pattern (Union[None, str], optional): regex pattern to filter keys. Only keys that match the pattern will be kept. Defaults to None, which means no filtering.
        clone (bool, optional): ``clone=False`` means that the tensors in returned state dict shares the same memory with the one in the model except being detached. Use ``clone=True`` to make a snapshot of the state dict. Use ``device="cpu"`` to save GPU memory. When ``copy=False``, you :abbr:`cannot` set ``device`` and ``dtype``.. Defaults to False.
        device (str, optional): device used for moving. Defaults to None, which means no moving. This cannot be used together with ``clone=False``.
        dtype (torch.dtype, optional): dtype used for casting. Defaults to None, which means no casting. This cannot be used together with ``clone=False``.
        name (str, optional): name of the state dict. Defaults to None.

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
    """Create a StateDict containing optimizer states from a model and an optimizer.

    >>> state = nnstat.from_optimizer_state(model, optimizer, optimizer_key="exp_avg")

    Args:
        model (nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        optimizer_key (str, optional): key of optimizer state. Defaults to None. For adam, ``"exp_avg"`` can be used.
        pattern (Union[None, str], optional): regex pattern to filter keys. Only keys that match the pattern will be kept. Defaults to None, which means no filtering.
        clone (bool, optional): ``clone=False`` means that the tensors in returned state dict shares the same memory with the one in the model except being detached. Use ``clone=True`` to make a snapshot of the state dict. Use ``device="cpu"`` to save GPU memory. When ``copy=False``, you :abbr:`cannot` set ``device`` and ``dtype``.. Defaults to False.
        device (str, optional): device used for moving. Defaults to None, which means no moving. This cannot be used together with ``clone=False``.
        dtype (torch.dtype, optional): dtype used for casting. Defaults to None, which means no casting. This cannot be used together with ``clone=False``.
        name (str, optional): name of the state dict. Defaults to None.

    Returns:
        StateDict: state dict of optimizer states
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
) -> tuple["StateDict", "StateDict", List["StateDict"]]:
    """Create a StateDict containing weights, gradients, and optimizer states from a model and an optimizer.

    >>> state = nnstat.from_together(model, optimizer, optimizer_keys=["exp_avg", "exp_avg_sq"])

    Args:
        model (nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        optimizer_keys (List[str], optional): keys of optimizer states. Defaults to None. For adam, ``["exp_avg", "exp_avg_sq"]`` can be used.
        pattern (Union[None, str], optional): regex pattern to filter keys. Only keys that match the pattern will be kept. Defaults to None, which means no filtering.
        clone (bool, optional): ``clone=False`` means that the tensors in returned state dict shares the same memory with the one in the model except being detached. Use ``clone=True`` to make a snapshot of the state dict. Use ``device="cpu"`` to save GPU memory. When ``copy=False``, you :abbr:`cannot` set ``device`` and ``dtype``.. Defaults to False.
        device (str, optional): device used for moving. Defaults to None, which means no moving. This cannot be used together with ``clone=False``.
        dtype (torch.dtype, optional): dtype used for casting. Defaults to None, which means no casting. This cannot be used together with ``clone=False``.

    Returns:
        tuple[StateDict, StateDict, List[StateDict]]: state dict of weights, gradients, and optimizer states
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
        name (str, optional): name. Defaults to None.

    Returns:
        StateDict: state dict
    """
    return StateDict(torch.load(path, map_location="cpu"), name=name)


class ForwardHook:
    def __init__(self, model: nn.Module, pattern: Union[None, str] = None):
        self.pattern = Pattern(pattern)
        self.model = model
        self.hooks = {}
        self.state_dict = StateDict()

        for name, module in model.named_modules():
            if self.pattern.match(name):
                self.hooks[name] = module.register_forward_hook(self._get_hook_fn(name))

    def _get_hook_fn(self, name: str) -> Callable:
        def _hook_fn(module, input, output):
            self.state_dict[name] = output.detach()

        return _hook_fn

    def __del__(self):
        for hook in self.hooks.values():
            hook.remove()


def from_activation(model: nn.Module, pattern: Union[None, str] = None) -> ForwardHook:
    """Register forward hooks to a model. The state dict of activations will be stored in the ``state_dict`` attribute of the returned object. After a forward pass, the state dict will be updated.

    >>> state = nnstat.from_activation(model) # state is empty
    >>> model(x) # forward pass
    >>> print(state) # state is updated

    Args:
        model (nn.Module): model
        pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

    Returns:
        ForwardHook: class for wrapping forward hooks and has a ``state_dict`` attribute
    """
    forward_hooks = ForwardHook(model, pattern=pattern)
    return forward_hooks


class BackwardHook:
    def __init__(self, model: nn.Module, pattern: Union[None, str] = None):
        self.pattern = Pattern(pattern)
        self.model = model
        self.hooks = {}
        self.state_dict = StateDict()

        for name, module in model.named_modules():
            if self.pattern.match(name):
                self.hooks[name] = module.register_full_backward_hook(self._get_hook_fn(name))

    def _get_hook_fn(self, name: str) -> Callable:
        def _hook_fn(module, grad_input, grad_output):
            self.state_dict[name] = grad_output[0].detach()

        return _hook_fn

    def __del__(self):
        for hook in self.hooks.values():
            hook.remove()


def from_activation_grad(model: nn.Module, pattern: Union[None, str] = None) -> "StateDict":
    """Register backward hooks to a model. The state dict of gradients of activations will be stored in the ``state_dict`` attribute of the returned object. After a backward pass, the state dict will be updated.

    >>> state = nnstat.from_activation_grad(model) # state is empty
    >>> loss.backward() # backward pass
    >>> print(state) # state is updated

    Args:
        model (nn.Module): model
        pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

    Returns:
        BackwardHook: class for wrapping backward hooks and has a ``state_dict`` attribute
    """
    backward_hooks = BackwardHook(model, pattern=pattern)
    return backward_hooks


# endregion: factory


class StateDict(OrderedDict):
    """
    ``StateDict`` is the main class of nnstat. The keys are supposed to be the names of PyTorch modules. The values can be (detached) tensors or other objects. It provides many useful functions to analyze the state dict and supports basic math operations. Methods of ``StateDict`` are divided into three categories: update, reduction, and visualization.

    * **Update functions** will return a state dict with the same keys and shapes as the input state dict. More details can be found in apply method :meth:`apply`. Examples include :meth:`abs`, :meth:`clip`, :meth:`sign`, etc.

    * **Reduction functions** will return a scalar or a dict of scalars. More details can be found in apply_reduction method :meth:`apply_reduction`. Examples include :meth:`norm`, :meth:`mean`, :meth:`var`, etc. All reduction functions support a ``lw`` argument, which means "layer-wise" applying reduction, and a ``pattern`` argument to filter keys. Otherwise, the reduction will be applied to the whole state dict.

    * **Visualization functions** will give a summary of the state dict. This includes :meth:`describe`, :meth:`hist`, :meth:`ecdf`, and :meth:`heatmap`.

    :ivar str _name: name of the state dict. This is used for display.
    :ivar bool _is_tensor: whether all values are tensors
    :ivar bool _light_dict: a light dict means that all values are scalars (or strings). A light dict is typically a result of reduction functions. Light dicts have different display styles.
    :ivar str device: device for moving tensors. If None, the current device is used.
    :ivar torch.dtype dtype: dtype for casting tensors. If None, the current dtype is used.
    :ivar Union[None, torch.Tensor, List] _flattened: holder for the flattened tensor. This is used for lazy evaluation.
    """

    # region: basic
    def __init__(
        self,
        *args,
        name: str = None,
        **kwargs,
    ):
        """``StateDict`` is a subclass of ``OrderedDict``. Avoid calling class ``__init__`` directly. Instead, create StateDict using the factory functions such as :func:`from_state_dict`, :func:`from_weight`, :func:`from_grad`, :func:`from_optimizer_state`, :func:`from_together`, :func:`from_activation`, and :func:`from_activation_grad`.

        Args:
            name (str, optional): name of the state dict. Defaults to None.
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
        """Make a clone of the state dict.

        Args:
            device (str, optional): device. None means the current device. Defaults to None.
            dtype (torch.dtype, optional): dtype. None means the current dtype. Defaults to None.

        Returns:
            StateDict: cloned state dict
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
        """Move the state dict to a device and cast to a dtype inplacely.

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

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
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

    def at(self, index: int) -> Union[torch.Tensor, Any]:
        """Get the value at a given index.

        Args:
            index (int): index

        Returns:
            Union[torch.Tensor, Any]: value at the index
        """
        return self[list(self.keys())[index]]

    def state_dict(self) -> Union[dict[str, torch.Tensor], dict[str, Any]]:
        """Return a state dict of type dict, which can be loaded by ``torch.load``.

        Returns:
            Union[dict[str, torch.Tensor], dict[str, Any]]: state dict of type ``dict``
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
        """Flatten the state dict.

        .. note::

            Reduction functions on the whole state dict are implemented on the flattened tensor. By default, the flattened tensor is cached. If you want to use a different pattern, you should set ``lazy=False``.

        Args:
            lazy (bool, optional): If True, return the cached flattened tensor. Defaults to True. This cannot be used together with pattern.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            torch.Tensor: If the value is a tensor, a flattened tensor will be returned. If the value is not a tensor, a list of values will be converted to a tensor and returned.
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
            return torch.tensor([v for k, v in self.items() if pattern.match(k)])

    # endregion: basic-functions
    # region: math-update
    def apply(self, func: Callable) -> "StateDict":
        """This method is used to construct update functions. The update method applys element-wise operations to each tensor in the state dict, and returns a new state dict with the same keys and shapes as the input state dict. For example, we can use the following code to compute the adam update:

        .. code-block:: python

            m = nnstat.from_optimizer_state(model, optimizer, optimizer_key="exp_avg")
            v = nnstat.from_optimizer_state(model, optimizer, optimizer_key="exp_avg_sq")
            update = m / (v.sqrt() + 1e-8)

        Args:
            func (Callable): function

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
            pattern (Union[None, str]): regex pattern to filter keys. Only keys that match the pattern will be kept. Defaults to None, which means no filtering.

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
        """element-wise square root"""
        return self.apply(lambda x: x**0.5)

    def sqrt_(self) -> "StateDict":
        """Inplace version of :meth:`sqrt`."""
        return self.apply_(lambda x: x**0.5)

    def abs(self) -> "StateDict":
        """element-wise absolute value"""
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
        """Return a state dict of zeros with the same keys and shapes as the input state dict."""
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
        """Return a state dict of ones with the same keys and shapes as the input state dict."""
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
        """Compute the sign of the state dict."""
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
            min (float, optional): min value. Defaults to None.
            max (float, optional): max value. Defaults to None.

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
    ) -> Union["StateDict", Any]:
        """Apply a reduction function to the state dict, or to each layer of the state dict. For example:

        >>> state = nnstat.from_weight(model) # a resnet model
        >>> state.norm(p=2)
        tensor(98.5264)
        >>> state.norm(p=2, lw=True)
        ResNet_weights_stat
        (
            01: bn1.weight = 2.280656576156616
            02: bn1.bias = 2.7802467346191406
            03: layer1.0.conv1.weight = 10.269349098205566
            04: layer1.0.bn1.weight = 2.8918473720550537
            05: layer1.0.bn1.bias = 1.6855857372283936
            (...truncate)
        )
        >>> state.norm(p=2, lw=True, pattern=".*conv.*")
        ResNet_weights_stat
        (
            00: conv1.weight = 12.579392433166504
            01: layer1.0.conv1.weight = 10.269349098205566
            02: layer1.0.conv2.weight = 8.680017471313477
            03: layer1.1.conv1.weight = 9.771225929260254
            04: layer1.1.conv2.weight = 8.446762084960938
            05: layer2.0.conv1.weight = 11.297880172729492
            (...truncate)
        )

        Args:
            func (Callable): reduction function
            lw (bool, optional): If True, return a ``StateDict`` of results per layer. Otherwise, return the result of the whole state dict. Defaults to False.
            pattern (Union[None, str], optional): regex pattern to filter keys. Only keys that match the pattern will be kept. Defaults to None, which means no filtering.

        Returns:
            Union[StateDict, Any]: If lw is True, return a ``StateDict`` of results per layer. Otherwise, return the result of the whole state dict.
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
        """All implemented reduction functions are registered in ``columns_group["all"]``. You can register your own reduction functions to ``columns_group["all"]`` by calling this method. See :meth:`describe` for more details.

        Args:
            name (str): name of the function
            func (Callable): reduction function
        """
        assert name not in columns_group["all"]
        columns_group["all"].append(name)
        setattr(self, name, lambda **kwargs: self.apply_reduction(func, **kwargs))

    def max(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union["StateDict", Any]:
        """Compute the maximum of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of maximums per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[StateDict, Any]: maximum
        """
        return self.apply_reduction(lambda x: x.max(**kwargs), lw=lw, pattern=pattern)

    def min(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union["StateDict", Any]:
        """Compute the minimum of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of minimums per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[StateDict, Any]: minimum
        """
        return self.apply_reduction(lambda x: x.min(**kwargs), lw=lw, pattern=pattern)

    def no(self, *, lw: bool = False, pattern: Union[None, str] = None) -> Union["StateDict", int]:
        """If lw is True, return a dict of number ids per layer. Otherwise, return the number of tensors.

        Args:
            lw (bool, optional): If True, return a dict of numbers per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[StateDict, int]: If ``lw`` is True, return the index number of each layer. Otherwise, return the number of layers.
        """
        pattern = Pattern(pattern)
        if lw:
            return {k: i for i, k in enumerate(self.keys()) if pattern.match(k)}
        return len(self)

    def name(self, *, lw: bool = False, pattern: Union[None, str] = None) -> Union["StateDict", str]:
        """Get the name of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of names per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[StateDict, str]: If ``lw`` is True, return the name of each layer. Otherwise, return the name of the state dict.
        """
        pattern = Pattern(pattern)
        if lw:
            return {k: k for k in self.keys() if pattern.match(k)}
        return self._name

    def numel(self, *, lw: bool = False, pattern: Union[None, str] = None) -> Union["StateDict", int]:
        """Compute the number of elements of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of number of elements per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[StateDict, int]: If ``lw`` is True, return the number of elements of each layer. Otherwise, return the number of elements of the state dict.
        """
        return self.apply_reduction(lambda x: x.numel(), lw=lw, pattern=pattern)

    def shape(self, *, lw: bool = False, pattern: Union[None, str] = None) -> Union["StateDict", Any]:
        """Get the shape of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of shapes per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[StateDict, Any]: If ``lw`` is True, return the shape of each layer. Otherwise, return the shape of the state dict.
        """
        return self.apply_reduction(lambda x: x.shape, lw=lw, pattern=pattern)

    def norm(
        self, p: float = 2, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs
    ) -> ["StateDict", float]:
        """Compute the Lp norm of the state dict.

        Args:
            p (float, optional): p. Defaults to 2.
            lw (bool, optional): If True, return a dict of Lp norms per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union[StateDict, float]: Lp norm
        """
        return self.apply_reduction(lambda x: x.norm(p, **kwargs), lw=lw, pattern=pattern)

    def norm1(self, **kwargs) -> ["StateDict", float]:
        """Compute the L1 norm of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of L1 norms per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: L1 norm
        """
        return self.norm(p=1, **kwargs)

    def norm2(self, **kwargs) -> Union["StateDict", float]:
        """Compute the L2 norm of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of L2 norms per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: L2 norm
        """
        return self.norm(p=2, **kwargs)

    def sum(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union["StateDict", float]:
        """Compute the sum of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of sums per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: sum
        """
        return self.apply_reduction(lambda x: x.sum(**kwargs), lw=lw, pattern=pattern)

    def norm1_mean(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union["StateDict", float]:
        """Compute the mean of the L1 norm of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of means per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: mean of L1 norm
        """
        return self.apply_reduction(lambda x: x.norm(1, **kwargs) / x.numel(**kwargs), lw=lw, pattern=pattern)

    def mean(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union["StateDict", float]:
        """Compute the mean of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of means per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: mean
        """
        return self.apply_reduction(lambda x: x.mean(**kwargs), lw=lw, pattern=pattern)

    def var(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union["StateDict", float]:
        """Compute the variance of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of variances per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: variance
        """
        return self.apply_reduction(lambda x: x.var(**kwargs), lw=lw, pattern=pattern)

    def std(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union["StateDict", float]:
        """Compute the standard deviation of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of standard deviations per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: standard deviation
        """
        return self.apply_reduction(lambda x: x.std(**kwargs), lw=lw, pattern=pattern)

    def skew(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union["StateDict", float]:
        """Compute the skewness of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of skewnesses per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: skewness
        """
        return self.apply_reduction(lambda x: stats.skew(x, axis=None, **kwargs), lw=lw, pattern=pattern)

    def kurtosis(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union["StateDict", float]:
        """Compute the kurtosis of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of kurtosis per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: kurtosis
        """
        return self.apply_reduction(lambda x: stats.kurtosis(x, axis=None, **kwargs), lw=lw, pattern=pattern)

    def abs_max(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union["StateDict", float]:
        """Compute the absolute maximum of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of absolute maximums per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: absolute maximum
        """
        return self.apply_reduction(lambda x: x.abs().max(**kwargs), lw=lw, pattern=pattern)

    def abs_min(self, *, lw: bool = False, pattern: Union[None, str] = None, **kwargs) -> Union["StateDict", float]:
        """Compute the absolute minimum of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of absolute minimums per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: absolute minimum
        """
        return self.apply_reduction(lambda x: x.abs().min(**kwargs), lw=lw, pattern=pattern)

    def value(self, *, lw: bool = False, pattern: Union[None, str] = None) -> Union["StateDict", float]:
        """Get the value of the state dict.

        Args:
            lw (bool, optional): If True, return a dict of values per layer. Defaults to False.
            pattern (Union[None, str], optional): pattern to filter keys. Defaults to None.

        Returns:
            Union["StateDict", float]: value
        """
        if not lw:
            return str(self)
        return self.apply_reduction(lambda x: x, lw=lw, pattern=pattern)

    # endregion: math

    def describe(
        self,
        lw: bool = False,
        pattern: Union[None, str] = None,
        group: str = None,
        display: bool = True,
        include_keys: Union[str, List[str]] = None,
        exlude_keys: Union[str, List[str]] = None,
        additional_info: "StateDict" = None,
    ) -> Union[None, "StateDict"]:
        """This method collects statistics of the state dict. There are two ways to use this method. First, with ``display=True``, it will print the statistics of the state dict. This is useful for ``print`` or ``breakpoint()``. For example:

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

        The second way is to use ``display=False``. In this case, it will return a dict of statistics. This is useful for logging to 3rd party tools such as tensorboard and wandb. For example:

        >>> state_logs = state.describe(display=False, include_keys=['norm1', 'norm2'])
        >>> print(state_logs)
        {'norm1': 235752.828125, 'norm2': 113.10881805419922}
        >>> wandb.log({"iter": iter_num, **state_logs}, step=iter_num)


        The ``columns_group`` defines the groups of statistics. You can use ``group`` to select a group of statistics. The default groups are:

        .. code-block:: python

            {
                "all": ["name", "shape", "numel", "norm1", "norm1_mean", "norm2", "sum", "mean", "var", "std", "skew", "kurtosis", "max", "min", "abs_min", "abs_max"]
                "default": ["name", "shape", "numel", "norm1", "norm1_mean", "norm2"]
                "_light": ["name", "shape", "numel", "value"]
                "custom": []
            }

        Args:
            lw (bool, optional): If True, return a dict of statistics per layer. Defaults to False.
            pattern (Union[None, str], optional): regex pattern to filter keys. Defaults to None.
            group (str, optional): group name of statistics. The groups are defined in ``columns_group``. Defaults to None. If None, use ``columns_group["default"]``. If the state dict is light (i.e., not a dict of tensors), ``group`` will be ignored, and ``columns_group["_light"]`` will be used. See :meth:`register_reduction` for adding custom reduction functions.
            display (bool, optional): If True, print the statistics. Otherwise, return a dict of statistics. Defaults to True.
            include_keys (Union[str, List[str]], optional): keys to include. Keys must be in ``columns_group["all"]``. Defaults to None.
            exlude_keys (Union[str, List[str]], optional): keys to exclude. Defaults to None.
            additional_info (StateDict, optional): additional information to include. Defaults to None.

        Returns:
            Union[None, StateDict]: If ``display`` is False, return a dict of statistics. Otherwise, return None.
        """
        if self._light_dict:
            group = "_light"
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
        """Plot histogram of the state dict. The file will be saved to cache directory. By default, the cache directory is ``~/cache_nnstat``. You can change the cache directory by :meth:`set_cache_dir`. For example:

        .. code-block:: python

            # plot the histogram
            state.hist()
            # plot the ecdf for each layer whose name contains "conv"
            state.ecdf(lw=True, pattern=".*conv.*")
            # change the cahce directory
            nnstat.utils.set_cache_dir("my_cache_dir")
            # plot the heatmap for each layer whose name contains "conv"
            state.abs().heatmap(lw=True, pattern=".*conv.*")

        Args:
            lw (bool, optional): If True,  histograms per layer will be plotted and multiple files will be saved. Defaults to False.
            pattern (Union[None, str], optional): regex pattern to filter keys. Defaults to None.
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
        """Plot Empirical Cumulative Distribution Function (ECDF) of the state dict. The file will be saved to cache directory. By default, the cache directory is ``~/cache_nnstat``. You can change the cache directory by :meth:`set_cache_dir`.

        Args:
            lw (bool, optional): If True,  histograms per layer will be plotted and multiple files will be saved. Defaults to False.
            pattern (Union[None, str], optional): regex pattern to filter keys. Defaults to None.
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
        """Plot heatmap of the state dict. The file will be saved to cache directory. By default, the cache directory is ``~/cache_nnstat``. You can change the cache directory by :meth:`set_cache_dir`.

        Args:
            lw (bool, optional): If True,  histograms per layer will be plotted and multiple files will be saved. Defaults to False.
            pattern (Union[None, str], optional): regex pattern to filter keys. Defaults to None.
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
