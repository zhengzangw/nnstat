import os

import torch
from torchvision import models

import nnstat

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
state_dict = nnstat.from_weight(model)
nst = nnstat.StateDict({"a": 1, "b": 2})


def test_data_type():
    assert state_dict.at(0).data_ptr() == model.conv1.weight.data_ptr()
    assert state_dict.at(0).dtype == model.conv1.weight.dtype
    assert state_dict.at(0).device == model.conv1.weight.device
    assert not state_dict.at(0).requires_grad

    assert nst["a"] == 1
    assert nst["b"] == 2
    assert nst.at(0) == 1
    assert nst.at(1) == 2
    assert not nst._is_tensor
    assert nst._light_dict
    assert nst.device is None
    assert nst.dtype is None


def test_clone():
    new_state_dict = state_dict.clone()
    assert new_state_dict.at(0).data_ptr() != state_dict.at(0).data_ptr()
    assert new_state_dict.at(0).dtype == state_dict.at(0).dtype
    assert new_state_dict.at(0).device == state_dict.at(0).device
    assert not new_state_dict.at(0).requires_grad

    new_nst = nst.clone()
    assert new_nst.at(0) == nst.at(0)


def test_print():
    print(state_dict)
    print(nst)
    print(nnstat.StateDict())


def test_save_load(tmpdir):
    file_name = os.path.join(tmpdir, "state.pth")
    state_dict.clone().save(file_name)
    loaded_state_dict = nnstat.load(file_name)
    assert state_dict.is_compatible(loaded_state_dict)
    assert (state_dict.at(0) == loaded_state_dict.at(0)).all()

    file_name = os.path.join(tmpdir, "nst.pth")
    nst.clone().save(file_name)
    loaded_nst = nnstat.load(file_name)
    assert nst.is_compatible(loaded_nst)
    assert nst["a"] == loaded_nst["a"]


def test_flatten():
    new_state_dict = state_dict.flatten()
    assert new_state_dict.numel() == state_dict.numel()

    state_dict.flatten(lazy=True)
    assert state_dict._flattened.numel() == state_dict.numel()

    new_nst = nst.flatten()
    assert len(new_nst) == len(nst)
    new_nst = nst.flatten(pattern="a")
    assert len(new_nst) == 1


def test_math_basic_tensor():
    assert torch.equal((state_dict * 2).at(0), (state_dict + state_dict).at(0))
    assert torch.equal((state_dict * 2).at(0), (2 * state_dict).at(0))
    assert torch.equal((state_dict + 1).at(0).sum(), state_dict.at(0).sum() + state_dict.at(0).numel())
    assert torch.equal((state_dict - state_dict).at(0), state_dict.zeros().at(0))
    assert torch.equal((state_dict**0).at(0), state_dict.ones().at(0))


def test_math_basic_float():
    assert (nst * 2).at(0) == (nst + nst).at(0)
    assert (nst * 2).at(0) == (2 * nst).at(0)
    assert (nst + 1).at(0) == nst.at(0) + 1
    assert (nst - nst).at(0) == nst.zeros().at(0)
    assert (nst**0).at(0) == nst.ones().at(0)


def test_filter():
    assert state_dict.filter(pattern=".*conv.*").numel() == len(state_dict.flatten(pattern=".*conv.*"))


def test_create():
    sd1 = nnstat.from_state_dict(model.state_dict())
    sd2 = nnstat.from_weight(model)
    assert torch.equal(sd1.at(0), sd2.at(0))


def test_describe_tensor():
    r1 = state_dict.describe(lw=True, pattern=".*conv.*", display=False, exlude_keys=["no"])
    r2 = nnstat.from_weight(model, pattern=".*conv.*").describe(lw=True, display=False, exlude_keys=["no"])
    assert str(r1) == str(r2)

    nst.describe(lw=True)
    state_dict.norm(lw=True).describe(lw=True)


def test_mean():
    assert state_dict.mean() == state_dict.flatten().mean()
    assert state_dict.var() == state_dict.flatten().var()
    assert state_dict.std() == state_dict.flatten().std()
    assert state_dict.norm1() == state_dict.flatten().norm(1)
