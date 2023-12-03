import os

import torch
from transformers import BertModel, BertConfig

import nnstat

model = BertModel(BertConfig())
state_dict = nnstat.from_weight(model)


def test_save_load(tmpdir):
    file_name = os.path.join(tmpdir, "state.pth")
    state_dict.clone().save(file_name)
    loaded_state_dict = nnstat.load(file_name)
    assert state_dict.is_compatible(loaded_state_dict)
    assert (state_dict.at(0) == loaded_state_dict.at(0)).all()


def test_add():
    new_state_dict = state_dict + 1
    assert torch.allclose(new_state_dict.at(0), state_dict.at(0) + 1)
    new_state_dict = torch.tensor(1.0) + state_dict
    assert torch.allclose(new_state_dict.at(0), state_dict.at(0) + 1)
    new_state_dict = state_dict + state_dict
    new_state_dict_2 = 2 * state_dict
    assert torch.allclose(new_state_dict.at(0), new_state_dict_2.at(0))
    new_state_dict = state_dict - state_dict
    new_state_dict_2 = nnstat.zeros_like(state_dict)
    assert torch.allclose(new_state_dict.at(0), new_state_dict_2.at(0))
    new_state_dict = state_dict**0
    new_state_dict_2 = nnstat.ones_like(state_dict)
    assert torch.allclose(new_state_dict.at(0), new_state_dict_2.at(0))


def test_flatten():
    new_state_dict = state_dict.flatten()
    assert new_state_dict.numel() == state_dict.numel()


def test_mean():
    assert state_dict.mean() == state_dict.flatten().mean()
    assert state_dict.var() == state_dict.flatten().var()
    assert state_dict.std() == state_dict.flatten().std()
    assert state_dict.norm1() == state_dict.flatten().norm(1)


# state_dict.describe(group="default", layerwise=True)
