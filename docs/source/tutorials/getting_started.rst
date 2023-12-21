.. _tutorials.getting_started:

Getting Started
===================

``NNstat`` is a library for analyzing the status and statistics of neural networks. The basic element for NNstat is ``StateDict``. ``StateDict`` is a ``dict``-like object, the keys are the names of the parameters. The values can be ``torch.Tensor`` including weights, gradients, optimizer states, and activations. The values can also be ``float`` describing the statistics of the parameters.

Get a StateDict
-------------------

With ``NNstat``, it is easy to get the information of a neural network. For example, we can get the weights of a ResNet18 model.

>>> import nnstat
>>> from torchvision import models
>>> state = nnstat.from_state_dict(models.resnet18().state_dict())
# or in a recommended way
>>> state = nnstat.from_weight(models.resnet18())
>>> print(state)
ResNet_weights[L2=98.53, Numel=11,689,512, device=cpu, dtype=None]
(
        00: conv1.weight                    (64, 3, 7, 7)
        01: bn1.weight                              (64,)
        02: bn1.bias                                (64,)
        03: layer1.0.conv1.weight          (64, 64, 3, 3)
        04: layer1.0.bn1.weight                     (64,)
        05: layer1.0.bn1.bias                       (64,)
        (...truncated)
)

Get the Update
-------------------

Compute the Trust Ratio
----------------------------
