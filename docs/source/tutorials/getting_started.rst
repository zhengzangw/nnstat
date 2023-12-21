.. _tutorials.getting_started:

Getting Started
============

NNstat is a library for analyzing the status and statistics of neural networks. The basic element for NNstat is ``StateDict``. ``StateDict`` is a ``dict``-like object, the keys are the names of the parameters. The values can be ``torch.Tensor`` including weights, gradients, optimizer states, and activations. The values can also be ``float`` describing the statistics of the parameters. 



