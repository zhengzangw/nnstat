
NNstat documentation
====================
NNstat is a toolkit to track and analyze the status and statistics of neural network.

- Tracking: easily compute the status and statistics of neural network and can be recorded by Tensorboard, Wandb, etc.
- Analyze: easily analyze and visualize the status of neural networks.

Currently, we have tested ``NNstat`` in the following frameworks. We will continue to test and support more frameworks.

- `nanoGPT <https://github.com/karpathy/nanoGPT>`_: GPT-2

A list of previous optimizers can be found in `Awesome-Optimizer <https://github.com/zhengzangw/Awesome-Optimizer/tree/main>`_.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   Installation <tutorials/installation>
   Getting Started <tutorials/getting_started>

.. toctree::
   :caption: Python API
   :titlesonly:
   :maxdepth: 2

   nnstat <api/nnstat>
   nnstat.functional <api/functional>
   nnstat.accum <api/accum>
   nnstat.optim <api/optim>
   nnstat.utils <api/utils>

