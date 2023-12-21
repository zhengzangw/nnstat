.. _tutorials.getting_started:

Getting Started
===================

NNstat is a library for analyzing the status and statistics of neural networks. The basic element for NNstat is ``StateDict``. ``StateDict`` is a ``dict``-like object, the keys are the names of the parameters. The values can be ``torch.Tensor`` including weights, gradients, optimizer states, and activations. The values can also be ``float`` describing the statistics of the parameters. 



 All values will be detached automatically. For example:

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

    The second type applys reduction operations to the flattened state dict. These methods have a ``lw`` argument, which can be set to True to return a dict of results per layer. For example, to calculate the L1 norm of the state dict, we can do

    Visualization
    ----------------
    

    The second way is to export the statistics for tools such as tensorboard and wandb. To export the statistics, do the following:

    The third way is to visualize the statistics by plotting. The plotting API automatically plot and save the figures to :file:`_nnstat_cache` folder.

    
