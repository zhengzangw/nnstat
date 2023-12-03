# nnstat

A toolkit to track and analyze the status and statistics of neural network.

- Tracking: easily compute the status and statistics of neural network and can be recorded by Tensorboard, Wandb, etc.
- Analyze: easily analyze and visualize the status of neural networks.

## Todo

- [ ] value-order plot
- [ ] interpolation
- Hessian related computation
  - [ ] Hessian heatmap
  - [ ] Hessian eigenvalue & eigenvector efficiently
  - [ ] Conditioning number
  - [ ] Gauss-Newton decomposition, Fisher information matrix, varaince of gradient, Hessian approximation
  - [ ] CA-sharpness
  - [ ] directional sharpness
  - [ ] top eigen subspace percentage & overlap
  - [ ] gradient variance
- [ ] changes in norm and value
- [ ] correlation
- [ ] activation-related
- [ ] update strength related
- [ ] Initialization related
- [ ] Frequency related

梯度相关性；最大学习率与梯度方向（学习率分配问题，如何 Diverge）；激活值稀疏性与 Adam 关系

## Optimizer TestBet

- [nanoGPT](https://github.com/karpathy/nanoGPT): GPT-2

## Install

Download the source code and install it by pip.

```bash
cd nnstat
pip install -e .
```

## Basic

The basic usage of nnstat is to track the status and statistics of neural network. You can use `get_weight`, `get_grad` and `get_optimizer_state` to get a `NetDict` of the corresponding information.

```python
W0 = nnstat.get_weight(model)
loss = model(X, Y)
loss.backward()
G = nnstat.get_grad(model)
optimizer.step()
model.zero_grad()
Mom = nnstat.get_optimizer_state(model, optimizer, "exp_avg")
W1 = nnstat.get_weight(model)
update = W1 - W0
```

The `NetDict` class provides many useful methods to compute the statistics of neural network.

```python
W0.norm(2) # get the l2 norm of W0
W0.describe(layerwise=False) # get a bunch of statistics of W0
W0.describe() # get a bunch of statistics of W0 for each layer
W0.describe(pattern="ln") # get a bunch of statistics of W0 for each layer whose name contains "ln"
nnstat.compute_stats(model, optimizer, ["exp_avg", "exp_avg_sg"]) # describe weight, grad, optimizer state together
```

The `NetDict` class also provides many visualization methods to visualize each layer. Figures are saved to `cache_nnstat` directory by default.

```python
W0.hist() # plot the histogram of W0
W0.ecdf(pattern="proj") # plot the ecdf of W0 for each layer whose name contains "proj"
W0.heatmap(op="abs", pattern="ln", log=True) # plot the heatmap of W0 for each layer whose name contains "ln"
```

To log the statistics of neural network, you can log the statistics to Tensorboard, Wandb, etc.

```python
wandb.log(W0.describe(display=False), step=step)
```

## Advanced

The documentation of nnstat is still under construction. You can check the source code for more details.

### trust ratio

> Proposed in [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888).

The trust ratio is defined as $\frac{||W_{t}||\_2}{||W_t - W_{t-1}||_2}$. It can be used to analyze the layerwise update strength.

### noise scale

> Proposed in [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)

The noise scale is defined as $\frac{\text{tr}(H\Sigma)}{G^THG}$. The direct computation of noise scale is expensive. The paper proposes to use two different batch sizes to estimate the noise scale. In addition to this method, we also provide a curve-fitting method to estimate the noise scale. It can be used to analyze whether we need to scale the learning rate with batch size.
