# NNstat

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
- 梯度相关性；最大学习率与梯度方向（学习率分配问题，如何 Diverge）；激活值稀疏性与 Adam 关系

## Advanced

The documentation of nnstat is still under construction. You can check the source code for more details.

### trust ratio

> Proposed in [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888).

The trust ratio is defined as $\frac{||W_{t}||\_2}{||W_t - W_{t-1}||_2}$. It can be used to analyze the layerwise update strength.

### noise scale

> Proposed in [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)

The noise scale is defined as $\frac{\text{tr}(H\Sigma)}{G^THG}$. The direct computation of noise scale is expensive. The paper proposes to use two different batch sizes to estimate the noise scale. In addition to this method, we also provide a curve-fitting method to estimate the noise scale. It can be used to analyze whether we need to scale the learning rate with batch size.
