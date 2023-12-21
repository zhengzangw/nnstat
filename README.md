# NNstat

A toolkit to track and analyze the status and statistics of neural network.

- Tracking: easily compute the status and statistics of neural network and can be recorded by Tensorboard, Wandb, etc.
- Analyze: easily analyze and visualize the status of neural networks.

Check the [Documentation](https://zhengzangw.github.io/nnstat/) and [Getting Started](https://zhengzangw.github.io/nnstat/tutorials/getting_started.html) for more details.

## Todo

We will add more features (*things we want to revisit for neural networks*) in the future. The features will be added either as an API in `NNstat`, or as a tutorial in document that can be easily implemented based on basic APIs.

- [ ] value-order plot
- [ ] interpolation
- [ ] changes in norm and value
- [ ] correlation
- [ ] activation-related
- [ ] update strength related
- [ ] Initialization related
- [ ] Frequency related
- [ ] gradient relevance
- Hessian related computation
  - [ ] Hessian heatmap
  - [ ] Hessian eigenvalue & eigenvector efficiently
  - [ ] Conditioning number
  - [ ] Gauss-Newton decomposition, Fisher information matrix, varaince of gradient, Hessian approximation
  - [ ] CA-sharpness
  - [ ] directional sharpness
  - [ ] top eigen subspace percentage & overlap
  - [ ] gradient variance
