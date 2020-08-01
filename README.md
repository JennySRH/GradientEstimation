# PyTorch Gradient Estimators for Discrete Latent Variables
PyTorch implementation of gradient estimating methods in binary latent variables models (especially Sigmoid Belief Nets). Currently this implementation supports

- `multiple linear` layers of binary latent variables,
- `non-linear` layers of binary latent variables,
- `single-linear` layers of binary latent variables, and
- `auto-regressive` layers (not fully tested).

Note that these implementations are not well-optimized.

# Requirements

- [PyTorch 1.5.1](https://pytorch.org/)

# How to Run
The launcher script is available as `run_sbn.py`.

# Models
To train the Sigmoid Belief Networks (SBNs), we use variational inference methods, which adopt an inference network to approximate the true posterior. Since SBNs consist of discrete stochastic units, they are not directly compatible with automatic differentiation mechanism, due to which we resort to gradient estimating methods. Currently the following gradient estimators are implemented:

- [NVIL](<https://arxiv.org/abs/1402.0030>),
- [VIMCO](<https://arxiv.org/abs/1602.06725>),
- [ARM](https://arxiv.org/abs/1807.11143),
- [DisARM](https://arxiv.org/abs/2006.10680),
- [REINFORCE-LOO](https://openreview.net/pdf?id=r1lgTGL5DE) (REINFORCE estimators with leave-one-out baselines), and
- [RAM](http://proceedings.mlr.press/v70/tokui17a/tokui17a.pdf) (Reparameterization And Marginalization estimators).


# TODO list

- [ ] Gradient estimation for categorical latent variables;
- [ ] Adding [REBAR](https://arxiv.org/abs/1703.07370), [RELAX](https://arxiv.org/abs/1711.00123) models.

