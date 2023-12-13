# DropGrad: A Simple Method for Regularization and Accelerated Optimization of Neural Networks

- [DropGrad: A Simple Method for Regularization and Accelerated Optimization of Neural Networks](#dropgrad-a-simple-method-for-regularization-and-accelerated-optimization-of-neural-networks)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Using pip](#using-pip)
    - [Using git](#using-git)
  - [Usage](#usage)
    - [Basic Usage](#basic-usage)
    - [Use with Learning Rate Schedulers](#use-with-learning-rate-schedulers)
    - [Varying `drop_rate` per `Parameter`](#varying-drop_rate-per-parameter)

DropGrad is a regularization method for neural networks that works by randomly (and independently) setting gradient values to zero before an optimization step. Similarly to Dropout, it has a single parameter, `drop_rate`, the probability of setting each parameter gradient to zero. In order to de-bias the remaining gradient values, they are divided by `1.0 - drop_rate`.

> To the best of my knowledge DropGrad is an original contribution. However, I have no plans of publishing a paper.
> If indeed, it is an original method, please feel free to publish a paper about DropGrad. If you do so, all I ask is
> that you mention me in your publication and cite this repository.

## Installation

The PyTorch implementation of DropGrad can be installed simply using pip or by cloning the current GitHub repo.

### Requirements

The only requirement for DropGrad is PyTorch. (Only versions of PyTorch >= 2.0 have been tested, although DropGrad should be compatible with any version of PyTorch)

### Using pip

To install using pip:

```bash
pip install dropgrad
```

### Using git

```bash
git clone https://github.com/dingo-actual/dropgrad.git
cd dropgrad
python -m build
pip install dist/dropgrad-0.1.0-py3-none-any.whl
```

## Usage

### Basic Usage

To use DropGrad in your neural network optimization, simply import the `DropGrad` class to wrap your optimizer.

```python
from dropgrad import DropGrad
```

Wrapping an optimizer is similar to using a learning rate scheduler:

```python
opt_unwrapped = Adam(net.parameters(), lr=1e-3)
opt = DropGrad(opt_unwrapped, drop_rate=0.1)
```

During training, the application of DropGrad is automatically handled by the wrapper. Simply call `.step()` on
the wrapped optimizer to apply DropGrad then `.zero_grad()` to reset the gradients.

```python
opt.step()
opt.zero_grad()
```

### Use with Learning Rate Schedulers

If you use a learning rate scheduler as well as DropGrad, simply pass the base optimizer to both the DropGrad
wrapper and the learning rate scheduler:

```python
opt_unwrapped = Adam(net.parameters(), lr=1e-3)
lr_scheduler = CosineAnnealingLR(opt_unwrapped, T_max=100)
opt = DropGrad(opt_unwrapped, drop_rate=0.1)
```

During the training loop, you call `.step()` on the DropGrad wrapper before calling `.step()` on the learning rate
scheduler, similarly to using an optimizer without DropGrad:

```python
for epoch_n in range(n_epochs):
    for x_batch, y_batch in dataloader:
        pred_batch = net(x_batch)
        loss = loss_fn(pred_batch, y_batch)

        loss.backward()

        opt.step()
        opt.zero_grad()

    lr_scheduler.step()
```

### Varying `drop_rate` per `Parameter`

DropGrad allows the user to set a different drop rate for each `Parameter` under optimization. To do this, simply
pass a dictionary mapping `Parameters` to drop rates to the `drop_rate` argument of the DropGrad wrapper. If a dictionary
is passed to DropGrad during initialization, all optimized `Parameter`s that are not present in that dictionary will have
the drop rate passed to the DropGrad wrapper at initialization (if `drop_rate=None` then drop grad simply won't be applied
to `Parameter`s that are not present in the dictionary).

The example below will apply a `drop_rate` of 0.1 to all optimized weights and a `drop_rate` of 0.01 to all optimized biases,
with no DropGrad applied to any other optimized `Parameter`s:

```python
drop_rate_weights = 0.1
drop_rate_biases = 0.01

params_weights = [p for name, p in net.named_parameters() if p.requires_grad and 'weight' in name]
params_biases = [p for name, p in net.named_parameters() if p.requires_grad and 'bias' in name]

param_drop_rates = {p: drop_rate_weights for p in params_weights}
param_drop_rates.update({p: drop_rate_biases for p in params_biases})

opt_unwrapped = Adam(net.parameters(), lr=1e-3)
opt = DropGrad(opt_unwrapped, drop_rate=None, params=param_drop_rates)
```
