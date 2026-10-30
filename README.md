# selfsne

**Self-Supervised Noise Embeddings** — A PyTorch Lightning library for learning
low-dimensional embeddings via noise contrastive estimation and divergence
minimisation.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)

## Overview

`selfsne` trains neural encoders to produce embeddings that preserve the
structure of data by contrasting positive pairs against negative samples drawn
from a noise distribution. The library provides:

- **`SelfSNE`** — A `LightningModule` that handles training, validation, and
  prediction. Supply a loss and at least one encoder; a missing context/target
  encoder shares the other.
- **`LikelihoodRatioEstimator`** — The central loss module. It computes
  positive and negative similarities via configurable kernels, estimates a log
  baseline, and forms logits as `similarity - log_baseline`. The loss is then
  computed using a chosen divergence.
- **10+ kernels** — euclidean/normal, student-t, Cauchy, inverse, Laplace,
  von Mises-Fisher (cosine), wrapped Cauchy, inner product, and their
  pairwise/precise variants.
- **12+ divergence functions** — KL, reverse KL, Jensen-Shannon, binary
  cross-entropy, Wasserstein, density ratio, Pearson chi-squared, Hellinger,
  Neymann, Le Cam, softened reverse KL, and the alpha-divergence family
  (`AlphaNCE`, `ForwardNCE`, `ReverseNCE`).
- **9 baseline estimators** — batch, batch conditional, momentum, reverse
  variants, parametric, constant, and embedding baselines. Composable additive
  and log-additive baselines are also available.
- **K-nearest neighbor sampling** — A FIFO queue for storing embeddings during
  training to generate negative samples from a memory bank.
- **Neural network building blocks** — attention mechanisms, temporal
  convolutions (TCN), positional embeddings, patch embeddings, gated MLPs,
  and pre-normalization wrappers (LayerNorm, RMSNorm, DyT).
- **Data utilities** — paired datasets, correlated Gaussian sampler, Lorenz
  attractor, MNIST loader, memory-mapped and HDF5 datasets.

## Installation

```bash
# Core package (CPU, no CUDA)
pip install -e '.[dev]'

# With optional extras
pip install -e '.[dev,data,plot,rich]'
```

See [docs/development/cuda.md](docs/development/cuda.md) for GPU installation.

## Quick start

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator

# Build the model: supply a loss and at least one encoder.
model = SelfSNE(
    loss=LikelihoodRatioEstimator(divergence="bce"),
    target_encoder=nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ),
)

# Prepare data: (context, target) tensor pairs.
observations = torch.stack([
    torch.sin(torch.linspace(0, 8 * torch.pi, 129)),
    torch.cos(torch.linspace(0, 8 * torch.pi, 129)),
    torch.sin(2 * torch.linspace(0, 8 * torch.pi, 129)),
    torch.cos(2 * torch.linspace(0, 8 * torch.pi, 129)),
], dim=1)
loader = DataLoader(
    TensorDataset(observations[:-1], observations[1:]),
    batch_size=32,
)

# Train with PyTorch Lightning.
import pytorch_lightning as pl
trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=10)
trainer.fit(model, loader)

# Predict embeddings.
batches = trainer.predict(model, loader)
embeddings = torch.cat([torch.from_numpy(b["target_embedding"]) for b in batches])
```

See `examples/quickstart.py` for a complete runnable script.

## Key classes

| Class | Module | Description |
|-------|--------|-------------|
| `SelfSNE` | `selfsne` | Main LightningModule |
| `LikelihoodRatioEstimator` | `selfsne.losses` | Loss with kernel/divergence/baseline |
| `NearestNeighborSampler` | `selfsne.neighbors` | KNN sampling with FIFO queue |
| `TransformerEncoder` | `selfsne.transformers` | Transformer with latent cross-attention |
| `MixturePrior` | `selfsne.prior` | Mixture density prior |
| `NormalPrior` | `selfsne.prior` | Normal distribution prior |

### Kernels

```python
from selfsne.kernels import ROWWISE_KERNELS, PAIRWISE_KERNELS
# ROWWISE_KERNELS: euclidean, normal, student_t, cauchy, inverse, laplace,
#   von_mises, cosine, wrapped_cauchy, inner_product
# PAIRWISE_KERNELS: same + bilinear + precise_* variants
```

### Divergences

```python
from selfsne.divergences import DIVERGENCES
# DIVERGENCES: dr, wasserstein, binary, bce, jsd, rkld, kld, pearson,
#   hellinger, jeffreys, le_cam, neymann, soft_rkld
```

### Baselines

```python
from selfsne.baselines import BASELINES
# BASELINES: batch, batch_conditional, momentum, reverse_batch,
#   reverse_batch_conditional, reverse_momentum, parametric, constant, embedding
```

## Development

See `docs/development/` for maintainer documentation on acceptance gates,
API contracts, and CUDA validation.

### Running tests

```bash
pip install -e '.[dev]'
pytest
```

### Building

```bash
pip install build
python -m build
```

## License

[Apache 2.0](LICENSE.md).
