# selfsne

**Self-Supervised Noise Embeddings** — A PyTorch Lightning library for learning
low-dimensional embeddings via contrastive divergence estimation.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)

## Installation

```bash
# Core package (CPU, no CUDA)
pip install -e '.[dev]'

# With optional extras
pip install -e '.[dev,data,plot,rich]'
```

See [docs/development/cuda.md](docs/development/cuda.md) for GPU installation.
