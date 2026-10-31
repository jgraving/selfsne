# User guide

Self-Supervised Noise Embeddings learn low-dimensional representations by contrasting noisy context views with clean targets.

## Quick start

```python
import torch
import pytorch_lightning as pl
from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator

model = SelfSNE(
    loss=LikelihoodRatioEstimator(divergence="bce"),
    target_encoder=torch.nn.Sequential(
        torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)
    )
)

# Data loader with adjacent observations as context/target pairs
# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
trainer.fit(model, train_dataloader)
```

See `examples/quickstart.py` for a complete runnable script.

## What it does

- Context views are generated from clean targets with controllable noise.
- `LikelihoodRatioEstimator` scores similarity via kernel + divergence.
- The encoder learns embeddings that preserve the likelihood ratio structure.

## Optional extras

- `pip install selfsne[plot]` for visualization utilities.
- `pip install selfsne[data]` for built-in datasets.

Run tests with `pytest` and build the package with `python -m build`.
