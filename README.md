# selfsne

Self-supervised embeddings in PyTorch. Learn a low-dimensional representation
from context–target pairs, then embed new observations with the trained encoder.

## Get started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/jgraving/selfsne.git
cd selfsne
uv sync
uv run python examples/quickstart.py
```

That's it. uv creates the environment and installs the dependencies. On Linux
and Windows it uses CPU PyTorch wheels by default; macOS uses the default
PyTorch distribution. Python 3.10 or newer is required.

The example creates a small synthetic recording, trains on adjacent
observations, and embeds 128 observations into two dimensions. No dataset
download, account, or GPU is needed.

### Prefer pip?

```bash
git clone https://github.com/jgraving/selfsne.git
cd selfsne
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python examples/quickstart.py
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell instead. The uv commands above do not require activation.

### GPU and optional features

For CUDA, install the PyTorch build appropriate for your machine. With uv,
`uv sync --no-sources` uses the default package index instead of this project's
CPU index. GPU execution also requires configuring the Lightning trainer.

| Extra | Includes |
| --- | --- |
| `data` | HDF5 and MNIST helpers |
| `plot` | Plotting and contour helpers |
| `rich` | Rich progress callbacks |
| `dev` | Tests, linting, and package builds |

```bash
uv sync --extra data --extra plot
# Or, in an activated pip environment:
python -m pip install -e '.[data,plot]'
```

## Train on your data

The model accepts tensors or batches containing `(context, target)` pairs.
For example, `observations` below is a float tensor with shape
`(n_observations, n_features)`:

```python
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator

observations = torch.randn(128, 4)  # Replace with your own float features.
pairs = TensorDataset(observations[:-1], observations[1:])
loader = DataLoader(pairs, batch_size=32, shuffle=True, drop_last=True)

model = SelfSNE(
    target_encoder=nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2)),
    loss=LikelihoodRatioEstimator(divergence="bce"),
)
trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=10, logger=False)
trainer.fit(model, loader)
```

The context and target encoders share weights when only one encoder is supplied.
Tokenizers default to identity mappings, so feature vectors work directly.
You can supply separate encoders, tokenizers, projection heads, and a baseline
head when needed.

Split recordings into training and evaluation sets **before** creating pairs
or windows. Construct pairs within each recording to avoid crossing recording
boundaries. Fit preprocessing on the training set only.

## Get embeddings

For a tensor already on the model's device:

```python
model.eval()
with torch.inference_mode():
    embeddings = model(observations)["target_embedding"]
```

For batched inference, Lightning handles devices and returns NumPy arrays:

```python
import numpy as np

prediction_loader = DataLoader(observations, batch_size=32, shuffle=False)
batches = trainer.predict(model, prediction_loader)
embeddings = np.concatenate([batch["target_embedding"] for batch in batches])
```

Keep `shuffle=False` during inference to preserve observation order. Ordinary
embedding prediction supports a final batch of one observation.

## Pairs, references, and scores

Training batches can also be dictionaries with `context`, `target`, and optional
`reference` tensors. Each reference tensor has shape
`(batch_size, number_of_references, ...)` before encoding.

Without explicit references, the loss uses other targets in the batch as
references and excludes the matching diagonal. Training and score prediction
therefore require at least two pairs per batch; use `drop_last=True` when the
last batch could contain one pair. Explicit references support single-pair batches.

Set `return_logits=True` to return scores from `trainer.predict`. Combine it
with `return_embeddings=False` to return scores alone. These are pairwise
contrastive scores, not normalized next-observation probabilities.

## Save and reload

```python
trainer.save_checkpoint("model.ckpt")
restored = SelfSNE.load_from_checkpoint(
    "model.ckpt",
    target_encoder=nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2)),
    loss=LikelihoodRatioEstimator(divergence="bce"),
)
restored.eval()
```

Supply the same architecture when loading: checkpoints store weights and
training state, while encoder, tokenizer, head, and loss modules are supplied
by your code. Only load checkpoints from trusted sources.

## Development

```bash
uv sync --extra dev --extra data --extra plot --extra rich
uv run pytest
uv run ruff check selfsne tests examples
uv run python -m build
```

`uv.lock` records the resolved environment. Use `uv sync --frozen` to install it
without changing the lockfile. pip installs remain supported with
`python -m pip install -e '.[dev,data,plot,rich]'`.

This is development software. Tests cover installation, losses, kernels,
baselines, batch validation, scheduling, data helpers, training, prediction,
and checkpoint restoration. Scientific performance depends on your data,
pairing strategy, and model configuration.

## License

[Apache 2.0](LICENSE.md).
