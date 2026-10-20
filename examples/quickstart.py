"""CPU training and embedding example; no dataset download required."""

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator


def make_model():
    return SelfSNE(
        loss=LikelihoodRatioEstimator(divergence="bce"),
        target_encoder=nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2)),
    )


def main():
    pl.seed_everything(42, workers=True)
    # Adjacent observations form context/target pairs within one synthetic recording.
    time = torch.linspace(0, 8 * torch.pi, 129)
    observations = torch.stack(
        [time.sin(), time.cos(), (2 * time).sin(), (2 * time).cos()], dim=1
    )
    loader = DataLoader(
        TensorDataset(observations[:-1], observations[1:]), batch_size=32
    )
    model = make_model()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)
    batches = trainer.predict(model, loader)
    embeddings = np.concatenate([batch["target_embedding"] for batch in batches])
    assert embeddings.shape == (128, 2)
    assert np.isfinite(embeddings).all()
    print(
        f"Embedded {len(embeddings)} observations into {embeddings.shape[1]} dimensions."
    )


if __name__ == "__main__":
    main()
