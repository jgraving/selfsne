"""Minimal example showing SelfSNE on synthetic data."""
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator

def main():
    pl.seed_everything(0)
    # Simple 2D sinusoidal data
    t = torch.linspace(0, 4*torch.pi, 256)
    x = torch.stack([t.sin(), t.cos(), (2*t).sin(), (2*t).cos()], dim=1)

    ds = TensorDataset(x[:-1], x[1:])
    loader = DataLoader(ds, batch_size=64)

    model = SelfSNE(
        loss=LikelihoodRatioEstimator(divergence="bce"),
        target_encoder=torch.nn.Sequential(
            torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
        )
    )

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, loader)

    emb = trainer.predict(model, loader)
    embeddings = torch.cat([b["target_embedding"] for b in emb])
    print("Embeddings shape:", embeddings.shape)

if __name__ == "__main__":
    main()
