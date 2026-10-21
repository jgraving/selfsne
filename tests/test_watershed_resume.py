import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from selfsne.prior import MixturePrior


def test_resumed_watershed_matches_uninterrupted_optimization(tmp_path):
    torch.manual_seed(7)
    initial = MixturePrior(num_features=2, num_components=3, lr=0.1)

    def model():
        prior = MixturePrior(num_features=2, num_components=3, lr=0.1)
        prior.load_state_dict(initial.state_dict())
        return prior

    def train(prior, epochs, checkpoint=None):
        trainer = pl.Trainer(
            accelerator='cpu', devices=1, max_epochs=epochs,
            logger=False, enable_progress_bar=False,
            enable_model_summary=False, enable_checkpointing=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(prior, DataLoader(torch.zeros(2), batch_size=1), ckpt_path=checkpoint)
        return trainer

    continuous = model()
    train(continuous, 2)
    interrupted = model()
    trainer = train(interrupted, 1)
    checkpoint = tmp_path / 'watershed.ckpt'
    trainer.save_checkpoint(checkpoint)
    resumed = model()
    train(resumed, 2, checkpoint)

    assert not torch.equal(interrupted.watershed_locs, interrupted.locs)
    torch.testing.assert_close(resumed.watershed_locs, continuous.watershed_locs)
    torch.testing.assert_close(resumed.watershed_assignments, continuous.watershed_assignments)


def test_fresh_watershed_optimization_uses_current_component_means(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prior = MixturePrior(num_features=2, num_components=3)
    with torch.no_grad():
        prior.locs.add_(10)
    means = prior.locs.detach().clone()
    prior.optimize_watershed(max_epochs=1, steps_per_epoch=1, lr=0, verbose=False)
    torch.testing.assert_close(prior.watershed_locs, means)
    torch.testing.assert_close(prior.locs, means)
