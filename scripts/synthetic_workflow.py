"""Development-only synthetic context workflows; not a release API.

Run from a source checkout. Dataset generation and epoch sampling have independent
seeds so epoch-boundary resume reproduces uninterrupted training on one device.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.precision import MixedPrecision
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator
from selfsne.nn import (
    GatedMLP,
    InputNorm1d,
    PreDyT,
    SinusoidalPositionalEmbedding,
    init_linear,
)
from selfsne.transformers import TransformerEncoder

from paired_workflow import TrainingHealth, trainer_for, write_json


def amp_plugin(config):
    if config["precision"] != "16-mixed":
        return None
    device = "cuda" if config["accelerator"] == "gpu" else "cpu"
    # The default 65536 scale can consume a tiny validation run in overflow
    # calibration. Keep dynamic scaling and strict finite-update acceptance.
    return MixedPrecision(
        "16-mixed",
        device,
        scaler=torch.amp.GradScaler(
            device, init_scale=config.get("amp_initial_scale", 1024)
        ),
    )


class ContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.walk = config["example"] == "von-mises"
        self.chunk = config["chunk_size"] if self.walk else 1
        width = config["width"]
        self.norm = InputNorm1d(2 if self.walk else 3)
        self.project = (
            init_linear(nn.Conv1d(2, width, self.chunk, self.chunk))
            if self.walk
            else init_linear(nn.Linear(3, width))
        )
        self.position = (
            SinusoidalPositionalEmbedding(width) if self.walk else nn.Identity()
        )
        self.transformer = TransformerEncoder(
            width,
            config["depth"],
            4,
            2 if self.walk else 1,
            norm_type="layernorm" if self.walk else "dyt",
            expansion_factor=2,
        )

    def preprocess(self, positions):
        # Difference FIRST: padding absolute positions invents a jump to zero.
        steps = torch.diff(positions, dim=1)
        if steps.shape[1] < 1:
            raise ValueError("A trajectory needs at least two positions")
        return F.pad(steps, (0, 0, 0, (-steps.shape[1]) % self.chunk))

    def forward(self, x):
        if self.walk:
            x = self.preprocess(x)
        x = self.norm(x.transpose(1, 2))
        x = (
            self.project(x).transpose(1, 2)
            if self.walk
            else self.project(x.transpose(1, 2))
        )
        return self.transformer(self.position(x))


class TokenHead(nn.Module):
    def __init__(self, width, dimensions, token=0):
        super().__init__()
        self.token = token
        self.project = nn.Sequential(
            PreDyT(width, GatedMLP(width, 2)), init_linear(nn.Linear(width, dimensions))
        )

    def forward(self, x):
        return self.project(x[:, self.token])


def model_for(config):
    walk = config["example"] == "von-mises"
    return SelfSNE(
        context_encoder=ContextEncoder(config),
        context_head=TokenHead(config["width"], config["embedding_dim"]),
        baseline_head=TokenHead(config["width"], config["embedding_dim"], 1)
        if walk
        else None,
        loss=LikelihoodRatioEstimator(
            kernel="cauchy",
            divergence=config["divergence"],
            baseline="embedding" if walk else "parametric",
        ),
        learning_rate=config["learning_rate"],
        weight_decay=1e-5,
    )


def generate(config):
    rng = np.random.default_rng(config["seed"])
    count = config["samples"]
    if config["example"] == "swiss-roll":
        from sklearn.datasets import make_swiss_roll

        points, labels = make_swiss_roll(count, noise=0, random_state=config["seed"])
        return torch.tensor(points, dtype=torch.float32), labels
    mu = rng.uniform(-np.pi, np.pi, count)
    kappa = np.exp(rng.uniform(-1, 1, count))
    turns = rng.vonmises(
        mu[:, None], kappa[:, None], (count, config["trajectory_length"] - 1)
    )
    headings = turns.cumsum(1)
    steps = np.stack((np.cos(headings), np.sin(headings)), -1)
    positions = np.concatenate((np.zeros((count, 1, 2)), steps.cumsum(1)), 1)
    return torch.tensor(positions, dtype=torch.float32), np.column_stack((mu, kappa))


def circular_statistics(positions):
    """Sufficient sums of cos/sin of observed relative turns (fixed count).

    No fictitious turn relative to an assumed initial heading is included.
    """
    steps = torch.diff(positions.double(), dim=1)
    angles = torch.atan2(steps[..., 1], steps[..., 0])
    turns = torch.diff(angles, dim=1)
    return torch.stack((turns.cos().sum(1), turns.sin().sum(1)), -1).numpy()


def split_ids(config):
    ids = np.random.default_rng(config["seed"] + 1).permutation(config["samples"])
    cut = int(0.75 * len(ids))
    return np.sort(ids[:cut]), np.sort(ids[cut:])


class Batches(IterableDataset):
    def __init__(self, config, data, ids):
        self.config, self.data, self.ids = config, data, torch.as_tensor(ids)
        self.epoch = 0

    def __len__(self):
        c = self.config
        windows = (
            c["trajectory_length"] - c["context_length"] - c["target_length"] + 1
            if c["example"] == "von-mises"
            else 1
        )
        return len(self.ids) * windows // c["batch_size"]

    def __iter__(self):
        c = self.config
        rng = torch.Generator().manual_seed(c["seed"] + 1000 + self.epoch)
        windows = (
            c["trajectory_length"] - c["context_length"] - c["target_length"] + 1
            if c["example"] == "von-mises"
            else 1
        )
        # Arithmetic indexing avoids storing millions of Python window tuples.
        order = torch.randperm(len(self.ids) * windows, generator=rng)
        for start in range(0, len(self) * c["batch_size"], c["batch_size"]):
            selected = order[start : start + c["batch_size"]]
            rows = self.ids[selected // windows]
            if c["example"] == "swiss-roll":
                target = self.data[rows, None]
                noise = torch.randn(len(rows), c["views"], 3, generator=rng)
                yield target + noise * math.sqrt(3 * c["views"]), target
            else:
                offset = selected % windows
                indices = offset[:, None] + torch.arange(c["context_length"])[None]
                future = (
                    offset[:, None]
                    + c["context_length"]
                    + torch.arange(c["target_length"])[None]
                )
                yield (
                    self.data[rows[:, None], indices],
                    self.data[rows[:, None], future],
                )


class EpochSeed(pl.Callback):
    def __init__(self, dataset):
        self.dataset = dataset
        self.finite_steps = 0
        self.skipped_steps = 0
        self.overflow = False
        self.completed_epochs = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.dataset.epoch = trainer.current_epoch
        pl.seed_everything(
            self.dataset.config["seed"] + 2000 + trainer.current_epoch, workers=True
        )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["synthetic_config"] = self.dataset.config
        checkpoint["synthetic_epochs_completed"] = self.completed_epochs

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.completed_epochs = checkpoint["synthetic_epochs_completed"]

    def on_train_epoch_end(self, trainer, pl_module):
        self.completed_epochs = trainer.current_epoch + 1

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.overflow = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in pl_module.parameters()
        )
        if self.overflow:
            scaler = getattr(trainer.precision_plugin, "scaler", None)
            if scaler is None:
                raise ValueError("Nonfinite training gradient")
            self.scale_before = scaler.get_scale()
            self.before_skip = [p.detach().clone() for p in pl_module.parameters()]
        else:
            self.finite_steps += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if any(not torch.isfinite(p).all() for p in pl_module.parameters()):
            raise ValueError("Nonfinite model parameter")
        if self.overflow:
            scaler = trainer.precision_plugin.scaler
            if scaler.get_scale() >= self.scale_before or any(
                not torch.equal(p, before)
                for p, before in zip(pl_module.parameters(), self.before_skip)
            ):
                raise ValueError("AMP overflow was not safely skipped")
            self.skipped_steps += 1
            self.before_skip = None

    def on_train_end(self, trainer, pl_module):
        if self.finite_steps <= self.skipped_steps:
            raise ValueError(
                "Training did not complete enough finite optimizer steps: "
                f"finite={self.finite_steps}, skipped={self.skipped_steps}"
            )

    def diagnostics(self, trainer):
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        return {
            "finite_steps": self.finite_steps,
            "skipped_steps": self.skipped_steps,
            "amp_scale": scaler.get_scale() if scaler is not None else None,
        }


def export(model, config, data, run):
    """Stream embeddings in original row order, including the final short batch."""
    device = next(model.parameters()).device
    model.eval()
    shape = (len(data), config["embedding_dim"])
    destination = run / "embeddings.npy"
    staged = run / "embeddings.partial.npy"
    output = np.lib.format.open_memmap(staged, mode="w+", dtype=np.float32, shape=shape)
    try:
        with torch.inference_mode():
            for start in range(0, len(data), config["batch_size"]):
                values = data[start : start + config["batch_size"]].to(device)
                if config["example"] == "swiss-roll":
                    values = values[:, None]
                embeddings = (
                    model.target_head(model.target_encoder(values))
                    .float()
                    .cpu()
                    .numpy()
                )
                if not np.isfinite(embeddings).all():
                    raise ValueError("Nonfinite prediction")
                output[start : start + len(values)] = embeddings
        output.flush()
        del output
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def evaluate(config, data, labels, run):
    if config["example"] != "von-mises":
        return {"geometry_evaluated": False}
    from scipy.special import i0e, i1e
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    train, test = split_ids(config)
    embeddings = np.load(run / "embeddings.npy")
    sums = circular_statistics(data)
    np.save(run / "sufficient-statistics.npy", sums)
    mu, kappa = labels.T
    resultant = i1e(kappa) / i0e(kappa)
    population = resultant[:, None] * np.column_stack((np.cos(mu), np.sin(mu)))
    report = {
        "split": "trajectory",
        "training_trajectories": len(train),
        "heldout_trajectories": len(test),
    }
    for name, target in [
        ("sample_mean_cos_sin", sums / (data.shape[1] - 2)),
        ("population_mean_cos_sin", population),
    ]:
        decoder = LinearRegression().fit(embeddings[train], target[train])
        score = r2_score(
            target[test], decoder.predict(embeddings[test]), multioutput="raw_values"
        )
        report[name + "_r2"] = score.tolist()
    return report


def validate(config):
    initial_scale = config.get("amp_initial_scale", 1024)
    if not math.isfinite(initial_scale) or initial_scale <= 0:
        raise ValueError("amp_initial_scale must be positive and finite")
    for key in (
        "samples",
        "batch_size",
        "width",
        "depth",
        "embedding_dim",
        "views",
        "chunk_size",
    ):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["samples"] < 16 or config["batch_size"] < 2 or config["width"] % 4:
        raise ValueError(
            "Need samples >= 16, batch_size >= 2, and width divisible by four"
        )
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if config["example"] == "von-mises" and (
        min(config["context_length"], config["target_length"]) < 3
        or config["trajectory_length"]
        < config["context_length"] + config["target_length"]
    ):
        raise ValueError(
            "Need context/target lengths >= 3 and enough trajectory positions"
        )


def execute(args):
    run = args.run.resolve()
    resume = args.command == "resume"
    config = (
        json.loads((run / "config.json").read_text())
        if resume
        else {
            key: value
            for key, value in vars(args).items()
            if key not in ("run", "command", "epochs")
        }
    )
    validate(config)
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if resume and not (run / "last.ckpt").exists():
        raise ValueError("No completed epoch checkpoint")
    if resume:
        # Only load trusted checkpoints created locally by this workflow.
        saved = torch.load(run / "last.ckpt", map_location="cpu", weights_only=False)
        if saved.get("synthetic_config") != config:
            raise ValueError("Configuration changed since the checkpoint was saved")
        completed = saved["synthetic_epochs_completed"]
        if args.epochs <= completed:
            raise ValueError(
                "epochs must exceed completed epochs (total desired epochs)"
            )
    pl.seed_everything(config["seed"], workers=True)
    data, labels = generate(config)
    ids = (
        split_ids(config)[0]
        if config["example"] == "von-mises"
        else np.arange(len(data))
    )
    batches = Batches(config, data, ids)
    if len(batches) < 1:
        raise ValueError("batch_size exceeds available training samples")
    health = TrainingHealth()
    epoch_health = EpochSeed(batches)
    trainer = trainer_for(
        config,
        precision_plugin=amp_plugin(config),
        max_epochs=args.epochs,
        callbacks=[
            ModelCheckpoint(dirpath=run, save_last=True, save_top_k=0),
            epoch_health,
            health,
        ],
    )
    model = model_for(config)
    if not resume:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "config.json", config)
    write_json(
        run / "status.json", {"status": "running", "requested_epochs": args.epochs}
    )
    try:
        trainer.fit(
            model,
            DataLoader(batches, batch_size=None, num_workers=0),
            ckpt_path=str(run / "last.ckpt") if resume else None,
        )
        export(model, config, data, run)
        np.save(run / "labels.npy", labels)
        write_json(run / "evaluation.json", evaluate(config, data, labels, run))
        write_json(
            run / "status.json",
            {
                "status": "passed",
                "requested_epochs": args.epochs,
                "last_loss": health.last_loss,
                "embedding_shape": [len(data), config["embedding_dim"]],
                "cuda_validated": config["accelerator"] == "gpu",
                "torch": torch.__version__,
                "training_health": epoch_health.diagnostics(trainer),
            },
        )
    except Exception:
        write_json(
            run / "status.json",
            {
                "status": "failed",
                "requested_epochs": args.epochs,
                "training_health": epoch_health.diagnostics(trainer),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--example", choices=["swiss-roll", "von-mises"], required=True)
    for name, default in [
        ("samples", 256),
        ("batch-size", 32),
        ("views", 10),
        ("width", 64),
        ("depth", 2),
        ("embedding-dim", 2),
        ("chunk-size", 16),
        ("trajectory-length", 129),
        ("context-length", 65),
        ("target-length", 17),
        ("seed", 42),
    ]:
        train.add_argument("--" + name, type=int, default=default)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--amp-initial-scale", type=float, default=1024)
    train.add_argument("--divergence", choices=["kld", "jsd", "rkld"], default="kld")
    train.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu")
    train.add_argument(
        "--precision", choices=["32-true", "16-mixed", "bf16-mixed"], default="32-true"
    )
    resume = sub.add_parser("resume")
    for command in (train, resume):
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--epochs", type=int, default=1)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
