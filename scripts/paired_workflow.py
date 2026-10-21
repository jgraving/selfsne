"""Maintainer workflow for paired tabular .npy arrays; not a public release API."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import sys

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter, ModelCheckpoint
import torch
from torch import nn
from torch.utils.data import DataLoader

from selfsne import SelfSNE
from selfsne.data import NumpyDataLoader, PairedDataset
from selfsne.losses import LikelihoodRatioEstimator


def array_info(path):
    path = Path(path).resolve()
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{path.name}: expected a nonempty (rows, features) array")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{path.name}: expected real numeric data")
    chunk_rows = max(1, min(4096, 1024 * 1024 // array.shape[1]))
    for start in range(0, len(array), chunk_rows):
        with np.errstate(over="ignore", invalid="ignore"):
            chunk = np.asarray(array[start : start + chunk_rows], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise ValueError(
                f"{path.name}: values must be finite and representable as float32"
            )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return array, {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "shape": list(array.shape),
    }


def model_for(config):
    return SelfSNE(
        loss=LikelihoodRatioEstimator(divergence="bce"),
        target_encoder=nn.Sequential(
            nn.Linear(config["features"], config["width"]),
            nn.SELU(),
            nn.Linear(config["width"], config["embedding_dim"]),
        ),
        learning_rate=config["learning_rate"],
    )


class EmbeddingWriter(BasePredictionWriter):
    """Write batches to a staged .npy file instead of retaining all predictions."""

    def __init__(self, path, rows, dimensions):
        super().__init__(write_interval="batch")
        self.path, self.rows, self.dimensions = Path(path), rows, dimensions
        self.offset = 0
        self.mapping = None
        self.staged = None
        self.total = np.zeros(dimensions, dtype=np.float64)
        self.squares = self.total.copy()

    def on_predict_start(self, trainer, pl_module):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.path.parent, suffix=".partial", delete=False
        ) as handle:
            self.staged = Path(handle.name)
        self.mapping = np.lib.format.open_memmap(
            self.staged, mode="w+", dtype=np.float32, shape=(self.rows, self.dimensions)
        )

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        values = prediction["target_embedding"]
        if values.shape[1:] != (self.dimensions,) or not np.isfinite(values).all():
            raise ValueError("Invalid embedding output")
        end = self.offset + len(values)
        if end > self.rows:
            raise ValueError("Prediction produced too many rows")
        self.mapping[self.offset : end] = values
        values = values.astype(np.float64)
        self.total += values.sum(0)
        self.squares += np.square(values).sum(0)
        self.offset = end

    def on_predict_end(self, trainer, pl_module):
        if self.offset != self.rows:
            raise ValueError("Prediction did not produce every input row")
        self.mapping.flush()
        self.mapping = None
        os.replace(self.staged, self.path)

    def on_exception(self, trainer, pl_module, exception):
        self.mapping = None
        if self.staged is not None:
            self.staged.unlink(missing_ok=True)

    def standard_deviation(self):
        return np.sqrt(
            np.maximum(0, self.squares / self.rows - (self.total / self.rows) ** 2)
        ).tolist()


class TrainingHealth(pl.Callback):
    def __init__(self):
        self.last_loss = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"].detach()
        if not torch.isfinite(loss).all():
            raise ValueError("Training objective became nonfinite")
        self.last_loss = float(loss.cpu())


def trainer_for(config, precision_plugin=None, **kwargs):
    if config["accelerator"] == "gpu" and not torch.cuda.is_available():
        raise ValueError(
            "GPU requested but CUDA is unavailable; CPU fallback is disabled"
        )
    if config["accelerator"] == "cpu" and config["precision"] != "32-true":
        raise ValueError("This workflow requires 32-true precision on CPU")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return pl.Trainer(
        accelerator=config["accelerator"],
        devices=1,
        precision=config["precision"] if precision_plugin is None else None,
        plugins=[precision_plugin] if precision_plugin is not None else None,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        **kwargs,
    )


def write_json(path, values):
    staged = path.with_suffix(".json.partial")
    staged.write_text(json.dumps(values, indent=2) + "\n")
    os.replace(staged, path)


def peak_cpu_memory():
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def train(args):
    run = args.run.resolve()
    resume = args.command == "resume"
    if resume:
        config = json.loads((run / "config.json").read_text())
        checkpoint = run / "last.ckpt"
        if not checkpoint.exists():
            raise ValueError("No epoch checkpoint exists in this run")
        context, context_info = array_info(config["context"]["path"])
        target, target_info = array_info(config["target"]["path"])
        if context_info != config["context"] or target_info != config["target"]:
            raise ValueError("Training inputs changed since this run was created")
    else:
        context, context_info = array_info(args.context)
        target, target_info = array_info(args.target)
        if context.shape != target.shape or len(context) < 2:
            raise ValueError(
                "Paired arrays must have matching shapes and at least two rows"
            )
        config = {
            "context": context_info,
            "target": target_info,
            "features": context.shape[1],
            "width": args.width,
            "embedding_dim": args.embedding_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "accelerator": args.accelerator,
            "precision": args.precision,
        }
        checkpoint = None
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    pl.seed_everything(config["seed"], workers=True)
    model = model_for(config)
    health = TrainingHealth()
    callback = ModelCheckpoint(
        dirpath=run,
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
        enable_version_counter=False,
    )
    trainer = trainer_for(config, max_epochs=args.epochs, callbacks=[callback, health])
    if not resume:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "config.json", config)
    # Fixed order makes epoch-boundary recovery reproducible. Only singleton
    # training remainders are omitted; prediction always includes every row.
    pairs = PairedDataset(
        NumpyDataLoader(context).dataset, NumpyDataLoader(target).dataset
    )
    drop_last = len(context) % config["batch_size"] == 1
    loader = DataLoader(
        pairs, batch_size=config["batch_size"], shuffle=False, drop_last=drop_last
    )
    report = {
        "status": "running",
        "command": args.command,
        "rows": len(context),
        "training_rows_per_epoch": len(context) - (1 if drop_last else 0),
        "epochs_requested": args.epochs,
        "torch": torch.__version__,
        "lightning": pl.__version__,
        "numpy": np.__version__,
        "accelerator": config["accelerator"],
        "precision": config["precision"],
    }
    write_json(run / "run.json", report)
    try:
        if config["accelerator"] == "gpu":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        trainer.fit(model, loader, ckpt_path=checkpoint)
        if trainer.interrupted:
            raise KeyboardInterrupt("Resume from the last completed epoch")
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Model parameters became nonfinite")
        trainer.save_checkpoint(run / "last.ckpt")
        report.update(
            fit_seconds=time.perf_counter() - start,
            epochs_completed=trainer.current_epoch,
            last_training_loss=health.last_loss,
        )
        start = time.perf_counter()
        writer = predict_model(model, config, target, run / "target-embeddings.npy")
        report.update(
            status="passed",
            predict_seconds=time.perf_counter() - start,
            embedding_shape=[len(target), config["embedding_dim"]],
            embedding_std=writer.standard_deviation(),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()
            if config["accelerator"] == "gpu"
            else None,
        )
    except BaseException as exc:
        report.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc),
        )
        raise
    finally:
        report["peak_process_rss_bytes"] = peak_cpu_memory()
        write_json(run / "run.json", report)
    print(json.dumps(report, indent=2))


def predict_model(model, config, array, output):
    writer = EmbeddingWriter(output, len(array), config["embedding_dim"])
    trainer = trainer_for(config, enable_checkpointing=False, callbacks=[writer])
    trainer.predict(
        model,
        NumpyDataLoader(array, batch_size=config["batch_size"], shuffle=False),
        return_predictions=False,
    )
    return writer


def predict(args):
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    array, _ = array_info(args.input)
    if array.shape[1] != config["features"]:
        raise ValueError("Input feature count differs from the saved model")
    if args.output.exists():
        raise FileExistsError("Prediction output already exists; choose a new path")
    config.update(accelerator=args.accelerator, precision=args.precision)
    # The modules are reconstructed explicitly; checkpoints are local run files.
    model = SelfSNE.load_from_checkpoint(
        run / "last.ckpt",
        map_location="cpu",
        loss=LikelihoodRatioEstimator(divergence="bce"),
        target_encoder=model_for(config).target_encoder,
    )
    predict_model(model, config, array, args.output)
    print(f"Wrote {len(array)} ordered embeddings to {args.output}")


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("train")
    start.add_argument("--context", type=Path, required=True)
    start.add_argument("--target", type=Path, required=True)
    start.add_argument("--width", type=positive, default=128)
    start.add_argument("--embedding-dim", type=positive, default=2)
    start.add_argument("--batch-size", type=positive, default=256)
    start.add_argument("--learning-rate", type=positive_float, default=0.001)
    start.add_argument("--seed", type=int, default=42)
    for command in (start, commands.add_parser("resume")):
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--epochs", type=positive, required=True)
    infer = commands.add_parser("predict")
    infer.add_argument("--run", type=Path, required=True)
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    for command in (start, infer):
        command.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu")
        command.add_argument(
            "--precision",
            choices=["32-true", "16-mixed", "bf16-mixed"],
            default="32-true",
        )
    args = parser.parse_args()
    if args.command == "train" and args.batch_size < 2:
        parser.error("batch-size must be at least two for in-batch references")
    try:
        predict(args) if args.command == "predict" else train(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
