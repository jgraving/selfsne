"""Notebook-derived workflow contracts, including optional real CUDA acceptance."""

import importlib.util
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import selfsne


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def workflow(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "synthetic_workflow", SCRIPTS / "synthetic_workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(example):
    return dict(
        example=example,
        samples=24,
        batch_size=8,
        views=5,
        width=16,
        depth=1,
        embedding_dim=2,
        chunk_size=4,
        trajectory_length=25,
        context_length=9,
        target_length=5,
        seed=42,
        learning_rate=3e-4,
        divergence="kld",
        accelerator="cpu",
        precision="32-true",
    )


def invoke(*args):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "synthetic_workflow.py"), *map(str, args)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": str(Path(selfsne.__file__).resolve().parent.parent),
        },
    )
    assert result.returncode == 0, result.stderr
    return result


def test_swiss_noise_and_clean_target_contract(workflow):
    c = config("swiss-roll")
    c.update(samples=4096, batch_size=4096, views=10)
    data = torch.zeros(4096, 3)
    context, target = next(iter(workflow.Batches(c, data, np.arange(len(data)))))
    assert context.shape == (4096, 10, 3) and target.shape == (4096, 1, 3)
    assert torch.count_nonzero(target) == 0
    assert context.var().item() == pytest.approx(30, rel=0.03)
    assert context.mean(1).var().item() == pytest.approx(3, rel=0.04)


def test_walk_windows_and_padding_preserve_displacements(workflow):
    c = config("von-mises")
    data, _ = workflow.generate(c)
    train, test = workflow.split_ids(c)
    assert not set(train) & set(test)
    batches = workflow.Batches(c, data, train)
    context, target = next(iter(batches))
    assert context.shape == (8, 9, 2) and target.shape == (8, 5, 2)
    # Every sampled pair is contiguous within one training trajectory.
    for x, y in zip(context, target):
        joined = torch.cat((x, y))
        assert any(
            torch.equal(joined, data[i, start : start + 14])
            for i in train
            for start in range(12)
        )
    encoder = workflow.ContextEncoder(c)
    positions = torch.tensor([[[100.0, 100.0], [101.0, 100.0], [101.0, 101.0]]])
    actual = encoder.preprocess(positions)
    torch.testing.assert_close(
        actual, torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
    )
    torch.testing.assert_close(actual, encoder.preprocess(positions + 200))
    # No spurious turn to/from zero; translation/rotation leave relative turns unchanged.
    np.testing.assert_allclose(
        workflow.circular_statistics(positions), [[0, 1]], atol=1e-12
    )
    rotated = torch.stack((-positions[..., 1], positions[..., 0]), -1)
    np.testing.assert_allclose(
        workflow.circular_statistics(rotated + 200), [[0, 1]], atol=1e-12
    )


@pytest.mark.parametrize("example", ["swiss-roll", "von-mises"])
def test_models_have_finite_updates_and_swiss_permutation_invariance(workflow, example):
    c = config(example)
    data, _ = workflow.generate(c)
    context, target = next(iter(workflow.Batches(c, data, np.arange(len(data)))))
    model = workflow.model_for(c)
    assert model.context_encoder is model.target_encoder
    assert model.context_head is model.target_head
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    output = model((context, target))
    assert output["context_embedding"].shape == (8, 2)
    loss = model.loss(**output)["loss"]
    loss.backward()
    assert torch.isfinite(loss) and all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    torch.optim.Adam(model.parameters()).step()
    assert any(
        not torch.equal(before[key], value) for key, value in model.named_parameters()
    )
    model.eval()
    if example == "swiss-roll":
        with torch.no_grad():
            actual = model((context.flip(1), target))["context_embedding"]
            expected = model((context, target))["context_embedding"]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("example", ["swiss-roll", "von-mises"])
def test_resume_matches_continuous_and_exports_original_order(
    workflow, tmp_path, example
):
    options = [
        "--example",
        example,
        "--samples",
        25,
        "--batch-size",
        8,
        "--width",
        16,
        "--depth",
        1,
        "--trajectory-length",
        25,
        "--context-length",
        9,
        "--target-length",
        5,
        "--chunk-size",
        4,
    ]
    resumed, full = tmp_path / "resumed", tmp_path / "full"
    invoke("train", *options, "--run", resumed, "--epochs", 1)
    invoke("resume", "--run", resumed, "--epochs", 2)
    invoke("train", *options, "--run", full, "--epochs", 2)
    actual = np.load(resumed / "embeddings.npy")
    np.testing.assert_allclose(
        actual, np.load(full / "embeddings.npy"), rtol=1e-6, atol=1e-6
    )
    c = json.loads((resumed / "config.json").read_text())
    model = workflow.model_for(c).eval()
    state = torch.load(resumed / "last.ckpt", weights_only=False, map_location="cpu")
    assert state["synthetic_epochs_completed"] == 2
    model.load_state_dict(state["state_dict"])
    data, labels = workflow.generate(c)
    with torch.inference_mode():
        values = data if example == "von-mises" else data[:, None]
        expected = model.target_head(model.target_encoder(values)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(np.load(resumed / "labels.npy"), labels)
    assert json.loads((resumed / "status.json").read_text())["status"] == "passed"
    if example == "von-mises":
        report = json.loads((resumed / "evaluation.json").read_text())
        assert report["split"] == "trajectory" and report["heldout_trajectories"] == 7
    args = argparse.Namespace(command="resume", run=resumed, epochs=2)
    with pytest.raises(ValueError, match="exceed completed"):
        workflow.execute(args)
    c["seed"] += 1
    (resumed / "config.json").write_text(json.dumps(c))
    args.epochs = 3
    with pytest.raises(ValueError, match="Configuration changed"):
        workflow.execute(args)


@pytest.mark.cuda
@pytest.mark.parametrize("example", ["swiss-roll", "von-mises"])
@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
def test_synthetic_cuda_training_resume_export(tmp_path, example, precision):
    if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    run = tmp_path / "run"
    invoke(
        "train",
        "--example",
        example,
        "--samples",
        24,
        "--batch-size",
        8,
        "--width",
        16,
        "--depth",
        1,
        "--trajectory-length",
        25,
        "--context-length",
        9,
        "--target-length",
        5,
        "--chunk-size",
        4,
        "--accelerator",
        "gpu",
        "--precision",
        precision,
        "--run",
        run,
    )
    invoke("resume", "--run", run, "--epochs", 2)
    report = json.loads((run / "status.json").read_text())
    assert report["cuda_validated"] and report["status"] == "passed"
    assert np.isfinite(np.load(run / "embeddings.npy")).all()
