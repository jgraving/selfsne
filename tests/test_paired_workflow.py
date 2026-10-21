import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import selfsne


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paired_workflow.py"


def invoke(*arguments):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, arguments)],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    [
                        str(Path(selfsne.__file__).resolve().parent.parent),
                        os.environ.get("PYTHONPATH"),
                    ],
                )
            ),
        },
    )


def test_paired_workflow_trains_resumes_and_writes_ordered_predictions(tmp_path):
    datasets = pytest.importorskip("sklearn.datasets")
    context = datasets.load_digits().data[:33].astype(np.float32) / 16
    target = context + np.float32(0.01)
    x, y = tmp_path / "context.npy", tmp_path / "target.npy"
    np.save(x, context)
    np.save(y, target)
    run = tmp_path / "resumed"
    options = ["--context", x, "--target", y, "--batch-size", 16, "--width", 16]
    first = invoke("train", *options, "--run", run, "--epochs", 1)
    assert first.returncode == 0, first.stderr
    assert (run / "last.ckpt").exists()
    second = invoke("resume", "--run", run, "--epochs", 2)
    assert second.returncode == 0, second.stderr
    report = json.loads((run / "run.json").read_text())
    assert report["status"] == "passed" and report["epochs_completed"] == 2
    assert report["training_rows_per_epoch"] == 32
    assert report["embedding_shape"] == [33, 2]
    actual = np.load(run / "target-embeddings.npy")
    assert np.isfinite(actual).all() and (actual.std(0) > 1e-5).all()
    prediction = tmp_path / "prediction.npy"
    predicted = invoke("predict", "--run", run, "--input", y, "--output", prediction)
    assert predicted.returncode == 0, predicted.stderr
    np.testing.assert_allclose(np.load(prediction), actual, rtol=1e-6, atol=1e-6)
    continuous = tmp_path / "continuous"
    full = invoke("train", *options, "--run", continuous, "--epochs", 2)
    assert full.returncode == 0, full.stderr
    np.testing.assert_allclose(
        np.load(continuous / "target-embeddings.npy"), actual, rtol=1e-6, atol=1e-6
    )
    assert not list(run.glob("*.partial"))
    # Reusing an existing output and resuming with changed data both fail safely.
    overwrite = invoke("predict", "--run", run, "--input", y, "--output", prediction)
    assert overwrite.returncode != 0 and "already exists" in overwrite.stderr
    np.testing.assert_array_equal(np.load(prediction), actual)
    target[0, 0] += 1
    np.save(y, target)
    changed = invoke("resume", "--run", run, "--epochs", 3)
    assert changed.returncode != 0 and "inputs changed" in changed.stderr


def test_workflow_rejects_nonfinite_inputs_before_creating_run(tmp_path):
    values = np.ones((4, 3), dtype=np.float32)
    values[0, 0] = np.nan
    path = tmp_path / "input.npy"
    np.save(path, values)
    run = tmp_path / "run"
    result = invoke(
        "train", "--context", path, "--target", path, "--run", run, "--epochs", 1
    )
    assert result.returncode != 0 and "finite" in result.stderr
    assert not run.exists()
