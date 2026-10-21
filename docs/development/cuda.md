# CUDA development and acceptance

CUDA execution is required before calling the software GPU-ready. The completed
CPU gate remains a regression baseline; it is not evidence of CUDA correctness.
The first CUDA target is one NVIDIA GPU on Linux, Python 3.12, float32 and
float16 mixed precision. Bfloat16 is checked when the device supports it.
Multi-GPU/distributed training and MPS are separate, unverified targets.

## Developer installation

Use a separate environment for CUDA development. `uv sync` follows the project's
CPU lockfile; do not use it to maintain the CUDA environment. PyTorch accelerator
selection is done explicitly through the uv pip interface or PyTorch's pip index.
These are maintainer validation commands, not a public installation walkthrough.

With uv, from the checkout:

```sh
uv venv --python 3.12 .venv-cuda
uv pip install --python .venv-cuda/bin/python --torch-backend=cu130 -e '.[dev,data]'
uv pip check --python .venv-cuda/bin/python
.venv-cuda/bin/python -m pytest -m cuda --require-cuda -q --junitxml=cuda-source.xml
```

With pip, in a separately activated Python 3.12 virtual environment:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[dev,data]'
python -m pip check
python -m pytest -m cuda --require-cuda -q --junitxml=cuda-source.xml
```

`cu130` identifies CUDA 13.0 PyTorch wheels, not a Python version. Choose a
backend compatible with the installed NVIDIA driver using the official PyTorch
selector. The validation workflow also offers `cu126` and `cu128`; a passing run
only validates its recorded backend, package versions, and device. uv can select
a backend automatically with `--torch-backend=auto`, but the validation workflow
uses an explicit backend for identifiable results. A CUDA wheel installation
without working hardware/driver access does not pass `--require-cuda`.

References: [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/)
and [PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Execution gate

The manual `CUDA validation` workflow targets an existing self-hosted Linux x64
runner carrying the `cuda` label. It does not provision hardware and is not
triggered by pull requests. Dispatch a trusted revision once such a runner is
available, once with each installer for the selected backend.

The suite must execute on real CUDA hardware, not merely collect or skip tests:

- Float32 loss values and input/reference gradients agree with CPU across all
  built-in kernel families and both in-batch and explicit references.
- Device-local random sampling, queue state transfer through CPU checkpoints,
  and neighbor indices remain on the intended device.
- A roughly million-parameter MLP trains in float32 and float16 mixed precision
  on 28,752 augmented digit vectors. This scales a real-data workload; it is not
  a new independent dataset or an embedding-quality benchmark.
- Training losses, parameters, gradients, and predictions stay finite; parameters
  update and embeddings retain spread. Low-precision predictions export float32
  NumPy arrays. CPU/GPU prediction comparisons use precision-specific tolerances.
- CUDA-trained checkpoints load on CPU and return to CUDA with matching outputs;
  fixed-order epoch resume reproduces uninterrupted model and optimizer state.
- Reports record GPU identity, PyTorch/CUDA runtime, precision, sample/model size,
  elapsed time, throughput, peak allocated/reserved memory, and prediction error.
  Throughput includes acceptance checks and is not a standalone performance claim.
- Source and installed-wheel runs both pass and retain their JUnit reports. A
  device without bfloat16 support may skip only the bfloat16 workload explicitly.

## Current validation status

CUDA hardware execution is pending. CPU CI exercises the mixed-precision safety
regressions and skips hardware-only cases. Those green CPU jobs cannot satisfy
this gate. The manual workflow must produce passing source and wheel reports on
an actual GPU before CUDA support is marked validated.

Built-in likelihood kernels compute distances/reductions in float32 under
mixed precision. Custom kernels retain their own precision behavior and require
separate validation. Full-float64 prediction dtype remains unchanged.
