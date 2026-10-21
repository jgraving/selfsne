# CUDA development and acceptance

CUDA execution is required before calling the software GPU-ready. The completed
CPU gate remains a regression baseline; it is not evidence of CUDA correctness.
The first CUDA target is one NVIDIA GPU on Linux, Python 3.12, float32 and
float16 mixed precision. Bfloat16 is checked when the device supports it.
Multi-GPU/distributed training and MPS are separate, unverified targets.

## Developer installation

### Ubuntu bootstrap

For a native Ubuntu NVIDIA machine, run from the checkout as your normal user:

```sh
bash scripts/test-cuda-ubuntu.sh --install-driver
```

The script detects the GPU and running driver, downloads uv and Python 3.12 into
`.cuda-tools/`, creates `.venv-cuda/`, selects PyTorch with `--torch-backend=auto`,
and executes a real CUDA kernel before running source and built-wheel validation.
CUDA dependency resolution uses `--no-sources` to bypass the project's CPU-only
`tool.uv.sources` entry; selecting a backend alone does not override that entry.
Driver 535 uses the explicitly CUDA-tagged `torch==2.5.1+cu121` build rather than
automatic selection, which can fall back to CPU on that driver. Override backend
selection with `--torch-backend=cu121` when needed. A CPU-only wheel is a package
selection failure and never authorizes a driver change, even with
`--install-driver`.
It restores the editable source installation after wheel validation, including on
failure. Python packages, managed Python, download caches, and temporary files
stay inside the checkout; shell profiles and global Python installs are unchanged.
Reports are `.cuda-tools/reports/cuda-source.xml` and `cuda-wheel.xml`.

`--install-driver` explicitly permits system changes with sudo if the driver is
unavailable or CUDA cannot initialize: update APT metadata, install
`ubuntu-drivers-common`, show detected/recommended drivers, then run
`ubuntu-drivers install`. A working CUDA setup reuses its driver. Omitting this
option permits only detection and project-local installation. Driver packages
are system-wide; no standalone CUDA toolkit or third-party repository is added.
Driver installation stops with exit code 10: reboot and rerun the same command.
Complete Secure Boot key enrollment if requested during reboot. An unsupported
older GPU may still require a different PyTorch build; a driver update is not a
guarantee of compatibility. Kernel execution failures after CUDA initialization
stop for diagnosis rather than changing the driver. Dependency-download failures
do not trigger driver installation. WSL is rejected because its driver belongs
to the Windows host.

For a machine without Git or a downloader, first install those Ubuntu utilities
and clone the repository:

```sh
sudo apt-get update
sudo apt-get install -y git curl
git clone https://github.com/jgraving/selfsne.git
cd selfsne
bash scripts/test-cuda-ubuntu.sh --install-driver
```

Bootstrap orchestration is covered with mocked commands in CPU CI. CUDA source
and wheel execution has passed on the configuration recorded below. System
driver installation remains untested; that run reused its existing driver.

### Manual environments

Use a separate environment for CUDA development. `uv sync` follows the project's
CPU lockfile; do not use it to maintain the CUDA environment. PyTorch accelerator
selection is done explicitly through the uv pip interface or PyTorch's pip index.
These are maintainer validation commands, not a public installation walkthrough.

With uv, from the checkout:

```sh
uv venv --python 3.12 .venv-cuda
uv pip install --no-sources --python .venv-cuda/bin/python --torch-backend=cu130 -e '.[dev,data]'
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
selector. The validation workflow also offers `cu121`, `cu126`, and `cu128`; a passing run
only validates its recorded backend, package versions, and device. uv can select
a backend automatically with `--torch-backend=auto`, but the validation workflow
uses an explicit backend for identifiable results. A CUDA wheel installation
without working hardware/driver access does not pass `--require-cuda`.

References: [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/)
and [PyTorch installation selector](https://pytorch.org/get-started/locally/).
The CUDA 12.1 / PyTorch 2.5.1 option follows the
[official previous-version wheels](https://pytorch.org/get-started/previous-versions/)
and [NVIDIA's CUDA 12.1 driver requirements](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/).

## Execution gate

Hardware validation can run locally with the Ubuntu bootstrap or through the
manual `CUDA validation` workflow. The workflow targets an existing self-hosted
Linux x64 runner carrying the `cuda` label. It does not provision hardware and is
not triggered by pull requests. Validate each installer separately for the
selected backend; passing the uv path does not establish the pip installation.

The suite must execute on real CUDA hardware, not merely collect or skip tests:

- Float32 loss values and input/reference gradients agree with CPU across all
  built-in kernel families and both in-batch and explicit references.
- Device-local random sampling, queue state transfer through CPU checkpoints,
  and neighbor indices remain on the intended device.
- A roughly million-parameter MLP trains in float32 and float16 mixed precision
  on 28,752 augmented digit vectors. This scales a real-data workload; it is not
  a new independent dataset or an embedding-quality benchmark.
- Training losses, parameters, and predictions stay finite; parameters update
  and embeddings retain spread. Nonfinite gradients are allowed only when AMP
  reduces its loss scale and skips the update with every parameter unchanged.
  Finite optimizer steps must outnumber skipped steps; bfloat16 and float32 have
  no loss scaler and must always have finite gradients. Reports retain both counts.
  Low-precision predictions export float32 NumPy arrays. CPU/GPU checkpoint
  predictions are compared in float32; restored GPU predictions are compared at
  the original training precision. Cross-precision prediction error is reported
  separately and is not a checkpoint equivalence criterion.
- CUDA-trained checkpoints load on CPU and return to CUDA with matching outputs;
  fixed-order epoch resume reproduces uninterrupted model and optimizer state.
- Reports record GPU identity, PyTorch/CUDA runtime, precision, sample/model size,
  elapsed time, throughput, peak allocated/reserved memory, and prediction error.
  Throughput includes acceptance checks and is not a standalone performance claim.
- Source and installed-wheel runs both pass and retain their JUnit reports. A
  device without bfloat16 support may skip only the bfloat16 workload explicitly.

## Current validation status

The uv source and installed-wheel path passed on real hardware at commit
[`3139bbd`](https://github.com/jgraving/selfsne/commit/3139bbd41447ac10ac91ae24762518c3822c706c).
Evidence is a user-supplied terminal transcript reviewed on 2026-10-21; this is a
local hardware run, not a GPU execution in hosted CI.

| Configuration | Recorded value |
| --- | --- |
| OS | Ubuntu 20.04.6 LTS, Linux x86_64 |
| Python | 3.12.15 |
| PyTorch / bundled CUDA | 2.5.1+cu121 / 12.1 |
| NVIDIA driver | 535.183.01, reused without driver installation |
| GPU | RTX A5000, 24 GB; one device used for training |
| Installation | uv via `scripts/test-cuda-ubuntu.sh` |
| Source acceptance | 22 passed, no failures or skips |
| Installed-wheel acceptance | 22 passed, no failures or skips |

Float32, float16 mixed precision, and bfloat16 mixed precision all passed,
including checkpoint state equality, CPU/GPU prediction comparisons, matched-
precision GPU prediction restoration, and epoch-boundary resume. Both detected
GPUs passed the initial kernel probe; training used a single device and does not
establish distributed or multi-GPU training. After wheel validation the script
successfully restored the editable source installation.

The run generated `.cuda-tools/reports/cuda-source.xml` and `cuda-wheel.xml` on
the test machine. Those XML files were not supplied with the transcript, so no
numerical memory, throughput, or AMP skipped-step counts are claimed here.
The pip CUDA installer, system driver installation, and other GPU/backend
combinations remain unverified. CPU CI continues to exercise mixed-precision
regressions while skipping hardware-only cases; CPU CI alone is not CUDA evidence.

Built-in likelihood kernels compute distances/reductions in float32 under
mixed precision. Custom kernels retain their own precision behavior and require
separate validation. Full-float64 prediction dtype remains unchanged.
