# CPU alpha acceptance gate

Maintainer validation scope, not a release announcement or public usage guide.
The project remains under development. Passing this gate permits a private CPU
alpha evaluation; it does not authorize a public release. GPU readiness additionally
requires the separate [CUDA gate](cuda.md). Its uv source/wheel path has passed
on the recorded RTX A5000 configuration; the pip CUDA installer path is pending.

## Supported evaluation path

- Python 3.10 and 3.12; CPU, float32, one device.
- Linux with uv and pip; Python 3.12 portability checks on Windows and macOS.
- Direct runtime dependency floors in `constraints/minimum.txt` on Linux/Python
  3.10, plus the uv lockfile and the pip-resolved environment in the main CI matrix.
- Paired tabular observations, a shared MLP encoder, Cauchy kernel, BCE objective,
  parametric baseline, and AdamW; fit and predict through Lightning's Trainer.
- Ordered embedding prediction, matching-architecture checkpoint restoration,
  and epoch-boundary optimizer/scheduler resume with a fixed data order.

Other architectures and objectives keep their existing regression tests, but
are outside this CPU acceptance promise. CUDA and GPU mixed precision have a separate
required hardware gate. MPS,
multiple devices, other Python versions, exact replay of shuffled or mid-epoch
runs, noise transforms, plots, and callbacks are not acceptance targets.

## Blocking criteria

1. Installation and source/wheel tests pass in the existing platform matrix.
   The minimum direct runtime dependency job also passes dependency checks,
   the available regression suite, and the representative workflow.
2. A representative digits-data acceptance test trains on one split and reduces
   the fixed held-out objective by at least 5% from initialization. Labels must
   not enter training. All logged objectives, parameters, and embeddings remain
   finite; predicted embeddings have nonzero spread.
3. The workflow saves a checkpoint, resumes optimizer and scheduler state at an
   epoch boundary, and matches an uninterrupted run within stated test tolerances.
   Reconstructed-model predictions match the saved model, preserve input order,
   and include a final singleton batch.
4. Supported input/output contracts and architecture reconstruction requirements
   are recorded in maintainer documentation and exercised by tests. Incompatible
   checkpoint formats and parameter shapes must fail clearly; changed built-in
   loss settings must be rejected for new checkpoints. Historical limits and
   caller-owned architecture settings are listed in `api-contract.md`.
5. The README stays a development-only notice. No release tag, package publication,
   public walkthrough, or experimental quality claim is part of this gate.

## Completion evidence

The authoritative evidence is the CI run at the candidate commit, including
installed-wheel tests and the minimum-dependency job. The representative test
is `tests/test_cpu_acceptance.py`. It records its initial/final held-out loss,
model/optimizer resume error, prediction error,
embedding shape, and dependency versions in the `cpu_alpha_acceptance` property
of the source/wheel JUnit reports. CI retains those reports as artifacts for seven
days. Each candidate run must pass the representative test in its own environment.
A green gate means the defined workflow is ready for private evaluation, not
that every exposed module or dependency combination has been validated.
