# Synthetic context validation

These are maintainer validation workflows during refactoring, not a public usage
API or evidence of successful embedding geometry. Run from an existing development
checkout/environment. No divergence sweeps are performed.

```bash
.venv-cuda/bin/python scripts/synthetic_workflow.py train \
  --example swiss-roll --run ~/selfsne-runs/swiss-context \
  --samples 1024 --batch-size 64 --epochs 2 --accelerator gpu

.venv-cuda/bin/python scripts/synthetic_workflow.py train \
  --example von-mises --run ~/selfsne-runs/von-mises-context \
  --samples 256 --batch-size 32 --epochs 2 --accelerator gpu

.venv-cuda/bin/python scripts/synthetic_workflow.py resume \
  --run ~/selfsne-runs/von-mises-context --epochs 4
```

The pip CUDA environment `.venv-cuda-pip` works with the same commands. Use
`--accelerator cpu` for a small CPU check. GPU requests fail if CUDA is unavailable.
GPU precision options are `32-true` (default), `16-mixed`, and `bf16-mixed`.
Float16 uses dynamic loss scaling starting at 1024 (`--amp-initial-scale`), rather
than 65536, to reduce calibration skips in short validation runs. This change
still awaits hardware confirmation; the latest expanded CUDA run passed 27/28
source cases and stopped at Swiss-roll float16 before wheel testing. See the
[CUDA validation record](cuda.md#current-validation-status). The
scaler state is restored from checkpoints. The finite-step acceptance requirement
is unchanged: finite optimizer steps must outnumber safely skipped overflow steps.
`status.json` reports both counts and the final AMP scale, including on failure.
The six marked CUDA cases run automatically in the existing source/wheel CUDA
acceptance commands; skipped cases on CPU do not validate GPU execution.

## Swiss roll

Each clean point produces context `(views, 3)` and a single clean target `(1, 3)`.
Independent Gaussian noise is regenerated per epoch, with per-view standard
deviation `sqrt(3 * views)`, matching the supplied reference. Thus the mean noise
has variance **3 per coordinate**, expected squared Euclidean norm 9. It does not
have unit norm. This fixes the reference's comment, without changing its formula.

The encoder and embedding head are shared between context and target: input
normalization, linear projection, a DyT transformer with one output latent token,
and a two-dimensional head by default. No positional embeddings are used for this
unordered set. Context-token permutation invariance is tested in evaluation mode;
equivariance to changes in noise or view count is not established. Ordered exports
embed the clean targets. Labels are the Swiss-roll intrinsic parameter.

## Von Mises random walk

Each trajectory has independently sampled `mu ~ Uniform(-pi, pi)` and
`log(kappa) ~ Uniform(-1, 1)`. Unit steps integrate independent von Mises turns.
Contiguous past/future windows form positive pairs. Trajectories are split 75/25
**before training**; held-out trajectories never supply positive or negative
training examples. Arithmetic window indexing avoids a Python tuple per window.
The final incomplete training batch is dropped; export includes every trajectory.

The shared encoder differences positions **before** zero padding displacements
to a chunk boundary, then applies input normalization, strided convolution,
sinusoidal positions and a transformer with two output latent tokens. One token
feeds the embedding head and the other the conditional embedding baseline. This
uses the maintained transformer/loss implementations rather than copying the
reference's custom attention stack or its misleadingly named `info_nce_loss`:
that function actually used forward KL with a Cauchy embedding baseline.
The default divergence is `kld`; `jsd` and `rkld` are selectable for one run.
Constant learning rate remains the CLI default. Opt into a fixed schedule at
creation with `--schedule-epochs 100 --warmup-epochs 10`: linear warmup followed
by cosine decay, measured in optimizer steps from complete training batches.
The horizon is saved independently of the requested stopping epoch; resuming
restores scheduler state and cannot extend beyond that horizon. Old runs without
these settings retain their constant learning rate.

Exports embed each complete trajectory in original generation order. For observed
relative turns `theta`, sufficient statistics at fixed sample count are
`(sum(cos(theta)), sum(sin(theta)))`. We compute turns between successive observed
displacements, omitting any assumed initial-heading turn. Mean statistics divide
these sums by `trajectory_length - 2`. Their population expectation is
`I1(kappa)/I0(kappa) * (cos(mu), sin(mu))`.

`evaluation.json` reports held-out linear-decoder R² for each sample-mean statistic
and population-moment coordinate. These circular targets avoid a discontinuity
at `mu = -pi/pi`; they are not a claim that the embedding is sufficient. Short
validation runs can have poor or negative R². Broader geometry/parameter recovery
experiments come after software and hardware acceptance.

## Recovery and outputs

Each new run requires an unused directory. Configuration and dataset seeds are
saved in `config.json`; resume rejects configuration changes and requires a larger
total epoch count. `last.ckpt` includes optimizer state and configuration. Load
only trusted local checkpoints. Epoch-specific sampling and training seeds make
epoch-boundary recovery reproducible on the same software/device stack. Mid-epoch
recovery and reproducibility across versions/devices are not promised.

`embeddings.npy` is streamed in row order, including the short final batch, and
atomically replaced after successful export. `labels.npy` stores intrinsic Swiss
roll labels or `(mu, kappa)` for walks; `sufficient-statistics.npy` stores walk
statistics. `status.json` records pass/fail and the last finite objective. Resume
refreshes these outputs. A failed run may leave earlier exports; always inspect
status before using them. Tests check finite objectives/gradients/updates, shapes,
noise scaling, window boundaries, checkpoint equivalence and export row order.

## Swiss-roll maintainer notebook

[`notebooks/swiss_roll.ipynb`](../../notebooks/swiss_roll.ipynb) provides a
CPU-first training example in an existing development environment. It calls this workflow
with the notebook kernel's Python executable, trains ten epochs, resumes to 100
total epochs with ten warmup epochs and cosine decay, reconstructs the saved model, verifies ordered clean-target exports,
and optionally plots the data and embeddings when Matplotlib is available.
Outputs go to a newly created temporary run directory printed by the notebook;
copy that directory elsewhere if you need to retain it. Run cells top to bottom
from the checkout root or its `notebooks` directory. Start a fresh run when
re-executing the configuration cell; individual train/resume cells are not
idempotent, because existing runs and completed epoch requests are rejected.

The notebook contains no package installers, automatic device selection,
divergence sweeps, or precomputed experiment outputs. Its default uses 2,049 points and batch size 128; inspect the resulting geometry
without assuming unrolling or equivariance. Set `SELFSNE_NOTEBOOK_SMOKE=1` only
for a three-epoch software check on 129 points.
Its code cells execute with that explicit smoke configuration in CPU regression tests, including environments without
Matplotlib; plotting is checked when the maintainer plot extra is present.
