# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-31

### Added
- Updated documentation and examples
- Public code exports for core modules
- Tests covering key functionality

## [Unreleased]

### Added

- `SelfSNE` LightningModule with full training, validation, test, and predict
  lifecycle. Supports context/target encoders, tokenizers, heads, baselines,
  and learning rate scheduling (warmup + cosine annealing).
- `LikelihoodRatioEstimator` loss with configurable kernels, divergences, and
  baselines. Includes metric collection and classifier metrics.
- 10 row-wise and 13 pairwise kernel functions: euclidean/normal, student-t,
  Cauchy, inverse, Laplace, von Mises-Fisher, wrapped Cauchy, inner product.
- 12+ divergence functions: KL, reverse KL, Jensen-Shannon, BCE, Wasserstein,
  density ratio, Pearson chi-squared, Hellinger, Neymann, Le Cam, softened
  reverse KL, and alpha-divergence family.
- 9 baseline estimators: batch, batch conditional, momentum, reverse variants,
  parametric, constant, embedding. Composable additive and log-additive
  baselines.
- K-nearest neighbor sampling with FIFO queue for negative sample generation.
- Neural network subpackage (`selfsne.nn`): attention mechanisms, temporal
  convolutions (TCN/TCN2d/TCN3d), positional embeddings (sinusoidal,
  rotary, 1d/2d), patch embeddings, gated MLPs, pre-normalization wrappers.
- Data utilities: paired datasets, correlated Gaussian sampler, Lorenz
  attractor, MNIST loader, memory-mapped and HDF5 datasets.
- Noise augmentation modules: Gaussian noise, log-normal softmax, Gumbel
  softmax, conditional dropout.
- Prior models: `NormalPrior` and `MixturePrior` with watershed optimization.
- Transformer encoder with self-attention, GatedMLP, and latent
  cross-attention.
- Learning rate schedule with warmup, target, and cosine annealing phases.
- Checkpoint contract with loss configuration signature for built-in losses.
- AMP calibration fix: lowered initial loss scale from 65536 to 1024 for
  short validation runs.
- `TCN3d` now honors `residual=False` to disable skip connections.
- Multihead batch normalization covers heads and features.
- Student-t kernels honor non-unit scale.
- Queue occupancy is now persistent across checkpoints.
- `TotalStepsProgressBar` callback for global step tracking.
- `PCAAlign`, `EmbeddingDensity`, `density_colormaps`, and
  `highest_density_contours` plotting utilities.
- Synthetic workflow scripts for swiss-roll and von Mises random walk.
- Quickstart example script.

### Changed

- Lightning floor raised to 2.4.

### Fixed

- AMP overflow now safely skips updates; finite optimizer steps must
  outnumber skipped steps for acceptance.
- Checkpoint loading rejects mismatched loss configurations for new
  checkpoints.

## [0.0.1] — 2024-XX-XX

- Initial release (private).
