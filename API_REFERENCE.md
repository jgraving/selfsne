# selfsne

Self-Supervised Noise Embeddings.

## Quick start

```python
from selfsne import SelfSNE
from selfsne.losses import LikelihoodRatioEstimator
from torch import nn

model = SelfSNE(
    loss=LikelihoodRatioEstimator(divergence="bce"),
    target_encoder=nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2)),
)
```

## Public API

### Core

- `SelfSNE` — Main `LightningModule` for training and prediction.
- `SelfSNE.get_lr_scheduler` — Factory for warmup + cosine annealing schedules.
- `SelfSNE.visualize_lr_schedule` — Plot a learning rate schedule.

### Losses

- `LikelihoodRatioEstimator` — Configurable loss with kernel, divergence, and baseline.
- `classifier_metrics` — Compute classification metrics from logits.

### Kernels

- `ROWWISE_KERNELS` — Dictionary of row-wise kernel functions.
- `PAIRWISE_KERNELS` — Dictionary of pairwise kernel functions.
- Individual kernels: `normal`, `studentt`, `cauchy`, `inverse`, `laplace`, `von_mises`, `wrapped_cauchy`, `inner_product`, and their pairwise variants.

### Divergences

- `DIVERGENCES` — Dictionary of divergence functions.
- `GENERATORS` — Dictionary of pointwise generator functions.
- Individual divergences: `binary_cross_entropy`, `jensen_shannon_divergence`, `kullback_leibler_divergence`, `reverse_kullback_leibler_divergence`, `wasserstein`, `density_ratio`, `pearson_chi_sq`, `squared_hellinger`, `jeffreys_divergence`, `squared_le_cam_distance`, `neymann_divergence`, `softened_reverse_kullback_leibler_divergence`.
- `BaseNCE`, `ForwardNCE`, `ReverseNCE`, `AlphaNCE`, `AlphaJS`, `InterpolateDivergences`.

### Baselines

- `BASELINES` — Dictionary of baseline classes.
- Individual baselines: `BatchBaseline`, `BatchConditionalBaseline`, `MomentumBaseline`, `ReverseBatchBaseline`, `ReverseBatchConditionalBaseline`, `ReverseMomentumBaseline`, `ParametricBaseline`, `ConstantBaseline`, `ParametricConditionalBaseline` (alias `ConditionalBaseline`), `EmbeddingBaseline`, `AdditiveBaseline`, `LogAdditiveBaseline`.

### Neighbors

- `NearestNeighborSampler` — KNN sampling with FIFO queue.
- `knn_sampler` — Direct KNN sampling function.
- `Queue` — FIFO queue for embeddings.
- `IndexedDataset` — Dataset wrapper returning (data, index) pairs.
- `IndexParser` — Extract index from batch.

### Data

- `PairedDataset` — Pair two datasets together.
- `TensorDataLoader` — DataLoader over float32 tensors.
- `NumpyDataLoader` — DataLoader over NumPy arrays/memory maps.
- `CorrelatedGaussianDataLoader` — DataLoader that regenerates samples each epoch.
- `correlated_gaussian_sampler` — Generate correlated Gaussian pairs.
- `MNIST` — MNIST dataset loader.
- `Lorenz` — Lorenz attractor dataset.
- `LorenzRandomProjection` — Random projection of Lorenz attractor.
- `MemMapSequence`, `MemMap` — Memory-mapped array datasets.
- `HDF5`, `HDF5Sequence` — HDF5 dataset loaders.

### Prior

- `NormalPrior` — Normal distribution prior.
- `MixturePrior` — Mixture density prior with watershed optimization.

### Transformers

- `TransformerEncoder` — Transformer encoder with latent cross-attention.

### Noise

- `Noise` — Base noise augmentation module.
- `Normal` — Gaussian noise.
- `LogNormalSoftmax` — Log-normal noise with softmax.
- `GumbelSoftmax` — Gumbel-softmax noise.
- `ConditionalDropout` — Conditional dropout.

### Plot

- `PCAAlign` — PCA transformer with sign flipping.
- `align_embeddings` — Rotate embeddings to match a reference.
- `equalize_axes_limits` — Equalize axis ranges in a plot.
- `EmbeddingDensity` — Compute 2D embedding density.
- `density_colormaps` — Create density colormaps.
- `highest_density_contours` — Extract highest density contours.

### Callbacks

- `TotalStepsProgressBar` — Rich progress bar tracking global training steps.

### NN subpackage

- `Attention`, `SelfAttention`, `CrossAttention`, `LatentCrossAttention` — Attention mechanisms.
- `Conv1d`, `CausalConv1d`, `PadShift`, `TCN`, `TCN2d`, `TCN3d`, `ResNet2d` — Temporal convolutions.
- `PositionEmbedding2d`, `PositionEmbedding1d`, `SinusoidalPositionalEmbedding`, `RotaryPositionalEmbedding`, `PatchEmbedding1d`, `PatchEmbedding2d` — Positional embeddings.
- `init_linear`, `lecun_normal_` — Linear layer initialization.
- `Residual`, `ParametricResidual`, `Residual1d`, `Residual2d`, `Lambda`, `StopGradient` — Layer utilities.
- `MLP`, `MultiheadLinear`, `MultiheadMLP`, `MultiheadEncoder`, `Multilinear`, `GatedMLP` — Linear layers and MLPs.
- `ImageNetNorm`, `InputNorm1d`, `InputNorm2d`, `InputNorm3d`, `DyT`, `PreDyT`, `PreLayerNorm`, `PreRMSNorm` — Normalization.
- `TokenSampler` — Token sampling module.

### Utils

- `inverse_softplus`, `stop_gradient`, `set_grad`, `enable_grad`, `disable_grad`
- `random_sample_columns`, `shuffle_rows`
- `reassign_labels`
- `straight_through_estimator`
- `log_interpolate`, `log_lerp`
- `logmeanexp`
- `radicant_mean`, `radicant_add`
- `off_diagonal`, `remove_diagonal`
- `set_weight_decay`, `add_model_params`
- `split_logits`
- `concat_dicts`
- `config_to_filename`
- `sort_by_variance`
- `Trainer` — Custom Trainer with predict aggregation.
- `EpochOnlyProgressBar` — Progress bar callback.
