# CPU alpha API and checkpoint contract

Maintainer contract for the evaluation path in [acceptance.md](acceptance.md).
This does not change the development status or authorize a public release.

## Entry points and data

| Entry point | Contract |
| --- | --- |
| `SelfSNE` | A LightningModule. Supply a loss and at least one encoder. A missing context/target encoder shares the other encoder; tokenizers default to identity. |
| `Trainer.fit` | Trains on tensor batches, `(context, target)` pairs, three-item tuples with references, or dictionaries with `context`, `target`, and optional `reference`. |
| `SelfSNE.forward` | Returns differentiable embedding tensors in a dictionary. It does not aggregate batches or calculate the training objective. |
| `Trainer.predict` | Returns a list of dictionaries with NumPy arrays. `target_embedding` and `context_embedding` are enabled by default. Concatenation must preserve loader order. |
| `TensorDataLoader` | Owns a detached float32 copy of its input. Normal sampler, generator, batching, and collation options apply. |
| `NumpyDataLoader` | Converts requested rows/batches to float32 without materializing the complete array or memory map. Normal loader options and Lightning prediction reconstruction apply. |

For the supported tabular path, inputs have shape `(batch, features)` and encoder
outputs have shape `(batch, embedding_features)`. Context and target must have
matching nonempty batch dimensions. Loss input embedding shapes must match.
Explicit reference inputs have shape `(batch, samples, features)`, with at least
one sample; they pass through the target encoder/head.

In-batch training requires at least two pairs. Choose a batch division that does
not leave a singleton, or drop the final training batch. Embedding-only prediction
accepts a final singleton. Logits use batch-local references: a singleton needs
explicit references, and negative-logit widths can vary between batches. The
alpha gate promises ordered embedding output, not a global all-pairs logit matrix.

## Checkpoint reconstruction

Architecture modules and the loss object are deliberately excluded from Lightning
hyperparameters. Checkpoints store weights and optimizer/scheduler state, not an
executable architecture specification. Restoration requires the original encoder,
tokenizers, heads, and loss settings. Missing loss/encoder inputs fail clearly.
Shape-incompatible weights fail strict state loading. Changing an activation or
other architecture setting with identical parameter shapes is still the caller's
responsibility; parameter shapes alone cannot establish equivalence.

New full Lightning checkpoints have a `selfsne` metadata block with format version
1 and, for built-in loss configurations, a loss signature. The signature records
kernel, kernel scale, divergence, baseline type, and whether positives are also
negatives. Equivalent registered aliases are canonicalized. Changed settings or
an unsupported format version fail before state loading. Constant-baseline values
and momentum buffers are restored from the state dictionary.

Custom kernels, divergences, baseline modules, and loss subclasses do not receive
a complete configuration guarantee; their signature is absent. Their configuration
must be reconstructed and checked by the caller. Raw `load_state_dict` operations
also bypass the full-checkpoint metadata checks.

Historical checkpoints without metadata retain strict weight loading, but cannot
verify original loss settings. They are outside the alpha compatibility promise.
There is no automatic migration or inference of missing state.

## Resume boundary

The acceptance workflow compares uninterrupted training with an epoch-boundary
checkpoint resume, including model weights, AdamW buffers, scheduler step, and
learning rate. Both runs use the same input order, batch size, architecture, loss,
and total schedule duration. Exact replay of shuffled-loader state or mid-epoch
positions is not promised. Restore evaluation mode before direct inference;
Lightning handles evaluation mode during prediction.

## Historical repairs relevant to restoration

| Change | Compatibility consequence |
| --- | --- |
| Queue occupancy is now persistent | Older queue states lack `_queue_size`; strict loading reports the missing key. Valid occupancy cannot be inferred from zeros. Rebuild the queue or migrate only when the original count is known. |
| `TCN3d` now honors `residual=False` | Earlier code always used skip connections. Reconstruct the older skip architecture with `residual=True`; corrected causal/shift behavior can require retraining. |
| Multihead batch normalization now covers heads and features | Old normalization parameter/buffer shapes can differ. Do not discard mismatches with non-strict loading. |
| Student-t kernels now honor non-unit scale | Older implementations always used unit scale. Reproducing those scores requires unit scale; corrected non-unit settings need revalidation. |
| Lightning floor is now 2.4 | Clean older Lightning environments can fail importing legacy `pkg_resources`. The floor is tested with Python 3.10 and the dependency constraints. |

## Acceptance evidence

`tests/test_checkpoint_contract.py` exercises mismatch rejection, equivalent
aliases, historical metadata-free loading, custom configuration handling, and
both supported prediction loaders. The representative acceptance test exercises
the complete new-checkpoint workflow on the current and minimum stacks. Only a
candidate commit whose CI passes both contracts satisfies the CPU alpha gate.
