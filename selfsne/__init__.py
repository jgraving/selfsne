"""Self-Supervised Noise Embeddings."""

from selfsne.selfsne import SelfSNE, get_lr_scheduler, visualize_lr_schedule

__version__ = "0.0.2.dev0"
__all__ = [
    "SelfSNE",
    "__version__",
    "get_lr_scheduler",
    "visualize_lr_schedule",
]
