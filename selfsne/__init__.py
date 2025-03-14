from selfsne.selfsne import SelfSNE

__version__ = "0.0.2.dev"

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*epoch parameter.*deprecated.*",
    module=r"torch\.optim\.lr_scheduler",
)
