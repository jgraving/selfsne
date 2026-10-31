import pytest

try:
    import rich
    has_rich = True
except ImportError:
    has_rich = False

pytestmark = pytest.mark.skipif(not has_rich, reason="rich not installed")

from selfsne.callbacks import TotalStepsProgressBar


def test_total_steps_progress_bar_instantiation():
    cb = TotalStepsProgressBar()
    assert cb is not None
    assert hasattr(cb, "progress")
    assert cb.train_progress_bar_id is None


def test_total_steps_progress_bar_inherits():
    from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
    cb = TotalStepsProgressBar()
    assert isinstance(cb, RichProgressBar)
